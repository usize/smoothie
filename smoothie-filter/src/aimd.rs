use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

/// Configuration for the AIMD controller.
pub struct AimdConfig {
    /// Inter-token latency floor in milliseconds (derived from `floor_tps`).
    pub floor_ms: f64,
    /// Headroom below the floor before additive increase (ms).
    pub headroom_ms: f64,
    /// Multiplicative decrease factor.
    pub beta: f64,
    /// Consecutive steps required before acting.
    pub hysteresis_steps: u32,
    /// Minimum ceiling value.
    pub ceiling_min: u32,
    /// Maximum ceiling value.
    pub ceiling_max: u32,
}

/// Inner mutable state protected by a Mutex.
struct AimdInner {
    /// Consecutive steps where slowest stream exceeded the floor.
    over_threshold_count: u32,
    /// Consecutive steps where slowest stream was under (floor - headroom).
    under_with_headroom_count: u32,
}

/// AIMD concurrency controller.
///
/// Adapts an admission ceiling using additive-increase / multiplicative-decrease
/// based on the observed maximum inter-token latency across active streams.
///
/// The ceiling is stored as an `AtomicU32` for lock-free reads. The control
/// step (`observe`) takes a Mutex — the critical section is a few comparisons
/// and one atomic store.
pub struct AimdController {
    ceiling: AtomicU32,
    inner: Mutex<AimdInner>,
    config: AimdConfig,
}

impl AimdController {
    /// Create a new controller with the given config and initial ceiling.
    pub fn new(config: AimdConfig, initial_ceiling: u32) -> Self {
        Self {
            ceiling: AtomicU32::new(initial_ceiling),
            inner: Mutex::new(AimdInner {
                over_threshold_count: 0,
                under_with_headroom_count: 0,
            }),
            config,
        }
    }

    /// Current ceiling (lock-free read).
    pub fn ceiling(&self) -> u32 {
        self.ceiling.load(Ordering::Relaxed)
    }

    /// Feed the controller with the maximum smoothed inter-token latency
    /// across all active streams. Returns the (possibly updated) ceiling.
    ///
    /// Three zones:
    /// - **Over floor:** latency >= floor_ms. After `hysteresis_steps`
    ///   consecutive observations, multiply ceiling by beta (decrease).
    /// - **Under with headroom:** latency < (floor_ms - headroom_ms).
    ///   After `hysteresis_steps`, increment ceiling by 1 (increase).
    /// - **Dead zone:** between the two thresholds. Reset both counters
    ///   to prevent oscillation at equilibrium.
    pub fn observe(&self, max_smoothed_itl_ms: f64) -> u32 {
        let Ok(mut inner) = self.inner.lock() else {
            // Mutex poisoned — return current ceiling without updating.
            return self.ceiling();
        };

        let floor = self.config.floor_ms;
        let headroom_threshold = floor - self.config.headroom_ms;

        if max_smoothed_itl_ms >= floor {
            // Over the floor — stream is too slow.
            inner.under_with_headroom_count = 0;
            inner.over_threshold_count += 1;

            if inner.over_threshold_count >= self.config.hysteresis_steps {
                inner.over_threshold_count = 0;
                let current = self.ceiling();
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                let new_ceiling = ((f64::from(current) * self.config.beta) as u32)
                    .max(self.config.ceiling_min);
                self.ceiling.store(new_ceiling, Ordering::Relaxed);
            }
        } else if max_smoothed_itl_ms < headroom_threshold {
            // Under floor with headroom — room to grow.
            inner.over_threshold_count = 0;
            inner.under_with_headroom_count += 1;

            if inner.under_with_headroom_count >= self.config.hysteresis_steps {
                inner.under_with_headroom_count = 0;
                let current = self.ceiling();
                let new_ceiling = (current + 1).min(self.config.ceiling_max);
                self.ceiling.store(new_ceiling, Ordering::Relaxed);
            }
        } else {
            // Dead zone — at equilibrium, reset counters.
            inner.over_threshold_count = 0;
            inner.under_with_headroom_count = 0;
        }

        self.ceiling()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AimdConfig {
        AimdConfig {
            floor_ms: 100.0,   // 10 tok/s
            headroom_ms: 15.0, // increase threshold at 85ms
            beta: 0.8,
            hysteresis_steps: 3,
            ceiling_min: 1,
            ceiling_max: 64,
        }
    }

    #[test]
    fn initial_ceiling() {
        let ctrl = AimdController::new(test_config(), 32);
        assert_eq!(ctrl.ceiling(), 32);
    }

    #[test]
    fn decrease_after_hysteresis() {
        let ctrl = AimdController::new(test_config(), 10);

        // 3 consecutive over-threshold observations.
        ctrl.observe(110.0); // step 1
        assert_eq!(ctrl.ceiling(), 10, "no change before hysteresis");
        ctrl.observe(120.0); // step 2
        assert_eq!(ctrl.ceiling(), 10);
        ctrl.observe(105.0); // step 3 — triggers decrease
        assert_eq!(ctrl.ceiling(), 8, "10 * 0.8 = 8");
    }

    #[test]
    fn increase_after_hysteresis() {
        let ctrl = AimdController::new(test_config(), 5);

        // 3 consecutive under-headroom observations (< 85ms).
        ctrl.observe(80.0);
        ctrl.observe(70.0);
        ctrl.observe(60.0);
        assert_eq!(ctrl.ceiling(), 6, "should additive-increase by 1");
    }

    #[test]
    fn dead_zone_resets_counters() {
        let ctrl = AimdController::new(test_config(), 10);

        // 2 over-threshold, then one in dead zone.
        ctrl.observe(110.0);
        ctrl.observe(120.0);
        ctrl.observe(90.0); // dead zone: 85 <= 90 < 100

        // Now 3 more over-threshold: should need full 3 again.
        ctrl.observe(110.0);
        assert_eq!(ctrl.ceiling(), 10, "counter was reset by dead zone");
        ctrl.observe(110.0);
        ctrl.observe(110.0); // triggers
        assert_eq!(ctrl.ceiling(), 8);
    }

    #[test]
    fn clamps_to_ceiling_min() {
        let ctrl = AimdController::new(test_config(), 1);

        // Try to decrease below min.
        for _ in 0..3 {
            ctrl.observe(200.0);
        }
        assert_eq!(ctrl.ceiling(), 1, "should not go below ceiling_min");
    }

    #[test]
    fn clamps_to_ceiling_max() {
        let ctrl = AimdController::new(test_config(), 64);

        // Try to increase above max.
        for _ in 0..3 {
            ctrl.observe(50.0);
        }
        assert_eq!(ctrl.ceiling(), 64, "should not exceed ceiling_max");
    }

    #[test]
    fn multiple_decrease_rounds() {
        let ctrl = AimdController::new(test_config(), 20);

        // First round: 20 * 0.8 = 16
        for _ in 0..3 {
            ctrl.observe(150.0);
        }
        assert_eq!(ctrl.ceiling(), 16);

        // Second round: 16 * 0.8 = 12 (12.8 truncated)
        for _ in 0..3 {
            ctrl.observe(150.0);
        }
        assert_eq!(ctrl.ceiling(), 12);
    }

    #[test]
    fn interleaved_increase_decrease() {
        let ctrl = AimdController::new(test_config(), 10);

        // Increase to 11.
        for _ in 0..3 {
            ctrl.observe(50.0);
        }
        assert_eq!(ctrl.ceiling(), 11);

        // Decrease: 11 * 0.8 = 8 (8.8 truncated)
        for _ in 0..3 {
            ctrl.observe(150.0);
        }
        assert_eq!(ctrl.ceiling(), 8);
    }

    #[test]
    fn exactly_at_floor_triggers_over() {
        let ctrl = AimdController::new(test_config(), 10);

        // 100ms is exactly at the floor — should count as over.
        for _ in 0..3 {
            ctrl.observe(100.0);
        }
        assert_eq!(ctrl.ceiling(), 8);
    }

    #[test]
    fn exactly_at_headroom_threshold_is_dead_zone() {
        let ctrl = AimdController::new(test_config(), 10);

        // 85ms is exactly at headroom threshold — should be dead zone (not under).
        // headroom_threshold = 100 - 15 = 85, and 85 < 85 is false.
        for _ in 0..10 {
            ctrl.observe(85.0);
        }
        assert_eq!(ctrl.ceiling(), 10, "at threshold boundary should be dead zone");
    }
}
