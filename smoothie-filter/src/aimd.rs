use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

/// Configuration for the AIMD controller.
pub struct AimdConfig {
    /// Inter-token latency floor in milliseconds (derived from `floor_tps`).
    /// `None` enables derivative-based crossover detection.
    pub floor_ms: Option<f64>,
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
    /// Fractional ITL increase that triggers decrease in derivative mode.
    pub sensitivity: f64,
}

/// Phase of the derivative probe cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbePhase {
    /// Semaphore is not saturated — probing is meaningless.
    WaitingForSaturation,
    /// Accumulating baseline ITL observations at the current ceiling.
    Settling,
    /// Ceiling raised by 1 — accumulating observations to compare against baseline.
    Measuring,
}

/// Inner mutable state protected by a Mutex.
struct AimdInner {
    /// Consecutive steps where slowest stream exceeded the floor.
    over_threshold_count: u32,
    /// Consecutive steps where slowest stream was under (floor - headroom).
    under_with_headroom_count: u32,

    // --- derivative probe state ---
    /// Current phase of the probe cycle.
    probe_phase: ProbePhase,
    /// Number of ITL observations accumulated in the current phase.
    phase_observation_count: u32,
    /// Sum of ITL observations in the current phase.
    phase_itl_sum: f64,
    /// Baseline average ITL established during Settling.
    baseline_itl: f64,
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
                probe_phase: ProbePhase::WaitingForSaturation,
                phase_observation_count: 0,
                phase_itl_sum: 0.0,
                baseline_itl: 0.0,
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
    /// `active_count` is the number of currently active streams in the
    /// semaphore. In derivative mode this is used to detect saturation.
    pub fn observe(&self, max_smoothed_itl_ms: f64, active_count: u32) -> u32 {
        let Ok(mut inner) = self.inner.lock() else {
            // Mutex poisoned — return current ceiling without updating.
            return self.ceiling();
        };

        match self.config.floor_ms {
            Some(floor) => self.observe_fixed_floor(&mut inner, max_smoothed_itl_ms, floor),
            None => self.observe_derivative(&mut inner, max_smoothed_itl_ms, active_count),
        }
    }

    /// Fixed-floor AIMD logic (original behavior).
    ///
    /// Three zones:
    /// - **Over floor:** latency >= floor_ms. After `hysteresis_steps`
    ///   consecutive observations, multiply ceiling by beta (decrease).
    /// - **Under with headroom:** latency < (floor_ms - headroom_ms).
    ///   After `hysteresis_steps`, increment ceiling by 1 (increase).
    /// - **Dead zone:** between the two thresholds. Reset both counters
    ///   to prevent oscillation at equilibrium.
    fn observe_fixed_floor(
        &self,
        inner: &mut AimdInner,
        max_smoothed_itl_ms: f64,
        floor: f64,
    ) -> u32 {
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

    /// Derivative-based crossover detection probe cycle.
    ///
    /// Discovers the compute-memory crossover point by probing: increase the
    /// ceiling by 1, observe if ITL increased, and decide accordingly.
    fn observe_derivative(
        &self,
        inner: &mut AimdInner,
        max_smoothed_itl_ms: f64,
        active_count: u32,
    ) -> u32 {
        let ceiling = self.ceiling();

        match inner.probe_phase {
            ProbePhase::WaitingForSaturation => {
                // If active < ceiling - 1, the semaphore isn't saturated.
                // Raising the ceiling wouldn't change the batch size.
                if ceiling > 1 && active_count < ceiling - 1 {
                    return ceiling;
                }
                // Saturated — start settling.
                inner.probe_phase = ProbePhase::Settling;
                inner.phase_observation_count = 0;
                inner.phase_itl_sum = 0.0;
                self.observe_derivative(inner, max_smoothed_itl_ms, active_count)
            }
            ProbePhase::Settling => {
                inner.phase_observation_count += 1;
                inner.phase_itl_sum += max_smoothed_itl_ms;

                if inner.phase_observation_count >= self.config.hysteresis_steps {
                    // Baseline established.
                    inner.baseline_itl =
                        inner.phase_itl_sum / f64::from(inner.phase_observation_count);
                    inner.phase_observation_count = 0;
                    inner.phase_itl_sum = 0.0;

                    // Probe: additive increase by 1.
                    let new_ceiling = (ceiling + 1).min(self.config.ceiling_max);
                    if new_ceiling == ceiling {
                        // Already at max — can't probe higher. Stay settled.
                        inner.probe_phase = ProbePhase::WaitingForSaturation;
                    } else {
                        self.ceiling.store(new_ceiling, Ordering::Relaxed);
                        inner.probe_phase = ProbePhase::Measuring;
                    }
                }
                self.ceiling()
            }
            ProbePhase::Measuring => {
                // If we became unsaturated during measurement, the probe is
                // invalid — go back to waiting.
                if ceiling > 1 && active_count < ceiling - 1 {
                    inner.probe_phase = ProbePhase::WaitingForSaturation;
                    inner.phase_observation_count = 0;
                    inner.phase_itl_sum = 0.0;
                    return ceiling;
                }

                inner.phase_observation_count += 1;
                inner.phase_itl_sum += max_smoothed_itl_ms;

                if inner.phase_observation_count >= self.config.hysteresis_steps {
                    let measured =
                        inner.phase_itl_sum / f64::from(inner.phase_observation_count);
                    inner.phase_observation_count = 0;
                    inner.phase_itl_sum = 0.0;

                    if measured > inner.baseline_itl * (1.0 + self.config.sensitivity) {
                        // Crossed over — multiplicative decrease.
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss,
                            clippy::cast_precision_loss
                        )]
                        let new_ceiling = ((f64::from(ceiling) * self.config.beta) as u32)
                            .max(self.config.ceiling_min);
                        self.ceiling.store(new_ceiling, Ordering::Relaxed);
                    } else {
                        // Still memory-bound — keep the new ceiling as baseline.
                        inner.baseline_itl = measured;
                    }
                    inner.probe_phase = ProbePhase::Settling;
                }

                self.ceiling()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AimdConfig {
        AimdConfig {
            floor_ms: Some(100.0), // 10 tok/s
            headroom_ms: 15.0,     // increase threshold at 85ms
            beta: 0.8,
            hysteresis_steps: 3,
            ceiling_min: 1,
            ceiling_max: 64,
            sensitivity: 0.10,
        }
    }

    /// Helper: observe in fixed-floor mode with active_count == ceiling
    /// (simulates saturation so existing tests behave identically).
    fn observe_saturated(ctrl: &AimdController, itl: f64) -> u32 {
        let ceiling = ctrl.ceiling();
        ctrl.observe(itl, ceiling)
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
        observe_saturated(&ctrl, 110.0); // step 1
        assert_eq!(ctrl.ceiling(), 10, "no change before hysteresis");
        observe_saturated(&ctrl, 120.0); // step 2
        assert_eq!(ctrl.ceiling(), 10);
        observe_saturated(&ctrl, 105.0); // step 3 — triggers decrease
        assert_eq!(ctrl.ceiling(), 8, "10 * 0.8 = 8");
    }

    #[test]
    fn increase_after_hysteresis() {
        let ctrl = AimdController::new(test_config(), 5);

        // 3 consecutive under-headroom observations (< 85ms).
        observe_saturated(&ctrl, 80.0);
        observe_saturated(&ctrl, 70.0);
        observe_saturated(&ctrl, 60.0);
        assert_eq!(ctrl.ceiling(), 6, "should additive-increase by 1");
    }

    #[test]
    fn dead_zone_resets_counters() {
        let ctrl = AimdController::new(test_config(), 10);

        // 2 over-threshold, then one in dead zone.
        observe_saturated(&ctrl, 110.0);
        observe_saturated(&ctrl, 120.0);
        observe_saturated(&ctrl, 90.0); // dead zone: 85 <= 90 < 100

        // Now 3 more over-threshold: should need full 3 again.
        observe_saturated(&ctrl, 110.0);
        assert_eq!(ctrl.ceiling(), 10, "counter was reset by dead zone");
        observe_saturated(&ctrl, 110.0);
        observe_saturated(&ctrl, 110.0); // triggers
        assert_eq!(ctrl.ceiling(), 8);
    }

    #[test]
    fn clamps_to_ceiling_min() {
        let ctrl = AimdController::new(test_config(), 1);

        // Try to decrease below min.
        for _ in 0..3 {
            observe_saturated(&ctrl, 200.0);
        }
        assert_eq!(ctrl.ceiling(), 1, "should not go below ceiling_min");
    }

    #[test]
    fn clamps_to_ceiling_max() {
        let ctrl = AimdController::new(test_config(), 64);

        // Try to increase above max.
        for _ in 0..3 {
            observe_saturated(&ctrl, 50.0);
        }
        assert_eq!(ctrl.ceiling(), 64, "should not exceed ceiling_max");
    }

    #[test]
    fn multiple_decrease_rounds() {
        let ctrl = AimdController::new(test_config(), 20);

        // First round: 20 * 0.8 = 16
        for _ in 0..3 {
            observe_saturated(&ctrl, 150.0);
        }
        assert_eq!(ctrl.ceiling(), 16);

        // Second round: 16 * 0.8 = 12 (12.8 truncated)
        for _ in 0..3 {
            observe_saturated(&ctrl, 150.0);
        }
        assert_eq!(ctrl.ceiling(), 12);
    }

    #[test]
    fn interleaved_increase_decrease() {
        let ctrl = AimdController::new(test_config(), 10);

        // Increase to 11.
        for _ in 0..3 {
            observe_saturated(&ctrl, 50.0);
        }
        assert_eq!(ctrl.ceiling(), 11);

        // Decrease: 11 * 0.8 = 8 (8.8 truncated)
        for _ in 0..3 {
            observe_saturated(&ctrl, 150.0);
        }
        assert_eq!(ctrl.ceiling(), 8);
    }

    #[test]
    fn exactly_at_floor_triggers_over() {
        let ctrl = AimdController::new(test_config(), 10);

        // 100ms is exactly at the floor — should count as over.
        for _ in 0..3 {
            observe_saturated(&ctrl, 100.0);
        }
        assert_eq!(ctrl.ceiling(), 8);
    }

    #[test]
    fn exactly_at_headroom_threshold_is_dead_zone() {
        let ctrl = AimdController::new(test_config(), 10);

        // 85ms is exactly at headroom threshold — should be dead zone (not under).
        // headroom_threshold = 100 - 15 = 85, and 85 < 85 is false.
        for _ in 0..10 {
            observe_saturated(&ctrl, 85.0);
        }
        assert_eq!(ctrl.ceiling(), 10, "at threshold boundary should be dead zone");
    }

    // --- Derivative mode tests ---

    fn derivative_config() -> AimdConfig {
        AimdConfig {
            floor_ms: None,
            headroom_ms: 15.0,
            beta: 0.5,
            hysteresis_steps: 3,
            ceiling_min: 1,
            ceiling_max: 20,
            sensitivity: 0.10,
        }
    }

    #[test]
    fn derivative_waits_for_saturation() {
        let ctrl = AimdController::new(derivative_config(), 5);

        // active=2, ceiling=5 → 2 < 5-1=4 → not saturated, no change.
        for _ in 0..10 {
            ctrl.observe(50.0, 2);
        }
        assert_eq!(ctrl.ceiling(), 5, "should not change while unsaturated");
    }

    #[test]
    fn derivative_probes_upward_when_memory_bound() {
        let ctrl = AimdController::new(derivative_config(), 5);
        let active = 5; // saturated

        // Settling: 3 observations at 50ms → baseline = 50
        ctrl.observe(50.0, active);
        ctrl.observe(50.0, active);
        ctrl.observe(50.0, active);

        // After settling, ceiling should bump to 6 (probe).
        assert_eq!(ctrl.ceiling(), 6, "should probe upward after settling");

        // Measuring: 3 observations still at 50ms → no increase detected.
        ctrl.observe(50.0, active);
        ctrl.observe(50.0, active);
        ctrl.observe(50.0, active);

        // No crossover → keeps ceiling at 6, enters Settling again.
        assert_eq!(ctrl.ceiling(), 6, "should keep probed ceiling when memory-bound");
    }

    #[test]
    fn derivative_detects_crossover_and_decreases() {
        let ctrl = AimdController::new(derivative_config(), 5);
        let active = 5;

        // Settling: baseline = 50ms.
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 6, "probed to 6");

        // Measuring: ITL jumps to 60ms (20% increase > 10% sensitivity).
        for _ in 0..3 {
            ctrl.observe(60.0, active);
        }
        // 6 * 0.5 = 3
        assert_eq!(ctrl.ceiling(), 3, "should multiplicative-decrease after crossover");
    }

    #[test]
    fn derivative_clamps_to_ceiling_max() {
        let ctrl = AimdController::new(derivative_config(), 19);
        let active = 19;

        // Settling → probe from 19 to 20 (max).
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 20, "should probe to max");

        // Measuring at same ITL → no crossover → keeps 20.
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 20);

        // Next settling: already at max → cannot probe higher.
        // Should go to WaitingForSaturation.
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 20, "should stay at max");
    }

    #[test]
    fn derivative_clamps_to_ceiling_min() {
        let mut cfg = derivative_config();
        cfg.ceiling_min = 2;
        let ctrl = AimdController::new(cfg, 3);
        let active = 3;

        // Settle → probe to 4.
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 4);

        // Crossover detected: 4 * 0.5 = 2, clamped to ceiling_min=2.
        for _ in 0..3 {
            ctrl.observe(80.0, active); // 60% increase > 10% sensitivity
        }
        assert_eq!(ctrl.ceiling(), 2, "should clamp to ceiling_min");
    }

    #[test]
    fn derivative_unsaturation_during_measurement_resets() {
        let ctrl = AimdController::new(derivative_config(), 5);

        // Settle with saturation.
        for _ in 0..3 {
            ctrl.observe(50.0, 5);
        }
        assert_eq!(ctrl.ceiling(), 6, "probed to 6");

        // First measurement observation saturated.
        ctrl.observe(50.0, 6);

        // Become unsaturated mid-measurement (active=2, ceiling=6, 2 < 5).
        ctrl.observe(50.0, 2);

        // Should have reset to WaitingForSaturation. Ceiling stays at 6.
        assert_eq!(ctrl.ceiling(), 6);

        // Further unsaturated observations → no change.
        for _ in 0..10 {
            ctrl.observe(50.0, 2);
        }
        assert_eq!(ctrl.ceiling(), 6, "should stay put while unsaturated");
    }

    #[test]
    fn derivative_multiple_probe_rounds() {
        let ctrl = AimdController::new(derivative_config(), 3);
        let active = 3;

        // Round 1: settle at 3 → probe to 4, measure stable → keep 4.
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 4);
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 4);

        // Round 2: settle at 4 → probe to 5, measure stable → keep 5.
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 5);
        for _ in 0..3 {
            ctrl.observe(50.0, active);
        }
        assert_eq!(ctrl.ceiling(), 5);
    }
}
