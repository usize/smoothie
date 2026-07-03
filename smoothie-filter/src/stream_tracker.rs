use std::time::Instant;

/// Per-stream EWMA inter-token latency tracker.
///
/// Computes a smoothed inter-token latency using exponentially weighted
/// moving average. Stored in a shared `DashMap` so the AIMD controller
/// can read `max(smoothed_itl_ms)` across all active streams.
pub struct StreamTracker {
    /// Current smoothed inter-token latency in milliseconds.
    smoothed_itl_ms: f64,
    /// Number of intervals observed (excludes first-token event).
    intervals: u64,
    /// Timestamp of the last token arrival.
    last_token_time: Instant,
    /// EWMA smoothing factor.
    alpha: f64,
}

impl StreamTracker {
    /// Create a new tracker with the given EWMA alpha.
    pub fn new(alpha: f64) -> Self {
        Self {
            smoothed_itl_ms: 0.0,
            intervals: 0,
            last_token_time: Instant::now(),
            alpha,
        }
    }

    /// Record the first token arrival. Seeds `last_token_time` without
    /// computing an interval (TTFT is not a decode interval).
    pub fn record_first_token(&mut self, now: Instant) {
        self.last_token_time = now;
    }

    /// Record a subsequent token arrival. Computes the inter-token interval
    /// and updates the EWMA. Returns the new smoothed ITL in milliseconds.
    pub fn record_token(&mut self, now: Instant) -> f64 {
        let elapsed_ms = now.duration_since(self.last_token_time).as_secs_f64() * 1000.0;
        self.last_token_time = now;

        if self.intervals == 0 {
            // First interval: seed the EWMA directly.
            self.smoothed_itl_ms = elapsed_ms;
        } else {
            self.smoothed_itl_ms =
                self.alpha * elapsed_ms + (1.0 - self.alpha) * self.smoothed_itl_ms;
        }
        self.intervals += 1;
        self.smoothed_itl_ms
    }

    /// Override the EWMA with llama.cpp ground-truth timing from the
    /// terminal SSE event.
    pub fn apply_terminal_correction(&mut self, ms: f64) {
        self.smoothed_itl_ms = ms;
    }

    /// Current smoothed inter-token latency in milliseconds.
    pub fn smoothed_itl_ms(&self) -> f64 {
        self.smoothed_itl_ms
    }

    /// Number of inter-token intervals observed (used for observability and tests).
    #[allow(dead_code)]
    pub fn intervals(&self) -> u64 {
        self.intervals
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn first_interval_seeds_ewma() {
        let mut tracker = StreamTracker::new(0.3);
        let t0 = Instant::now();
        tracker.record_first_token(t0);
        let t1 = t0 + Duration::from_millis(50);
        let itl = tracker.record_token(t1);
        assert!((itl - 50.0).abs() < 0.1, "first interval should seed EWMA directly");
        assert_eq!(tracker.intervals(), 1);
    }

    #[test]
    fn ewma_converges_to_steady_state() {
        let mut tracker = StreamTracker::new(0.3);
        let t0 = Instant::now();
        tracker.record_first_token(t0);

        // Feed 20 tokens at steady 100ms intervals.
        let mut t = t0;
        for _ in 0..20 {
            t += Duration::from_millis(100);
            tracker.record_token(t);
        }

        // Should converge close to 100ms.
        assert!(
            (tracker.smoothed_itl_ms() - 100.0).abs() < 1.0,
            "EWMA should converge to 100ms, got {}",
            tracker.smoothed_itl_ms()
        );
    }

    #[test]
    fn spike_recovery() {
        let mut tracker = StreamTracker::new(0.3);
        let t0 = Instant::now();
        tracker.record_first_token(t0);

        // Establish baseline at 50ms.
        let mut t = t0;
        for _ in 0..10 {
            t += Duration::from_millis(50);
            tracker.record_token(t);
        }
        let baseline = tracker.smoothed_itl_ms();
        assert!((baseline - 50.0).abs() < 1.0);

        // Inject a spike (500ms).
        t += Duration::from_millis(500);
        let after_spike = tracker.record_token(t);
        assert!(after_spike > baseline, "spike should raise EWMA");
        assert!(after_spike < 500.0, "EWMA should dampen spike");

        // Recover back toward 50ms.
        for _ in 0..20 {
            t += Duration::from_millis(50);
            tracker.record_token(t);
        }
        assert!(
            (tracker.smoothed_itl_ms() - 50.0).abs() < 5.0,
            "should recover toward baseline, got {}",
            tracker.smoothed_itl_ms()
        );
    }

    #[test]
    fn terminal_correction_overrides_ewma() {
        let mut tracker = StreamTracker::new(0.3);
        let t0 = Instant::now();
        tracker.record_first_token(t0);
        tracker.record_token(t0 + Duration::from_millis(100));

        tracker.apply_terminal_correction(42.0);
        assert!((tracker.smoothed_itl_ms() - 42.0).abs() < f64::EPSILON);
    }
}
