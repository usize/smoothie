use serde::Deserialize;

/// Configuration for the Smoothie latency-floor concurrency controller.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothieConfig {
    /// Per-stream decode-rate floor in tokens per second.
    /// When `None`, the controller uses derivative-based crossover detection
    /// instead of a fixed floor.
    #[serde(default)]
    pub floor_tps: Option<f64>,

    /// Fractional ITL increase that triggers multiplicative decrease in
    /// derivative mode. Must be in the open interval (0, 1).
    #[serde(default = "default_sensitivity")]
    pub sensitivity: f64,

    /// Slack below the floor (in ms) before additive increase.
    #[serde(default = "default_headroom_ms")]
    pub headroom_ms: u64,

    /// Multiplicative decrease factor.
    #[serde(default = "default_beta")]
    pub beta: f64,

    /// EWMA smoothing factor for inter-token latency.
    #[serde(default = "default_ewma_alpha")]
    pub ewma_alpha: f64,

    /// Consecutive over/under-threshold steps before acting.
    #[serde(default = "default_hysteresis_steps")]
    pub hysteresis_steps: u32,

    /// Minimum admission ceiling.
    #[serde(default = "default_ceiling_min")]
    pub ceiling_min: u32,

    /// Maximum admission ceiling.
    #[serde(default = "default_ceiling_max")]
    pub ceiling_max: u32,

    /// Initial ceiling (defaults to ceiling_max if unset).
    pub ceiling_init: Option<u32>,

    /// Consecutive 5xx/timeout failures before circuit opens.
    #[serde(default = "default_cb_consecutive_failures")]
    pub cb_consecutive_failures: u32,

    /// Seconds before half-open probe after circuit opens.
    #[serde(default = "default_cb_recovery_window_secs")]
    pub cb_recovery_window_secs: u64,
}

impl SmoothieConfig {
    /// Derived inter-token latency floor in milliseconds.
    /// Returns `None` when derivative mode is active (no fixed floor).
    pub fn floor_ms(&self) -> Option<f64> {
        self.floor_tps.map(|tps| 1000.0 / tps)
    }

    /// Validate all config invariants. Returns an error message on failure.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(tps) = self.floor_tps {
            if !tps.is_finite() || tps <= 0.0 {
                return Err("floor_tps must be a finite number greater than 0".into());
            }
        }
        if !self.sensitivity.is_finite() || self.sensitivity <= 0.0 || self.sensitivity >= 1.0 {
            return Err("sensitivity must be in the open interval (0, 1)".into());
        }
        if !self.beta.is_finite() || self.beta <= 0.0 || self.beta >= 1.0 {
            return Err("beta must be in the open interval (0, 1)".into());
        }
        if !self.ewma_alpha.is_finite() || self.ewma_alpha <= 0.0 || self.ewma_alpha > 1.0 {
            return Err("ewma_alpha must be in the half-open interval (0, 1]".into());
        }
        if self.ceiling_min < 1 {
            return Err("ceiling_min must be at least 1".into());
        }
        if self.ceiling_max < self.ceiling_min {
            return Err("ceiling_max must be >= ceiling_min".into());
        }
        if let Some(init) = self.ceiling_init
            && (init < self.ceiling_min || init > self.ceiling_max) {
                return Err("ceiling_init must be between ceiling_min and ceiling_max".into());
            }
        if self.cb_consecutive_failures == 0 {
            return Err("cb_consecutive_failures must be at least 1".into());
        }
        if self.cb_recovery_window_secs == 0 {
            return Err("cb_recovery_window_secs must be at least 1".into());
        }
        Ok(())
    }
}

fn default_sensitivity() -> f64 {
    0.10
}
fn default_headroom_ms() -> u64 {
    15
}
fn default_beta() -> f64 {
    0.8
}
fn default_ewma_alpha() -> f64 {
    0.3
}
fn default_hysteresis_steps() -> u32 {
    3
}
fn default_ceiling_min() -> u32 {
    1
}
fn default_ceiling_max() -> u32 {
    64
}
fn default_cb_consecutive_failures() -> u32 {
    5
}
fn default_cb_recovery_window_secs() -> u64 {
    30
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SmoothieConfig {
        serde_yaml::from_str("{}").unwrap()
    }

    #[test]
    fn defaults_are_valid() {
        let cfg = default_config();
        cfg.validate().unwrap();
    }

    #[test]
    fn default_floor_tps_is_none() {
        let cfg = default_config();
        assert!(cfg.floor_tps.is_none());
        assert!(cfg.floor_ms().is_none());
    }

    #[test]
    fn floor_ms_derivation() {
        let yaml = "floor_tps: 10.0\n";
        let cfg: SmoothieConfig = serde_yaml::from_str(yaml).unwrap();
        let expected = 100.0; // 1000 / 10
        assert!((cfg.floor_ms().unwrap() - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_zero_floor_tps() {
        let mut cfg = default_config();
        cfg.floor_tps = Some(0.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_negative_floor_tps() {
        let mut cfg = default_config();
        cfg.floor_tps = Some(-1.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_infinite_floor_tps() {
        let mut cfg = default_config();
        cfg.floor_tps = Some(f64::INFINITY);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_sensitivity_zero() {
        let mut cfg = default_config();
        cfg.sensitivity = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_sensitivity_one() {
        let mut cfg = default_config();
        cfg.sensitivity = 1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_beta_zero() {
        let mut cfg = default_config();
        cfg.beta = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_beta_one() {
        let mut cfg = default_config();
        cfg.beta = 1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_ewma_alpha_zero() {
        let mut cfg = default_config();
        cfg.ewma_alpha = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn accepts_ewma_alpha_one() {
        let mut cfg = default_config();
        cfg.ewma_alpha = 1.0;
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_ceiling_max_below_min() {
        let mut cfg = default_config();
        cfg.ceiling_min = 10;
        cfg.ceiling_max = 5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_ceiling_init_out_of_range() {
        let mut cfg = default_config();
        cfg.ceiling_init = Some(100);
        cfg.ceiling_max = 64;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn accepts_ceiling_init_in_range() {
        let mut cfg = default_config();
        cfg.ceiling_init = Some(32);
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_zero_cb_consecutive_failures() {
        let mut cfg = default_config();
        cfg.cb_consecutive_failures = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_cb_recovery_window() {
        let mut cfg = default_config();
        cfg.cb_recovery_window_secs = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn deserializes_partial_yaml() {
        let yaml = "floor_tps: 20.0\nbeta: 0.5\n";
        let cfg: SmoothieConfig = serde_yaml::from_str(yaml).unwrap();
        assert!((cfg.floor_tps.unwrap() - 20.0).abs() < f64::EPSILON);
        assert!((cfg.beta - 0.5).abs() < f64::EPSILON);
        // Rest should be defaults
        assert_eq!(cfg.headroom_ms, 15);
        assert_eq!(cfg.ceiling_max, 64);
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_unknown_fields() {
        let yaml = "floor_tps: 10.0\nunknown_field: true\n";
        let result: Result<SmoothieConfig, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err());
    }
}
