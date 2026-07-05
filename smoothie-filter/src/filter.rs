use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::DashMap;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
};

use crate::aimd::{AimdConfig, AimdController};
use crate::circuit_breaker::{BreakerState, FaultCircuitBreaker};
use crate::config::SmoothieConfig;
use crate::semaphore::AdaptiveSemaphore;
use crate::sse::SseParserState;
use crate::stream_tracker::StreamTracker;

/// Per-request state stored in the filter context via `insert_filter_state`.
struct SmoothieRequestState {
    stream_id: u64,
    admitted: bool,
    sse: SseParserState,
}

/// Smoothie: a latency-floor concurrency controller for LLM token streams.
///
/// Caps concurrent streams via an AIMD controller and sheds overflow as fast 429s.
/// Treats LLM token streams as streaming media with a per-stream bitrate floor.
pub struct SmoothieFilter {
    aimd: AimdController,
    semaphore: AdaptiveSemaphore,
    streams: DashMap<u64, StreamTracker>,
    next_stream_id: AtomicU64,
    circuit_breaker: FaultCircuitBreaker,
    ewma_alpha: f64,
}

impl SmoothieFilter {
    /// Factory function for constructing from YAML config.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: SmoothieConfig = praxis_filter::parse_filter_config("smoothie", config)?;
        cfg.validate().map_err(|e| -> FilterError {
            format!("smoothie: {e}").into()
        })?;

        let floor_ms = cfg.floor_ms();
        // Derivative mode starts at ceiling_min (probes upward).
        // Fixed-floor mode starts at ceiling_max (decreases toward equilibrium).
        let initial_ceiling = cfg
            .ceiling_init
            .unwrap_or(if floor_ms.is_none() { cfg.ceiling_min } else { cfg.ceiling_max });

        let aimd_config = AimdConfig {
            floor_ms,
            headroom_ms: cfg.headroom_ms as f64,
            beta: cfg.beta,
            hysteresis_steps: cfg.hysteresis_steps,
            ceiling_min: cfg.ceiling_min,
            ceiling_max: cfg.ceiling_max,
            sensitivity: cfg.sensitivity,
        };

        Ok(Box::new(Self {
            aimd: AimdController::new(aimd_config, initial_ceiling),
            semaphore: AdaptiveSemaphore::new(),
            streams: DashMap::new(),
            next_stream_id: AtomicU64::new(0),
            circuit_breaker: FaultCircuitBreaker::new(
                cfg.cb_consecutive_failures,
                cfg.cb_recovery_window_secs,
            ),
            ewma_alpha: cfg.ewma_alpha,
        }))
    }

    /// Compute the median smoothed ITL across all active streams.
    ///
    /// Using the median (p50) instead of the max makes the control signal
    /// robust to single-stream outliers (prefill spikes, scheduling jitter)
    /// while remaining sensitive to batch-wide ITL shifts at the capacity
    /// cliff.
    fn median_smoothed_itl(&self) -> f64 {
        let mut itls: Vec<f64> = self
            .streams
            .iter()
            .map(|entry| entry.value().smoothed_itl_ms())
            .filter(|&itl| itl > 0.0)
            .collect();

        if itls.is_empty() {
            return 0.0;
        }

        itls.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = itls.len() / 2;
        if itls.len() % 2 == 0 {
            (itls[mid - 1] + itls[mid]) / 2.0
        } else {
            itls[mid]
        }
    }
}

#[async_trait]
impl HttpFilter for SmoothieFilter {
    fn name(&self) -> &'static str {
        "smoothie"
    }

    async fn on_request(
        &self,
        ctx: &mut HttpFilterContext<'_>,
    ) -> Result<FilterAction, FilterError> {
        // Check circuit breaker first.
        match self.circuit_breaker.check() {
            BreakerState::Open => {
                return Ok(FilterAction::Reject(
                    Rejection::status(503)
                        .with_header("X-Smoothie-State", "circuit-open")
                        .with_body("service unavailable: circuit breaker open"),
                ));
            }
            BreakerState::Closed | BreakerState::HalfOpen => {}
        }

        // Try to acquire a semaphore slot.
        let ceiling = self.aimd.ceiling();
        if !self.semaphore.try_acquire(ceiling) {
            return Ok(FilterAction::Reject(
                Rejection::status(429)
                    .with_header("Retry-After", "1")
                    .with_header("X-Smoothie-Ceiling", ceiling.to_string()),
            ));
        }

        // Admitted — set up per-request state.
        let stream_id = self.next_stream_id.fetch_add(1, Ordering::Relaxed);
        self.streams
            .insert(stream_id, StreamTracker::new(self.ewma_alpha));

        ctx.insert_filter_state(SmoothieRequestState {
            stream_id,
            admitted: true,
            sse: SseParserState::new(),
        });

        Ok(FilterAction::Continue)
    }

    async fn on_response(
        &self,
        ctx: &mut HttpFilterContext<'_>,
    ) -> Result<FilterAction, FilterError> {
        // Feed circuit breaker based on response status.
        let is_server_error = ctx
            .response_header
            .as_ref()
            .is_some_and(|r| r.status.is_server_error());

        if is_server_error {
            self.circuit_breaker.record_failure();
        } else {
            self.circuit_breaker.record_success();
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        let Some(state) = ctx.get_filter_state_mut::<SmoothieRequestState>() else {
            // Not admitted (rejected at request phase) — nothing to do.
            return Ok(FilterAction::Continue);
        };

        if !state.admitted {
            return Ok(FilterAction::Continue);
        }

        let stream_id = state.stream_id;
        let now = Instant::now();

        // Feed any body bytes to the SSE parser.
        if let Some(chunk) = body.as_ref() {
            let result = state.sse.feed(chunk);

            // Process token observations.
            // NOTE: We must drop the DashMap guard before calling
            // median_smoothed_itl(), which iterates the map. Holding a
            // get_mut write-lock while iterating would deadlock on
            // the same shard.
            let mut should_observe_aimd = false;
            if result.new_tokens > 0 {
                if let Some(mut tracker) = self.streams.get_mut(&stream_id) {
                    if !state.sse.first_token_seen() || state.sse.token_count() <= 1 {
                        tracker.record_first_token(now);
                    } else {
                        for _ in 0..result.new_tokens {
                            tracker.record_token(now);
                        }
                        should_observe_aimd = true;
                    }
                }
                // Guard dropped here.
                if should_observe_aimd {
                    let max_itl = self.median_smoothed_itl();
                    if max_itl > 0.0 {
                        self.aimd.observe(max_itl, self.semaphore.active());
                    }
                }
            }

            // Apply terminal timing correction if available.
            if let Some(timing_ms) = result.terminal_timing_ms {
                if let Some(mut tracker) = self.streams.get_mut(&stream_id) {
                    tracker.apply_terminal_correction(timing_ms);
                }
                // Guard dropped here — safe to iterate the map.
                let max_itl = self.median_smoothed_itl();
                if max_itl > 0.0 {
                    self.aimd.observe(max_itl, self.semaphore.active());
                }
            }

            // Stream is done — clean up.
            if result.stream_done {
                if self.streams.remove(&stream_id).is_some() {
                    self.semaphore.release();
                }
                return Ok(FilterAction::BodyDone);
            }
        }

        // If end_of_stream without SSE done marker, clean up anyway.
        // Guard on remove() to prevent double-release (BodyDone is
        // not persisted across Pingora body filter invocations).
        if end_of_stream {
            if self.streams.remove(&stream_id).is_some() {
                self.semaphore.release();
            }
            return Ok(FilterAction::BodyDone);
        }

        Ok(FilterAction::Continue)
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn needs_request_context(&self) -> bool {
        false
    }
}
