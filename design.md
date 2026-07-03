# smoothie

*A latency-floor concurrency controller for LLM traffic, built as a Praxis response-phase plugin.*

## The idea

Tokens are streaming media. A decoding LLM stream has a consumption rate the same way a video has a bitrate, and — like video — there is a floor below which the experience degrades. For a chat or agent stream, that floor is roughly human-or-agent-usable throughput: below ~10 tok/s the stream feels stalled.

On a single shared local model, per-stream decode rate is a function of concurrency: every additional actively-decoding sequence shares the same forward pass, so each stream slows as the batch grows. Aggregate throughput rises; individual rate falls. That means the way to *guarantee a floor* is not to throttle anyone — it's to cap how many streams decode at once, and shed the overflow.

**smoothie** treats the excess decode rate above the floor as a reclaimable scheduling resource. It holds the slowest in-flight stream above a target rate by adapting an admission ceiling in real time, and returns a fast `429` when admitting one more request would drag the batch below the floor. It is an AI-gateway demonstration that thinking about tokens as streaming media — with a bitrate floor, congestion, and back-pressure — yields a cleaner control model than request-count rate limiting.

## Where it runs

smoothie runs in the **response body phase** of Praxis. It sits in the streaming path and observes token emission directly:

- **Rate signal comes from the body, not headers.** By parsing SSE chunks (or the terminal buffered body) in the response phase, smoothie measures *actual* inter-token latency — the real thing being controlled — rather than a request count that only proxies for it. Token count and timing are read from the body as chunks arrive.
- **Admission happens at request entry; measurement happens at response.** The two phases cooperate: the request phase acquires a slot from the semaphore (or rejects), the response phase measures the resulting per-token latency and feeds the controller that sizes the semaphore.

## The control loop

The core is a concurrency limiter with **adaptive capacity** (AIMD, in the lineage of TCP congestion control), *not* a binary circuit breaker. Open/closed can express "admit all" or "admit none"; smoothie needs "admit N, reject the N+1th," where N tracks a moving curve.

**Signal.** The live control signal is the *slowest currently-decoding stream* — the `max` smoothed inter-token latency across all active responses. That is the stream about to cross the floor. Each stream's inter-token latency is derived from SSE chunk arrival times in the response phase; the terminal `timings.predicted_per_token_ms` (llama.cpp) is used to correct the running estimate when a stream completes.

**Controller (AIMD over the latency signal).**
- **Additive increase:** when the slowest stream sits comfortably under the floor (with headroom), raise the ceiling by 1 — probe for free capacity.
- **Multiplicative decrease:** when it crosses the threshold, cut the ceiling hard (`ceiling *= β`, β ≈ 0.8). Decode latency degrades nonlinearly near KV/bandwidth saturation, so back off the cliff fast.

This is why an adaptive ceiling beats a static number: long-context batches settle at a lower ceiling and short-context batches drift higher, automatically, without re-benchmarking.

**Gate.** A semaphore whose size *is* the current ceiling sits in front of the model. Acquire on admit, release on stream completion. Overflow gets an immediate `429` (fast shed, not a hidden queue — no reintroduced latency).

**Prefill hysteresis.** A prompt-processing pass transiently spikes every stream's inter-token latency, even chunked. The controller smooths the signal (EWMA over a short window, or require the threshold to hold across 2–3 consecutive decode steps) before reacting, so it does not panic-cut on a spike that clears in a couple hundred milliseconds. Prefill/decode coupling is handled as controller hysteresis, not by pretending the phases are separable.

**Fault path (the one place the breaker idea survives).** A separate, true circuit-breaker state handles genuine faults — backend unreachable, 5xx, timeouts. That trips hard and probes on recovery. Two controllers, different signals, one semaphore: AIMD for the latency floor (common case), breaker for failure (rare case).

## Parameters

| Parameter | Meaning | Starting point |
|---|---|---|
| `floor_tps` | Per-stream decode-rate floor | 10 tok/s (100 ms/token) |
| `headroom_ms` | Slack below the floor before additive increase | 10–20 ms |
| `beta` | Multiplicative-decrease factor | 0.8 |
| `ewma_alpha` | Smoothing on the latency signal | tuned so prefill spikes don't trip β |
| `hysteresis_steps` | Consecutive over-threshold steps before a cut | 2–3 |
| `ceiling_min` / `ceiling_max` | Bounds on admission ceiling | `1` / `-np` of the backend |

Scope of the ceiling is **global** across all agents in v1 (one shared pool, one semaphore — fairest use of the box). Per-agent floors (weighted admission) are a later extension.

## Response contract

- **Admitted:** request proceeds; response phase streams normally while measuring.
- **Shed:** `429 Too Many Requests` with `Retry-After`, returned before the request reaches the model. The agent backs off and retries — it does not sit in a queue accruing latency.
- **Fault open:** `503` while the breaker is open; probes on the configured interval.

## Packaging

smoothie lives in its own repo, decoupled from Praxis. The repo pulls in Praxis as a dependency, builds it, and installs smoothie as a response-phase plugin — so the plugin ships and versions independently while riding the gateway's streaming machinery. The point of the demo: this control model is only clean *because* it lives in an AI gateway that can see the token stream as a first-class medium.

## What it demonstrates

- A per-stream rate **floor** is an admission/concurrency problem, not a throttling one.
- The right signal is **observed inter-token latency**, available only by reading the response body — which is exactly what a gateway response-phase plugin is positioned to do.
- **Tokens are streaming media**: bitrate floor, congestion, back-pressure, graceful shedding. The AI gateway is the natural place to apply media-style flow control to LLM traffic.
