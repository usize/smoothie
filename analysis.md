# Tokens Are Streaming Media: Landscape Analysis

*An analysis of prior art, adjacent systems, and theoretical foundations for treating LLM token streams as streaming media subject to flow control.*

## 1. The Thesis

Smoothie's core claim: a decoding LLM stream has a consumption rate the same way a video has a bitrate, and there is a floor below which the experience degrades. The right control is admission (how many streams decode concurrently), not throttling (slowing individual streams). The AI gateway is the natural place to apply media-style flow control to LLM traffic.

This document surveys who else has recognized the streaming media analogy, what architectures they've built, and where smoothie's approach — AIMD concurrency control over observed inter-token latency — sits in the landscape.

---

## 2. TokenFlow: The Explicit Analogy

**Citation:** Junyi Chen, Chuheng Du, Renyuan Liu, Shuochao Yao, et al. "TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling." arXiv:2510.02758v1, October 2025. Shanghai Jiao Tong University and George Mason University.

TokenFlow is the most direct prior statement of the thesis. It frames LLM token generation as analogous to video streaming: surplus tokens accumulate in output buffers, creating opportunities for non-disruptive preemption without depleting buffers.

Their QoS metric borrows the three pillars of video quality of experience:

> QoS = (1/T) * sum(w_{i,j}) - lambda * t_i^ttft - mu * Rebuffer_i

— "token usefulness" (video quality), "startup latency" (TTFT, analogous to initial buffering), and "playback continuity" penalizing rebuffering events.

TokenFlow's buffer-aware admission control prevents overcommitting when insufficient buffer reserves exist to survive I/O latency:

> b_i^rem >= mu * r_i * (tau_evict + tau_load + tau_schedule)

This directly parallels how video players maintain a minimum buffer threshold before starting playback.

The scheduling utility function uses an exponential decay penalty for empty buffers — effectively a back-pressure signal:

> U_i = v_i * t' - gamma * phi(b_i^rem)

**Results:** 82.5% higher effective throughput (accounting for actual user consumption, not raw generation), 80.2% reduction in P99 TTFT, 52.6% reduction in mean TTFT.

**Key difference from smoothie:** TokenFlow applies the streaming media analogy to *server-side scheduling* (which requests to serve when), while smoothie applies it to *gateway-side admission control* (how many streams to admit). TokenFlow assumes control of the inference engine; smoothie sits outside it, observing the stream as a black box.

---

## 3. Streaming Content Monitoring (SCM): The Partial-Content Problem

**Citation:** Yang Li, Qiang Sheng, Yehan Yang, Xueyao Zhang, Juan Cao. "From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring." arXiv:2506.09996v1, June 2025. Institute of Computing Technology, Chinese Academy of Sciences.

SCM addresses the fundamental challenge any streaming filter faces: making decisions on incomplete content. Content moderators trained on complete outputs fail when applied to partial outputs during streaming — what SCM calls the training-inference gap.

> "Gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average."

> "Over 80% of harmful responses can be detected within the first 30% of tokens."

SCM solves this with dual supervision: token-level and response-level loss functions connected by a logic constraint:

> "If response is harmful, at least one token should be predicted harmful."

Their Delay-k mechanism — requiring k harmful tokens before termination — trades off between false alarm rate and detection latency:

> "The larger the k, the lower the FAR [false alarm rate] and the higher the MAR [missing alarm rate]," allowing providers to adjust "strictness of harmfulness monitoring flexibly according to specific scenarios."

**Relevance to smoothie:** SCM validates that streaming content classification is a distinct problem from batch classification. The Delay-k threshold is analogous to smoothie's `hysteresis_steps` — both require sustained signal before acting, to avoid reacting to transient spikes. SCM's finding that harmful intent is front-loaded (95%+ accuracy at 18% of tokens) has implications for buffer sizing in any streaming content filter.

**Dataset:** FineHarm — 29K prompt-response pairs with token-level annotations using POS-based filtering. Harmful intents correlate more strongly with notional words (nouns, verbs, adjectives) than function words.

---

## 4. NeMo Guardrails: The RollingBuffer Architecture

**Citation:** NVIDIA. "Stream Smarter and Safer: Learn How NVIDIA NeMo Guardrails Enhance LLM Output Streaming." NVIDIA Developer Blog, 2024. GitHub: https://github.com/NVIDIA-NeMo/Guardrails

NeMo Guardrails is the most mature production system for streaming LLM output filtering. Its `RollingBuffer` implements the buffer-between-production-and-consumption pattern:

> "When output rails are active, the system cannot stream tokens immediately because they must be validated. To balance latency and safety, NeMo Guardrails uses Chunk-based Streaming."

The RollingBuffer yields `ChunkBatch` objects containing a `processing_context` (full text including overlap from the previous window) and `user_output_chunks` (only new tokens released if the check passes).

**Configuration parameters:**

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `chunk_size` | New tokens before triggering a rail check | 50–256 tokens |
| `context_size` | Overlap from previous chunk for coherence | 20–50 tokens |
| `stream_first` | Send first chunk without validation | true/false |
| `parallel` | Run multiple output rails concurrently | true |

The three-layer streaming approach:
1. Basic LLM streaming (token-by-token from provider)
2. Output rails streaming (chunked safety checks via RollingBuffer)
3. Parallel execution (concurrent rails on same chunks)

> "If true, the very first chunk is sent to the user immediately without waiting for the rail check (use with caution)."

**Relevance to smoothie:** NeMo's RollingBuffer is the closest production architecture to smoothie's buffer model. Both use sliding windows with overlap. The key difference: NeMo's `stream_first` is a static boolean — send the first chunk unsafely or wait. Smoothie's AIMD controller finds this operating point dynamically.

---

## 5. Qwen3Guard Stream: Per-Token Classification

**Citation:** Haiquan Zhao, Chenhan Yuan, Fei Huang, et al. (Qwen Team). "Qwen3Guard Technical Report." arXiv:2510.14276, October 2025.

Qwen3Guard's Stream variant represents the per-token extreme of the design space. Unlike chunk-based approaches, it evaluates each token incrementally using dual classification heads attached to the transformer's final layer:

> "Real-time, per-token moderation" with "two parallel and independent pathways: one dedicated to analyzing the model's generated response and the other for the user's query."

> "The loss for the response stream is computed at every generated token."

**Performance:** Nearly 86.0% exact hit rate for detecting unsafe content within the sentence where violations first occur. Detects unsafe content within the first 128 tokens in approximately 66.8% of cases.

**Efficiency:** Processing time scales linearly with response length, while generative approaches incur substantially higher overhead as responses grow.

**Model sizes:** 0.6B, 4B, 8B parameters. 119 languages. Three-tiered classification (Safe / Controversial / Unsafe) across nine safety categories.

**Relevance to smoothie:** Qwen3Guard Stream validates the design space. Per-token approaches give lowest latency but highest overhead; chunk/buffer approaches trade latency for efficiency. These are complementary to smoothie — Qwen3Guard answers "is this token safe?" while smoothie answers "can we afford to admit another stream?"

---

## 6. AI Gateways: The Convergent Architecture

Multiple AI gateways have converged on a streaming filter pattern where content inspection happens at the proxy layer as tokens flow through.

### Tyk AI Gateway

**Citation:** Tyk Technologies. "AI Studio Filters." Tyk Documentation, 2025.

Tyk evaluates filter scripts on each incoming chunk. Scripts receive both the current chunk and an accumulated buffer:

> The `current_buffer` field contains "accumulated response text so far," enabling pattern detection across message boundaries.

Scripts themselves control evaluation timing via buffer length conditions — a static version of adaptive evaluation:

```
if !input.is_chunk || len(response_text) >= 100 {
    // Run expensive pattern matching
}
```

> "This deferred evaluation pattern prevents false positives from incomplete content while maintaining performance."

Response filters operate in block-only mode — they cannot modify LLM output, only prevent problematic responses from reaching users.

### Other Gateways

- **Portkey** (acquired by Palo Alto Networks, 2025) — <1ms AI gateway, 60+ guardrails, streaming support, bring-your-own-guardrails via webhooks.
- **Cloudflare AI Gateway** — Llama Guard on Workers AI at the edge. Cross-provider consistency.
- **Kong AI Gateway** — SSE capture and normalization across providers.
- **LiteLLM** — OpenAI Moderation guardrail with full streaming support.

### Enterprise LLM Firewalls

Significant acquisition activity signals market validation:
- **CalypsoAI** acquired by F5 Networks for $180M — AI security at the inference layer, positioned into application delivery infrastructure.
- **Lakera** acquired by Check Point Software (November 2025) — sub-50ms prompt injection detection.
- **Aporia** acquired by Coralogix — first guardrails for audio, vision, and text AI.
- **Arthur AI Shield** — open-source guardrails engine.
- **Pangea** (CrowdStrike) — configurable detection policies with block/report/redact/encrypt actions.

### Cloud Provider Built-in

- **AWS Bedrock Guardrails** — six safeguard policies, works with any foundation model via ApplyGuardrail API.
- **Azure AI Content Safety** — segment-by-segment classification during streaming. Prompt Shields for adversarial injection detection.

---

## 7. The Broadcast Delay: Historical Predecessor

**Citations:**
- "Broadcast delay." Wikipedia.
- Eventide, Inc. "BD500/BD600 Broadcast Profanity Delay" product documentation.

The broadcast profanity delay, invented by C. Frank Cordaro at WKAP (Allentown, PA) in the 1950s, is the historical ancestor of all streaming content filtering. The architecture is a circular buffer between live production and audience consumption:

> "A delay unit holds the outgoing program in a circular buffer (typical delays 5–10 seconds). The operator listens to the program in real time while the audience hears the program delayed by that buffer. That time window is the opportunity to act."

**Technical details:**
- Buffer duration: 5–10 seconds (7 seconds standard)
- Dump amount: configurable, defaults to 3.00 seconds
- Rebuild time: ~120 seconds to rebuild full buffer after dump
- Rebuild mechanism: audio stretching via autocorrelation to find perceptually transparent stretch points
- Hardware evolution: tape loops (1950s) → digital RAM (1977, Eventide) → modern DSP/software

The dump operation is the key intervention:

> "DUMP causes the unit to delete several seconds (configurable) from the delay line including the audio information which has most recently been input to the delay line."

After a dump, the buffer must be rebuilt. Modern systems do this by stretching audio imperceptibly:

> "They send real-time audio, stretched inaudibly, building up the buffer over the course of a couple minutes. Utterly seamless."

| Broadcast Delay | LLM Token Filtering |
|----------------|---------------------|
| Circular audio buffer (5–7 sec) | Token buffer / sliding window |
| Operator monitors real-time feed | Safety classifier evaluates tokens |
| Dump button removes bad content | Filter blocks harmful tokens |
| Buffer rebuild via stretching | AIMD additive increase rebuilds admission headroom |
| Bleep replaces specific words | Token replacement / redaction |
| Dead air after dump | Latency spike after safety intervention |

The broadcast industry settled on 5–7 seconds empirically. Smoothie's AIMD controller finds this operating point dynamically.

---

## 8. Adaptive Bitrate Streaming: The Control Theory Peer

**Citation:** "Adaptive bitrate streaming." Wikipedia. Dacast, "Adaptive Bitrate Streaming: What it Is and How the ABR Algorithm Works," 2026.

ABR encodes video at multiple bitrates, segments it into small chunks (2–10 seconds), and uses a client-side algorithm to switch quality levels based on observed conditions:

> "An adaptive bitrate (ABR) algorithm in the client performs the key function of deciding which bit rate segments to download, based on the current state of the network."

Three algorithm families:

> "Throughput-based algorithms use the throughput achieved in recent prior downloads for decision-making; buffer-based algorithms use only the client's current buffer level; and hybrid algorithms combine both types of information."

Congestion response is immediate and graceful:

> "If the throughput drops sharply or the buffer is in danger of emptying, the player immediately switches to a lower-bitrate version."

| ABR Streaming | Smoothie |
|---------------|----------|
| Adapts bitrate based on network conditions | Adapts admission ceiling based on decode contention |
| Buffer-based algorithms use buffer occupancy | AIMD uses observed inter-token latency |
| Segment-by-segment decisions | Per-stream, continuous decisions |
| Avoids rebuffering (buffer starvation) | Avoids floor violation (latency starvation) |
| Protocols: HLS, DASH, CMAF | Protocol: SSE over HTTP |

The critical difference: ABR adapts *content quality* (bitrate) in response to *network quality* (throughput/buffer). Smoothie adapts *admission quantity* (concurrent streams) in response to *decode quality* (inter-token latency). Both are feedback control loops operating on streaming media, controlling different knobs in response to different signals.

---

## 9. TCP Congestion Control: The Formal Foundation

**Citations:**
- Van Jacobson, Michael J. Karels. "Congestion Avoidance and Control." ACM SIGCOMM '88, Stanford, CA, August 1988, pp. 314–329.
- Dah-Ming Chiu, Raj Jain. "Analysis of the Increase and Decrease Algorithms for Congestion Avoidance in Computer Networks." Computer Networks and ISDN Systems, 17(1):1–14, June 1989.
- Larry Peterson, Bruce Davie. "Computer Networks: A Systems Approach," 6th Edition, Section 6.3.

The AIMD algorithm provides the mathematical foundation for smoothie's admission controller. Jacobson's 1988 paper opens with the motivating catastrophe:

> "In October of '86, the Internet had the first of what became a series of 'congestion collapses.' During this period, the data throughput from LBL to UC Berkeley (sites separated by 400 yards and three IMP hops) dropped from 32 Kbps to 40 bps."

His solution rests on a conservation principle:

> "A new packet isn't put into the network until an old packet leaves."

The AIMD control law — additive increase, multiplicative decrease — was proven optimal by Chiu and Jain (1989):

> "A simple additive increase and multiplicative decrease algorithm satisfies the sufficient conditions for convergence to an efficient and fair state regardless of the starting state of the network."

The asymmetry is deliberate. Peterson and Davie explain why multiplicative decrease is essential:

> "The consequences of having too large a window are compounding. This is because when the window is too large, packets that are dropped will be retransmitted, making congestion even worse. It is important to get out of this state quickly."

And why additive increase probes gently:

> "TCP gently increases the data in flight to probe for the level at which congestion begins, then aggressively stepping back from the brink of congestion collapse when that level is detected."

The result is a saw-tooth pattern where the source:

> "is willing to reduce its congestion window at a much faster rate than it is willing to increase its congestion window."

Only AIMD converges to both efficiency and fairness. Other combinations fail:

> "AIAD and MIMD retain unfairness, and make no improvements toward fairness. MIAD increases unfairness, and AIMD converges toward fairness."

### Mapping to Smoothie

| TCP AIMD | Smoothie AIMD |
|----------|---------------|
| Congestion window (cwnd) | Admission ceiling (max concurrent streams) |
| Packet loss signal | Slowest stream crossing the latency floor |
| Additive increase (+1 MSS/RTT) | Ceiling += 1 when headroom exists |
| Multiplicative decrease (cwnd /= 2) | Ceiling *= beta (beta ~0.8) |
| RTT feedback loop | Inter-token latency feedback loop |
| Conservation of packets | Conservation of decode bandwidth |
| Congestion collapse (32 Kbps → 40 bps) | Floor collapse (all streams stall) |

The formal justification carries over: the consequences of admitting too many streams are compounding (every additional stream slows all streams nonlinearly near KV/bandwidth saturation), so multiplicative decrease is essential. The cost of being too conservative (underutilized GPU) is merely linear, so additive increase is appropriate.

---

## 10. The Gap

Most of the landscape is doing **content safety filtering** — inspecting what the tokens say. Smoothie is doing something different: **flow control** — controlling how many streams decode simultaneously to maintain a QoS floor.

Nobody in the landscape is doing AIMD-style adaptive concurrency control over token streams at the gateway layer. The AI gateways do rate limiting (request count, token budget) but not adaptive concurrency limiting based on observed inter-token latency. The inference servers do batch scheduling (vLLM, TGI) but not gateway-level admission control that treats the token stream as opaque streaming media with a bitrate floor.

Smoothie occupies the intersection: the control theory of TCP congestion avoidance, applied to a streaming medium (LLM tokens), at the architectural position of a CDN edge node (the AI gateway).
