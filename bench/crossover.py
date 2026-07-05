#!/usr/bin/env python3
"""
Crossover detection benchmark: verify that Smoothie's derivative-based
controller discovers the roofline-optimal batch size as sequence length
grows.

For each target context length, saturates the proxy with concurrent
streaming requests and observes where the AIMD ceiling stabilizes.
Compares the observed ceiling to the theoretical arithmetic-intensity
crossover point derived from the model architecture and hardware specs.

Requirements:
    pip install aiohttp

Setup:
    # 1. Start the inference backend with high parallelism:
    llama-server -m <model.gguf> --parallel 100 --port 8999

    # 2. Start smoothie in derivative mode (no floor_tps):
    cargo run --release -p smoothie-server -- \
        -c examples/configs/smoothie-derivative.yaml

    # 3. Run the benchmark:
    python bench/crossover.py --model qwen2.5-3b

    # Or test multiple models (restart llama-server between each):
    python bench/crossover.py --model qwen2.5-0.5b
    python bench/crossover.py --model qwen2.5-1.5b
    python bench/crossover.py --model qwen2.5-3b
"""

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Model architecture presets (Q4_K_M quantization assumed)
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "qwen2.5-0.5b": {
        "params": 494_000_000,
        "n_layers": 24,
        "n_kv_heads": 2,
        "d_head": 64,
        "weight_bytes": 400_000_000,
        "default_api_name": "qwen2.5:0.5b",
    },
    "qwen2.5-1.5b": {
        "params": 1_540_000_000,
        "n_layers": 28,
        "n_kv_heads": 2,
        "d_head": 128,
        "weight_bytes": 1_000_000_000,
        "default_api_name": "qwen2.5:1.5b",
    },
    "qwen2.5-3b": {
        "params": 3_090_000_000,
        "n_layers": 36,
        "n_kv_heads": 2,
        "d_head": 128,
        "weight_bytes": 2_000_000_000,
        "default_api_name": "qwen2.5:3b",
    },
    "llama-3.2-1b": {
        "params": 1_240_000_000,
        "n_layers": 16,
        "n_kv_heads": 8,
        "d_head": 64,
        "weight_bytes": 750_000_000,
        "default_api_name": "llama3.2:1b",
    },
    "llama-3.2-3b": {
        "params": 3_210_000_000,
        "n_layers": 28,
        "n_kv_heads": 8,
        "d_head": 128,
        "weight_bytes": 2_020_000_000,
        "default_api_name": "llama3.2:3b",
    },
}

# ---------------------------------------------------------------------------
# Hardware presets — FP16 peak FLOPS and memory bandwidth
# ---------------------------------------------------------------------------

HARDWARE: dict[str, dict] = {
    "M1":       {"mem_bw": 68.25e9,  "peak_flops": 2.6e12},
    "M1 Pro":   {"mem_bw": 200e9,  "peak_flops": 5.2e12},
    "M1 Max":   {"mem_bw": 400e9,  "peak_flops": 10.4e12},
    "M1 Ultra": {"mem_bw": 800e9,  "peak_flops": 20.8e12},
    "M2":       {"mem_bw": 100e9,  "peak_flops": 3.6e12},
    "M2 Pro":   {"mem_bw": 200e9,  "peak_flops": 7.1e12},
    "M2 Max":   {"mem_bw": 400e9,  "peak_flops": 14.2e12},
    "M2 Ultra": {"mem_bw": 800e9,  "peak_flops": 28.4e12},
    "M3":       {"mem_bw": 100e9,  "peak_flops": 4.1e12},
    "M3 Pro":   {"mem_bw": 150e9,  "peak_flops": 6.8e12},
    "M3 Max":   {"mem_bw": 400e9,  "peak_flops": 14.2e12},
    "M3 Ultra": {"mem_bw": 800e9,  "peak_flops": 28.4e12},
    "M4":       {"mem_bw": 120e9,  "peak_flops": 4.4e12},
    "M4 Pro":   {"mem_bw": 273e9,  "peak_flops": 8.7e12},
    "M4 Max":   {"mem_bw": 546e9,  "peak_flops": 17.4e12},
}

# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------

SMOOTHIE_URL = "http://127.0.0.1:8081"
CONCURRENCY = 200
MAX_TOKENS = 50
STABILIZE_SECS = 30
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

TIMEOUT = aiohttp.ClientTimeout(total=300, sock_read=120)
HEADERS = {"Connection": "close"}

# ---------------------------------------------------------------------------
# Theory: roofline crossover
# ---------------------------------------------------------------------------


def kv_bytes_per_token(model: dict) -> int:
    """FP16 KV cache bytes per token per sequence (K + V across all layers)."""
    return model["n_layers"] * model["n_kv_heads"] * model["d_head"] * 4


def theoretical_crossover(model: dict, hw: dict, seq_len: int) -> float:
    """Batch size at the roofline ridge point for a given sequence length.

    At the crossover, memory-load time equals compute time:

        (W + B * S * K) / mem_bw = (2P * B) / peak_flops

    Solving for B:

        B = (W * peak_flops) / (2P * mem_bw - S * K * peak_flops)

    Returns float (may be fractional). Returns inf when the system is
    compute-bound even at B=1 (denominator <= 0).
    """
    P = model["params"]
    W = model["weight_bytes"]
    K = kv_bytes_per_token(model)
    peak = hw["peak_flops"]
    bw = hw["mem_bw"]

    denom = 2 * P * bw - seq_len * K * peak
    if denom <= 0:
        return float("inf")
    return (W * peak) / denom


# ---------------------------------------------------------------------------
# Hardware detection (macOS / Apple Silicon)
# ---------------------------------------------------------------------------


def detect_chip() -> str | None:
    """Try to identify the Apple Silicon chip model."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        brand = result.stdout.strip()
    except Exception:
        return None

    # Match longest key first: "M4 Max" before "M4".
    for key in sorted(HARDWARE, key=len, reverse=True):
        if key in brand:
            return key
    return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def load_fill_text() -> str:
    """Load book.txt for use as context padding."""
    path = Path(__file__).parent / "book.txt"
    if path.exists():
        return path.read_text()
    # Fallback: generate repetitive text.
    para = (
        "The study of computational complexity reveals fundamental limits "
        "on what can be efficiently computed. Problems are classified by "
        "the resources required to solve them, including time and space. "
        "Understanding these boundaries guides algorithm design and helps "
        "identify when approximate solutions are necessary. "
    )
    return (para * 500)


def build_prompt(fill_text: str, target_tokens: int) -> str:
    """Build a prompt of approximately target_tokens length.

    Uses a rough 4-characters-per-token heuristic to size the excerpt.
    Wraps it in an instruction so the model generates decode tokens.
    """
    chars_needed = max(0, target_tokens * 4 - 100)
    excerpt = fill_text[:chars_needed]
    return (
        f"Summarize the following passage in exactly 3 bullet points."
        f"\n\n{excerpt}\n\nBullet points:"
    )


# ---------------------------------------------------------------------------
# Streaming client with ceiling observation
# ---------------------------------------------------------------------------


@dataclass
class CeilingLog:
    """Thread-safe-ish log of ceiling observations from 429 headers."""

    entries: list[tuple[float, int]] = field(default_factory=list)

    def record(self, ceiling: int) -> None:
        self.entries.append((time.monotonic(), ceiling))

    def last_n_secs(self, secs: float) -> list[int]:
        cutoff = time.monotonic() - secs
        return [c for t, c in self.entries if t >= cutoff]

    def stable_ceiling(self, window: float = 5.0) -> int | None:
        recent = self.last_n_secs(window)
        if not recent:
            return None
        # Return the mode (most frequent value).
        from collections import Counter

        counts = Counter(recent)
        return counts.most_common(1)[0][0]


class Rejected(Exception):
    def __init__(self, retry_after: float, ceiling: int):
        self.retry_after = retry_after
        self.ceiling = ceiling


async def consume_stream(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
) -> None:
    """Send a streaming completion and consume all tokens."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    async with session.post(
        f"{url}/v1/chat/completions",
        json=payload,
        headers=HEADERS,
        timeout=TIMEOUT,
    ) as resp:
        if resp.status == 429:
            retry_after = float(resp.headers.get("Retry-After", "1"))
            ceiling = int(resp.headers.get("X-Smoothie-Ceiling", "0"))
            raise Rejected(retry_after, ceiling)
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
        # Drain the body so the stream completes and releases the slot.
        async for _ in resp.content.iter_any():
            pass


async def worker(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    stop: asyncio.Event,
    log: CeilingLog,
) -> None:
    """Continuously send requests until stopped, recording ceiling from 429s."""
    while not stop.is_set():
        try:
            await consume_stream(session, url, model_name, prompt, max_tokens)
        except Rejected as e:
            log.record(e.ceiling)
            try:
                await asyncio.wait_for(stop.wait(), timeout=e.retry_after)
                return
            except asyncio.TimeoutError:
                pass
        except (aiohttp.ClientError, RuntimeError, asyncio.TimeoutError):
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
                return
            except asyncio.TimeoutError:
                pass


# ---------------------------------------------------------------------------
# Round runner
# ---------------------------------------------------------------------------


async def run_round(
    url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    duration: float,
) -> CeilingLog:
    """Run concurrent workers for `duration` seconds, return ceiling log."""
    log = CeilingLog()
    stop = asyncio.Event()
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                worker(session, url, model_name, prompt, max_tokens, stop, log)
            )
            for _ in range(concurrency)
        ]
        await asyncio.sleep(duration)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    return log


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results(
    model_name: str,
    hw_name: str,
    results: list[dict],
) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Crossover Benchmark Results")
    print(f"  Model: {model_name}    Hardware: {hw_name}")
    print(f"{'=' * 72}")

    header = f"  {'seq_len':>8}  {'observed':>10}  {'theory':>10}  {'ratio':>8}  {'429s':>6}"
    print(header)
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 6}")

    for r in results:
        obs = r["observed"]
        theory = r["theory"]
        obs_str = str(obs) if obs is not None else "n/a"
        theory_str = f"{theory:.1f}" if theory != float("inf") else "inf"
        if obs is not None and theory != float("inf") and theory > 0:
            ratio = obs / theory
            ratio_str = f"{ratio:.2f}"
        else:
            ratio_str = "n/a"
        total_429s = r["total_429s"]
        print(
            f"  {r['seq_len']:>8}  {obs_str:>10}  {theory_str:>10}"
            f"  {ratio_str:>8}  {total_429s:>6}"
        )

    print()


def print_theory_table(model_name: str, model: dict, hw: dict) -> None:
    """Print the theoretical crossover at each sequence length."""
    K = kv_bytes_per_token(model)
    R = hw["peak_flops"] / hw["mem_bw"]

    print(f"\n  Roofline parameters:")
    print(f"    Parameters:       {model['params'] / 1e9:.2f}B")
    print(f"    Weight bytes:     {model['weight_bytes'] / 1e9:.2f} GB")
    print(f"    KV bytes/token:   {K:,} B  ({K / 1024:.1f} KB)")
    print(f"    Peak FLOPS:       {hw['peak_flops'] / 1e12:.1f} TFLOPS")
    print(f"    Memory BW:        {hw['mem_bw'] / 1e9:.0f} GB/s")
    print(f"    Ridge (flops/B):  {R:.1f}")

    # Critical sequence length where B=1 is already compute-bound.
    s_crit = 2 * model["params"] * hw["mem_bw"] / (K * hw["peak_flops"])
    print(f"    S_critical:       {s_crit:.0f} tokens (B=1 compute-bound above this)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crossover detection benchmark for Smoothie derivative mode",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Model architecture preset",
    )
    parser.add_argument(
        "--api-model",
        help="Model name for the API (default: preset's default_api_name)",
    )
    parser.add_argument(
        "--hw",
        choices=list(HARDWARE.keys()),
        help="Hardware preset (default: auto-detect)",
    )
    parser.add_argument(
        "--url",
        default=SMOOTHIE_URL,
        help=f"Smoothie URL (default: {SMOOTHIE_URL})",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=CONCURRENCY,
        help=f"Concurrent workers (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=STABILIZE_SECS,
        help=f"Seconds per round (default: {STABILIZE_SECS})",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=SEQUENCE_LENGTHS,
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Max tokens per request (default: {MAX_TOKENS})",
    )
    args = parser.parse_args()

    model = MODELS[args.model]
    api_model = args.api_model or model["default_api_name"]

    # Resolve hardware.
    hw_name = args.hw
    if hw_name is None:
        hw_name = detect_chip()
        if hw_name is None:
            print("Could not auto-detect hardware. Use --hw to specify.", file=sys.stderr)
            sys.exit(1)
        print(f"  Detected hardware: {hw_name}")

    hw = HARDWARE[hw_name]

    # Load fill text for prompt construction.
    fill_text = load_fill_text()

    print(f"\n{'=' * 72}")
    print(f"  Crossover Benchmark")
    print(f"  Model: {args.model}  ({api_model})")
    print(f"  Hardware: {hw_name}")
    print(f"  Workers: {args.n}   Duration/round: {args.duration}s")
    print(f"  Sequence lengths: {args.seq_lens}")
    print(f"{'=' * 72}")

    print_theory_table(args.model, model, hw)

    results = []

    for seq_len in args.seq_lens:
        prompt = build_prompt(fill_text, seq_len)
        theory_b = theoretical_crossover(model, hw, seq_len)

        print(f"\n  --- seq_len={seq_len} (theory B={theory_b:.1f}) ---")
        print(f"  Prompt: {len(prompt)} chars, ~{len(prompt) // 4} tokens")
        print(f"  Running {args.n} workers for {args.duration}s ...", flush=True)

        log = await run_round(
            args.url,
            api_model,
            prompt,
            args.max_tokens,
            args.n,
            args.duration,
        )

        observed = log.stable_ceiling(window=args.duration / 2)
        total_429s = len(log.entries)

        if observed is not None:
            print(f"  Observed ceiling: {observed}  (from {total_429s} 429 responses)")
        else:
            print(f"  No ceiling observed ({total_429s} 429 responses)")

        results.append({
            "seq_len": seq_len,
            "observed": observed,
            "theory": theory_b,
            "total_429s": total_429s,
        })

        # Brief pause between rounds for streams to drain.
        await asyncio.sleep(2)

    print_results(args.model, hw_name, results)


if __name__ == "__main__":
    asyncio.run(main())
