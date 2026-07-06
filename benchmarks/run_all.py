#!/usr/bin/env python3
"""
Smoothie benchmark suite: characterize ITL vs concurrency and test the
floor-based AIMD controller.

Produces JSON data files and PNG charts in benchmarks/results/.

Usage:
    # 1. Start the inference backend:
    llama-server -m <model.gguf> --parallel 64 --port 8085 --ctx-size 65536 -ngl 99

    # 2. Start smoothie (for controller tests only):
    cargo run --release -p smoothie-server -- -c examples/configs/smoothie-bench.yaml

    # 3. Run everything:
    python benchmarks/run_all.py

    # Or run individual phases:
    python benchmarks/run_all.py --phase itl          # direct ITL vs batch size
    python benchmarks/run_all.py --phase controller   # floor-based controller convergence
    python benchmarks/run_all.py --phase charts       # regenerate charts from saved data
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = "http://127.0.0.1:8085"
SMOOTHIE_URL = "http://127.0.0.1:8081"
MODEL = "qwen2.5:3b"
PROMPT = "Count from 1 to 100, one number per line."
MAX_TOKENS = 80
TIMEOUT = aiohttp.ClientTimeout(total=300, sock_read=120)

ITL_BATCH_SIZES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 34, 36, 40, 44, 48, 52, 56, 60, 64]
ITL_SETTLE_SECS = 5
ITL_MEASURE_SECS = 12

CTRL_CONCURRENCY = 200
CTRL_DURATION = 180  # 3 minutes for controller convergence

# ---------------------------------------------------------------------------
# Phase 1: Direct ITL measurement
# ---------------------------------------------------------------------------


async def measure_itl_at(
    session: aiohttp.ClientSession,
    url: str,
    n: int,
    stop: asyncio.Event,
    itl_data: list[float],
) -> None:
    """Single worker: send streaming requests and collect inter-token latencies."""
    while not stop.is_set():
        try:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS,
                "stream": True,
            }
            async with session.post(
                f"{url}/v1/chat/completions", json=payload, timeout=TIMEOUT,
            ) as resp:
                if resp.status != 200:
                    await asyncio.sleep(0.5)
                    continue
                last_token_time = None
                buffer = ""
                async for chunk_bytes in resp.content.iter_any():
                    if stop.is_set():
                        return
                    chunk = chunk_bytes.decode("utf-8", errors="replace")
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        if not delta.get("content") and not delta.get("reasoning_content"):
                            continue
                        now = time.monotonic()
                        if last_token_time is not None:
                            itl_data.append((now - last_token_time) * 1000)
                        last_token_time = now
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(0.5)


async def run_itl_sweep(url: str) -> list[dict]:
    """Measure ITL at each batch size. Returns list of result dicts."""
    results = []
    for n in ITL_BATCH_SIZES:
        itl_data: list[float] = []
        stop = asyncio.Event()
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.create_task(measure_itl_at(session, url, n, stop, itl_data))
                for _ in range(n)
            ]
            await asyncio.sleep(ITL_SETTLE_SECS)
            itl_data.clear()
            await asyncio.sleep(ITL_MEASURE_SECS)
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)

        if not itl_data:
            results.append({"batch": n, "p50": 0, "p95": 0, "mean": 0, "samples": 0})
            continue

        itl_data.sort()
        p50 = itl_data[len(itl_data) // 2]
        p95 = itl_data[int(len(itl_data) * 0.95)]
        mean = statistics.mean(itl_data)
        results.append({
            "batch": n, "p50": round(p50, 2), "p95": round(p95, 2),
            "mean": round(mean, 2), "samples": len(itl_data),
        })
        print(f"    B={n:>3}  p50={p50:>7.1f}ms  p95={p95:>7.1f}ms  ({len(itl_data)} samples)")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Controller convergence
# ---------------------------------------------------------------------------


async def ctrl_worker(
    session: aiohttp.ClientSession,
    url: str,
    stop: asyncio.Event,
    ceiling_log: list,
) -> None:
    """Send requests through smoothie, record ceiling from 429 headers."""
    while not stop.is_set():
        try:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS,
                "stream": True,
            }
            async with session.post(
                f"{url}/v1/chat/completions", json=payload,
                headers={"Connection": "close"}, timeout=TIMEOUT,
            ) as resp:
                if resp.status == 429:
                    ceiling = int(resp.headers.get("X-Smoothie-Ceiling", "0"))
                    ceiling_log.append((time.monotonic(), ceiling))
                    retry = float(resp.headers.get("Retry-After", "1"))
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=retry)
                        return
                    except asyncio.TimeoutError:
                        pass
                elif resp.status == 200:
                    async for _ in resp.content.iter_any():
                        if stop.is_set():
                            return
                else:
                    await asyncio.sleep(1)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
                return
            except asyncio.TimeoutError:
                pass


async def run_controller_test(url: str, duration: int = CTRL_DURATION) -> list[dict]:
    """Run the controller convergence test. Returns time-series of ceiling."""
    ceiling_log: list[tuple[float, int]] = []
    stop = asyncio.Event()
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(ctrl_worker(session, url, stop, ceiling_log))
            for _ in range(CTRL_CONCURRENCY)
        ]
        t0 = time.monotonic()
        tick = 0
        while tick < duration:
            await asyncio.sleep(2)
            tick = time.monotonic() - t0
            recent = [c for t, c in ceiling_log if t >= time.monotonic() - 2]
            if recent:
                mode = Counter(recent).most_common(1)[0][0]
                print(f"    t={tick:5.0f}s  ceiling={mode:>3}  (range {min(recent)}-{max(recent)})")

        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Convert to relative time series (2-second buckets).
    if not ceiling_log:
        return []

    t0 = ceiling_log[0][0]
    series = []
    bucket_width = 2.0
    t_end = ceiling_log[-1][0]
    t = t0
    while t < t_end:
        bucket = [c for ts, c in ceiling_log if t <= ts < t + bucket_width]
        if bucket:
            mode = Counter(bucket).most_common(1)[0][0]
            series.append({
                "t": round(t - t0, 1),
                "ceiling": mode,
                "min": min(bucket),
                "max": max(bucket),
                "count": len(bucket),
            })
        t += bucket_width

    return series


# ---------------------------------------------------------------------------
# Phase 3: Charts
# ---------------------------------------------------------------------------


def generate_charts(itl_data: list[dict] | None, ctrl_data: list[dict] | None) -> None:
    """Generate PNG charts from saved data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f8",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })

    if itl_data and any(r["p50"] > 0 for r in itl_data):
        _chart_itl_curve(itl_data, plt)
        _chart_itl_regimes(itl_data, plt)

    if ctrl_data:
        _chart_controller(ctrl_data, plt)

    print(f"\n  Charts saved to {RESULTS_DIR}/")


def _chart_itl_curve(data: list[dict], plt) -> None:
    """ITL vs batch size with p50 and p95."""
    batches = [r["batch"] for r in data if r["p50"] > 0]
    p50s = [r["p50"] for r in data if r["p50"] > 0]
    p95s = [r["p95"] for r in data if r["p50"] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batches, p50s, "o-", color="#2563eb", linewidth=2, markersize=5, label="p50")
    ax.fill_between(batches, p50s, p95s, alpha=0.15, color="#2563eb", label="p50–p95 band")
    ax.plot(batches, p95s, "s--", color="#7c3aed", linewidth=1, markersize=3, label="p95")

    # Annotate the cliff (skip sequential regime — require batch > 10).
    for i in range(1, len(p50s)):
        if batches[i] <= 10:
            continue
        pct = (p50s[i] - p50s[i - 1]) / p50s[i - 1] * 100 if p50s[i - 1] > 0 else 0
        if pct > 30:
            ax.annotate(
                f"cliff\n+{pct:.0f}%",
                xy=(batches[i], p50s[i]),
                xytext=(batches[i] + 4, p50s[i] + 8),
                fontsize=10, color="#dc2626", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#dc2626"),
            )
            break

    ax.set_xlabel("Concurrent Batch Size")
    ax.set_ylabel("Inter-Token Latency (ms)")
    ax.set_title(f"ITL vs Concurrency — {MODEL} (direct, no proxy)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, max(batches) + 2)
    ax.set_ylim(0, max(p95s) * 1.15)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "itl_vs_batch.png", dpi=150)
    plt.close(fig)
    print(f"  Saved itl_vs_batch.png")


def _chart_itl_regimes(data: list[dict], plt) -> None:
    """ITL curve annotated with operating regimes."""
    batches = [r["batch"] for r in data if r["p50"] > 0]
    p50s = [r["p50"] for r in data if r["p50"] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Find regime boundaries (skip sequential regime at low batch sizes).
    cliff_idx = None
    onset_idx = None
    for i in range(1, len(p50s)):
        pct = (p50s[i] - p50s[i - 1]) / p50s[i - 1] * 100 if p50s[i - 1] > 0 else 0
        if pct < -15 and onset_idx is None:
            onset_idx = i
        if batches[i] > 10 and pct > 30 and cliff_idx is None:
            cliff_idx = i

    # Color the regimes.
    if onset_idx and cliff_idx:
        ax.axvspan(0, batches[onset_idx], alpha=0.08, color="#ef4444", label="Sequential (no batching)")
        ax.axvspan(batches[onset_idx], batches[cliff_idx], alpha=0.08, color="#22c55e", label="Efficient plateau")
        ax.axvspan(batches[cliff_idx], max(batches) + 2, alpha=0.08, color="#f59e0b", label="Over capacity")

    ax.plot(batches, p50s, "o-", color="#1e293b", linewidth=2.5, markersize=6)

    # Mark the sweet spot.
    if onset_idx and cliff_idx:
        sweet_start = batches[onset_idx]
        sweet_end = batches[cliff_idx]
        mid = (sweet_start + sweet_end) / 2
        mid_itl = min(p50s[onset_idx:cliff_idx]) if onset_idx < cliff_idx else p50s[onset_idx]
        ax.annotate(
            f"Sweet spot\nB={sweet_start}-{sweet_end}",
            xy=(mid, mid_itl), xytext=(mid + 8, mid_itl + 20),
            fontsize=11, fontweight="bold", color="#15803d",
            arrowprops=dict(arrowstyle="->", color="#15803d"),
        )

    ax.set_xlabel("Concurrent Batch Size")
    ax.set_ylabel("Inter-Token Latency p50 (ms)")
    ax.set_title(f"Operating Regimes — {MODEL} on llama.cpp / Apple Metal")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, max(batches) + 2)
    ax.set_ylim(0, max(p50s) * 1.15)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "itl_regimes.png", dpi=150)
    plt.close(fig)
    print(f"  Saved itl_regimes.png")


def _chart_controller(data: list[dict], plt) -> None:
    """Controller ceiling over time."""
    ts = [r["t"] for r in data]
    ceilings = [r["ceiling"] for r in data]
    mins = [r["min"] for r in data]
    maxs = [r["max"] for r in data]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(ts, mins, maxs, alpha=0.15, color="#2563eb", label="min–max range")
    ax.plot(ts, ceilings, "-", color="#2563eb", linewidth=2, label="ceiling (mode)")

    # Equilibrium line.
    last_quarter = ceilings[len(ceilings) * 3 // 4:]
    if last_quarter:
        eq = Counter(last_quarter).most_common(1)[0][0]
        ax.axhline(y=eq, color="#dc2626", linestyle="--", linewidth=1, alpha=0.7, label=f"equilibrium ≈ {eq}")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Admission Ceiling")
    ax.set_title(f"Floor-Based Controller Convergence — {MODEL}")
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(ts))
    ax.set_ylim(0, max(maxs) * 1.15 if maxs else 64)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "controller_convergence.png", dpi=150)
    plt.close(fig)
    print(f"  Saved controller_convergence.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    global MODEL

    parser = argparse.ArgumentParser(description="Smoothie benchmark suite")
    parser.add_argument(
        "--phase", choices=["itl", "controller", "charts", "all"], default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument("--backend-url", default=BACKEND_URL)
    parser.add_argument("--smoothie-url", default=SMOOTHIE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ctrl-duration", type=int, default=CTRL_DURATION)
    args = parser.parse_args()

    MODEL = args.model

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    itl_data = None
    ctrl_data = None

    # Phase 1: Direct ITL sweep.
    if args.phase in ("itl", "all"):
        print(f"\n{'=' * 60}")
        print(f"  Phase 1: Direct ITL vs Batch Size")
        print(f"  Backend: {args.backend_url}   Model: {MODEL}")
        print(f"{'=' * 60}\n")

        itl_data = await run_itl_sweep(args.backend_url)

        out = RESULTS_DIR / "itl_sweep.json"
        out.write_text(json.dumps(itl_data, indent=2))
        print(f"\n  Data saved to {out}")

    # Phase 2: Controller convergence.
    if args.phase in ("controller", "all"):
        print(f"\n{'=' * 60}")
        print(f"  Phase 2: Controller Convergence")
        print(f"  Smoothie: {args.smoothie_url}   Duration: {args.ctrl_duration}s")
        print(f"{'=' * 60}\n")

        ctrl_data = await run_controller_test(args.smoothie_url, args.ctrl_duration)

        out = RESULTS_DIR / "controller_timeline.json"
        out.write_text(json.dumps(ctrl_data, indent=2))
        print(f"\n  Data saved to {out}")

    # Phase 3: Generate charts.
    if args.phase in ("charts", "all"):
        # Load from files if not already in memory.
        if itl_data is None:
            p = RESULTS_DIR / "itl_sweep.json"
            if p.exists():
                itl_data = json.loads(p.read_text())
        if ctrl_data is None:
            p = RESULTS_DIR / "controller_timeline.json"
            if p.exists():
                ctrl_data = json.loads(p.read_text())

        print(f"\n{'=' * 60}")
        print(f"  Phase 3: Generating Charts")
        print(f"{'=' * 60}\n")

        generate_charts(itl_data, ctrl_data)


if __name__ == "__main__":
    asyncio.run(main())
