#!/usr/bin/env python3
"""
Smoothie benchmark: compare 100 concurrent 2-turn LLM conversations
sent directly to Ollama vs routed through Smoothie.

Requirements:
    pip install aiohttp

Usage:
    # 1. Ensure Ollama is running with qwen2.5:3b pulled
    # 2. Run control + treatment:
    python bench/run.py
    # Or run only one leg:
    python bench/run.py --control-only
    python bench/run.py --treatment-only
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 80
NUM_TASKS = 100

BACKEND_URL = "http://127.0.0.1:8999"
SMOOTHIE_URL = "http://127.0.0.1:8081"

TURN1_PROMPT = "Count from 1 to 20."
TURN2_PROMPT = "Now count backwards from 20 to 1."

TIMEOUT = aiohttp.ClientTimeout(total=300, sock_read=120)
HEADERS = {"Connection": "close"}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    ttft_ms: float = 0.0
    itl_ms: list[float] = field(default_factory=list)
    token_count: int = 0
    content: str = ""


@dataclass
class TaskResult:
    task_id: int = 0
    wall_ms: float = 0.0
    turn1: TurnResult | None = None
    turn2: TurnResult | None = None
    retries: int = 0
    status: str = "success"  # success | error
    error: str = ""


# ---------------------------------------------------------------------------
# Streaming conversation
# ---------------------------------------------------------------------------


class Rejected(Exception):
    """Raised when the server returns 429."""

    def __init__(self, retry_after: float = 1.0):
        self.retry_after = retry_after


async def consume_stream(
    session: aiohttp.ClientSession,
    url: str,
    messages: list[dict],
) -> TurnResult:
    """Send a chat completion request and consume the SSE stream."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    result = TurnResult()
    first_token_time = None
    last_event_time = None
    start = time.monotonic()

    async with session.post(
        f"{url}/v1/chat/completions",
        json=payload,
        headers=HEADERS,
        timeout=TIMEOUT,
    ) as resp:
        if resp.status == 429:
            retry_after = float(resp.headers.get("Retry-After", "1"))
            raise Rejected(retry_after)
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")

        buffer = ""
        async for chunk_bytes in resp.content.iter_any():
            chunk = chunk_bytes.decode("utf-8", errors="replace")
            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
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
                content = delta.get("content", "")
                if not content:
                    continue

                now = time.monotonic()
                result.token_count += 1
                result.content += content

                if first_token_time is None:
                    first_token_time = now
                    result.ttft_ms = (now - start) * 1000
                else:
                    itl = (now - last_event_time) * 1000
                    result.itl_ms.append(itl)
                last_event_time = now

    return result


async def consume_stream_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    messages: list[dict],
    result: TaskResult,
) -> TurnResult:
    """Call consume_stream, retrying on 429 with backoff."""
    while True:
        try:
            return await consume_stream(session, url, messages)
        except Rejected as e:
            result.retries += 1
            await asyncio.sleep(e.retry_after)


async def run_conversation(
    session: aiohttp.ClientSession,
    url: str,
    task_id: int,
) -> TaskResult:
    """Execute a 2-turn conversation and collect metrics."""
    result = TaskResult(task_id=task_id)
    wall_start = time.monotonic()

    try:
        messages = [{"role": "user", "content": TURN1_PROMPT}]
        t1 = await consume_stream_with_retry(session, url, messages, result)
        result.turn1 = t1

        messages.append({"role": "assistant", "content": t1.content})
        messages.append({"role": "user", "content": TURN2_PROMPT})
        t2 = await consume_stream_with_retry(session, url, messages, result)
        result.turn2 = t2

    except Exception as e:
        result.status = "error"
        result.error = str(e)[:200]

    result.wall_ms = (time.monotonic() - wall_start) * 1000
    return result


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


async def run_scenario(
    label: str,
    url: str,
    n: int = NUM_TASKS,
) -> list[TaskResult]:
    """Launch n concurrent conversations and return results."""
    print(f"\n{'=' * 60}")
    print(f"  {label}:  {n} concurrent 2-turn conversations → {url}")
    print(f"{'=' * 60}")

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [run_conversation(session, url, i) for i in range(n)]
        results = await asyncio.gather(*tasks)

    return list(results)


# ---------------------------------------------------------------------------
# Stats and reporting
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def print_stats(label: str, results: list[TaskResult]) -> None:
    successes = [r for r in results if r.status == "success"]
    errors = [r for r in results if r.status == "error"]
    total_retries = sum(r.retries for r in results)

    print(f"\n--- {label} ---")
    print(f"  Total tasks:  {len(results)}")
    print(f"  Successes:    {len(successes)}")
    print(f"  Errors:       {len(errors)}")
    if total_retries:
        print(f"  Total 429 retries: {total_retries}")

    if errors:
        for e in errors[:3]:
            print(f"    task {e.task_id}: {e.error}")

    if not successes:
        print("  (no successful completions)")
        return

    # -- Completion times --
    walls = sorted(r.wall_ms for r in successes)
    print(f"\n  Completion time (ms):")
    print(f"    min:    {walls[0]:>10.1f}")
    print(f"    p50:    {percentile(walls, 50):>10.1f}")
    print(f"    p95:    {percentile(walls, 95):>10.1f}")
    print(f"    p99:    {percentile(walls, 99):>10.1f}")
    print(f"    max:    {walls[-1]:>10.1f}")
    if len(walls) > 1:
        print(f"    stddev: {statistics.stdev(walls):>10.1f}")

    # -- TTFT --
    ttfts_t1 = sorted(
        r.turn1.ttft_ms for r in successes if r.turn1 and r.turn1.ttft_ms > 0
    )
    ttfts_t2 = sorted(
        r.turn2.ttft_ms for r in successes if r.turn2 and r.turn2.ttft_ms > 0
    )
    if ttfts_t1:
        print(f"\n  TTFT turn 1 (ms):")
        print(f"    p50:    {percentile(ttfts_t1, 50):>10.1f}")
        print(f"    p95:    {percentile(ttfts_t1, 95):>10.1f}")
    if ttfts_t2:
        print(f"  TTFT turn 2 (ms):")
        print(f"    p50:    {percentile(ttfts_t2, 50):>10.1f}")
        print(f"    p95:    {percentile(ttfts_t2, 95):>10.1f}")

    # -- Inter-token latency --
    all_itl: list[float] = []
    for r in successes:
        if r.turn1:
            all_itl.extend(r.turn1.itl_ms)
        if r.turn2:
            all_itl.extend(r.turn2.itl_ms)
    if all_itl:
        all_itl.sort()
        print(f"\n  Inter-token latency (ms):")
        print(f"    p50:    {percentile(all_itl, 50):>10.1f}")
        print(f"    p95:    {percentile(all_itl, 95):>10.1f}")

    # -- Throughput --
    total_tokens = sum(
        (r.turn1.token_count if r.turn1 else 0)
        + (r.turn2.token_count if r.turn2 else 0)
        for r in successes
    )
    wall_clock = max(r.wall_ms for r in successes)
    if wall_clock > 0:
        tps = total_tokens / (wall_clock / 1000)
        print(f"\n  Throughput:")
        print(f"    Total tokens:     {total_tokens}")
        print(f"    Wall clock (s):   {wall_clock / 1000:.1f}")
        print(f"    Tokens/sec:       {tps:.1f}")


def print_histogram(label: str, results: list[TaskResult]) -> None:
    successes = [r for r in results if r.status == "success"]
    if not successes:
        return

    walls = [r.wall_ms for r in successes]
    lo = min(walls)
    hi = max(walls)

    num_buckets = 20
    if hi <= lo:
        bucket_width = 1.0
    else:
        bucket_width = (hi - lo) / num_buckets

    buckets = [0] * num_buckets
    for w in walls:
        idx = min(int((w - lo) / bucket_width), num_buckets - 1)
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1
    bar_width = 40

    print(f"\n  Completion time histogram ({label}):")
    for i, count in enumerate(buckets):
        left = lo + i * bucket_width
        bar_len = int(count / max_count * bar_width) if max_count else 0
        bar = "#" * bar_len
        print(f"    {left:>8.0f} ms |{bar:<{bar_width}} {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoothie benchmark")
    parser.add_argument(
        "--control-only", action="store_true", help="Run control leg only"
    )
    parser.add_argument(
        "--treatment-only", action="store_true", help="Run treatment leg only"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=NUM_TASKS,
        help=f"Number of concurrent tasks (default: {NUM_TASKS})",
    )
    args = parser.parse_args()

    control_results = None
    treatment_results = None

    if not args.treatment_only:
        control_results = await run_scenario(
            "CONTROL (direct → Ollama)", BACKEND_URL, args.n
        )

    if not args.control_only:
        if control_results is not None:
            print("\n")
            input(
                "Start smoothie-server, then press Enter to run the treatment leg..."
            )
        treatment_results = await run_scenario(
            "TREATMENT (through Smoothie)", SMOOTHIE_URL, args.n
        )

    # -- Print results --
    print(f"\n\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")

    if control_results:
        print_stats("Control (direct → Ollama)", control_results)
        print_histogram("Control", control_results)

    if treatment_results:
        print_stats("Treatment (through Smoothie)", treatment_results)
        print_histogram("Treatment", treatment_results)


if __name__ == "__main__":
    asyncio.run(main())
