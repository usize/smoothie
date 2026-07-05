#!/usr/bin/env python3
"""
Test vLLM priority scheduling with diverse, long prompts.

Each request gets a unique passage from a book as context,
creating distinct KV cache entries. High-priority requests
should preempt low-priority ones under memory pressure.

Usage:
    python bench/priority_test.py
"""

import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen2.5-3B-Instruct"
VLLM_URL = "http://127.0.0.1:8999"
MAX_TOKENS = 60
CONTEXT_TOKENS = 2000  # ~8000 chars per passage
NUM_HIGH = 1
NUM_LOW = 99
TIMEOUT = aiohttp.ClientTimeout(total=300, sock_read=120)
HEADERS = {"Connection": "close"}

# ---------------------------------------------------------------------------
# Load book and extract passages
# ---------------------------------------------------------------------------

def load_passages(n: int, chars_per_passage: int = 8000) -> list[str]:
    with open("bench/book.txt") as f:
        text = f.read()
    rng = random.Random(42)
    passages = []
    max_start = len(text) - chars_per_passage
    for _ in range(n):
        start = rng.randint(0, max_start)
        passages.append(text[start : start + chars_per_passage])
    return passages


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Result:
    task_id: int = 0
    priority: int = 0
    label: str = ""
    wall_ms: float = 0.0
    ttft_ms: float = 0.0
    itl_ms: list[float] = field(default_factory=list)
    token_count: int = 0
    status: str = "success"
    error: str = ""


# ---------------------------------------------------------------------------
# Streaming request
# ---------------------------------------------------------------------------

async def run_request(
    session: aiohttp.ClientSession,
    passage: str,
    priority: int,
    task_id: int,
    label: str,
) -> Result:
    result = Result(task_id=task_id, priority=priority, label=label)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Read this passage and write a one-sentence summary:\n\n{passage}",
            }
        ],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "priority": priority,
    }

    wall_start = time.monotonic()
    first_token_time = None
    last_event_time = None

    try:
        async with session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            headers=HEADERS,
            timeout=TIMEOUT,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.status = "error"
                result.error = f"HTTP {resp.status}: {body[:200]}"
                result.wall_ms = (time.monotonic() - wall_start) * 1000
                return result

            async for chunk_bytes in resp.content.iter_any():
                for line in chunk_bytes.decode("utf-8", errors="replace").split("\n"):
                    line = line.strip()
                    if not line.startswith("data:") or "[DONE]" in line:
                        continue
                    try:
                        data = json.loads(line[5:].strip())
                        content = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if not content:
                            continue
                    except (json.JSONDecodeError, IndexError):
                        continue

                    now = time.monotonic()
                    result.token_count += 1
                    if first_token_time is None:
                        first_token_time = now
                        result.ttft_ms = (now - wall_start) * 1000
                    elif last_event_time:
                        result.itl_ms.append((now - last_event_time) * 1000)
                    last_event_time = now

    except Exception as e:
        result.status = "error"
        result.error = str(e)[:200]

    result.wall_ms = (time.monotonic() - wall_start) * 1000
    return result


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f, c = int(math.floor(k)), int(math.ceil(k))
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def print_group_stats(label: str, results: list[Result]) -> None:
    ok = [r for r in results if r.status == "success"]
    print(f"\n  {label} ({len(ok)}/{len(results)} succeeded)")
    if not ok:
        for r in results[:3]:
            print(f"    {r.error}")
        return

    walls = sorted(r.wall_ms for r in ok)
    ttfts = sorted(r.ttft_ms for r in ok if r.ttft_ms > 0)
    all_itl: list[float] = []
    for r in ok:
        all_itl.extend(r.itl_ms)
    all_itl.sort()
    total_tokens = sum(r.token_count for r in ok)

    print(f"    Wall time:   min={walls[0]:.0f}  p50={percentile(walls, 50):.0f}  p95={percentile(walls, 95):.0f}  max={walls[-1]:.0f} ms")
    if ttfts:
        print(f"    TTFT:        p50={percentile(ttfts, 50):.0f}  p95={percentile(ttfts, 95):.0f} ms")
    if all_itl:
        print(f"    ITL:         p50={percentile(all_itl, 50):.1f}  p95={percentile(all_itl, 95):.1f} ms")
        print(f"    Tok/s/stream: ~{1000 / percentile(all_itl, 50):.1f}")
    print(f"    Tokens:      {total_tokens}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    total = NUM_HIGH + NUM_LOW
    passages = load_passages(total)

    print(f"Launching {NUM_HIGH} high-priority (0) + {NUM_LOW} low-priority (10) requests")
    print(f"Each with ~{CONTEXT_TOKENS} tokens of unique book context, max_tokens={MAX_TOKENS}")

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(NUM_HIGH):
            tasks.append(
                run_request(session, passages[i], priority=0, task_id=i, label="HIGH")
            )
        for i in range(NUM_LOW):
            tasks.append(
                run_request(
                    session, passages[NUM_HIGH + i], priority=10, task_id=NUM_HIGH + i, label="LOW"
                )
            )
        results = await asyncio.gather(*tasks)

    results = list(results)
    high = [r for r in results if r.label == "HIGH"]
    low = [r for r in results if r.label == "LOW"]

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {total} concurrent requests with priority scheduling")
    print(f"{'=' * 60}")
    print_group_stats("HIGH priority (0)", high)
    print_group_stats("LOW priority (10)", low)

    # Overall
    all_ok = [r for r in results if r.status == "success"]
    if all_ok:
        wall = max(r.wall_ms for r in all_ok)
        tokens = sum(r.token_count for r in all_ok)
        print(f"\n  Aggregate: {tokens} tokens in {wall/1000:.1f}s = {tokens/(wall/1000):.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
