#!/usr/bin/env python3
"""
Floor-based controller test: observe ceiling convergence and stability.

Sends concurrent streaming requests through smoothie and tracks how the
AIMD ceiling evolves from 429 response headers. Reports the ceiling
timeline and final equilibrium.

Usage:
    python bench/floor_test.py [--duration 60] [-n 200]
"""

import asyncio
import json
import sys
import time
from collections import Counter

import aiohttp

SMOOTHIE_URL = "http://127.0.0.1:8081"
MODEL = "qwen2.5:3b"
PROMPT = "Count from 1 to 100, one number per line."
MAX_TOKENS = 80
CONCURRENCY = 200
DURATION = 60

TIMEOUT = aiohttp.ClientTimeout(total=300, sock_read=120)


async def worker(
    session: aiohttp.ClientSession,
    stop: asyncio.Event,
    ceiling_log: list,
) -> None:
    """Send requests in a loop, recording ceiling from 429s."""
    while not stop.is_set():
        try:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS,
                "stream": True,
            }
            async with session.post(
                f"{SMOOTHIE_URL}/v1/chat/completions",
                json=payload,
                headers={"Connection": "close"},
                timeout=TIMEOUT,
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


async def main() -> None:
    duration = DURATION
    concurrency = CONCURRENCY
    if "--duration" in sys.argv:
        idx = sys.argv.index("--duration")
        duration = int(sys.argv[idx + 1])
    if "-n" in sys.argv:
        idx = sys.argv.index("-n")
        concurrency = int(sys.argv[idx + 1])

    print(f"\n{'=' * 60}")
    print(f"  Floor-Based Controller Test")
    print(f"  Model: {MODEL}   Workers: {concurrency}   Duration: {duration}s")
    print(f"{'=' * 60}\n")

    ceiling_log: list[tuple[float, int]] = []
    stop = asyncio.Event()
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(worker(session, stop, ceiling_log))
            for _ in range(concurrency)
        ]

        t0 = time.monotonic()

        # Print ceiling timeline every 5 seconds.
        for tick in range(0, duration, 5):
            await asyncio.sleep(5)
            elapsed = time.monotonic() - t0
            recent = [c for t, c in ceiling_log if t >= time.monotonic() - 5]
            if recent:
                mode = Counter(recent).most_common(1)[0][0]
                lo, hi = min(recent), max(recent)
                print(f"  t={elapsed:5.0f}s  ceiling={mode:>3}"
                      f"  (range {lo}-{hi}, {len(recent)} 429s)")
            else:
                print(f"  t={elapsed:5.0f}s  (no 429s — ceiling may be above concurrency)")

        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Summary.
    if not ceiling_log:
        print("\n  No 429 responses observed. Ceiling never saturated.")
        return

    # Final equilibrium: mode of last 20% of observations.
    cutoff = time.monotonic() - duration * 0.2
    final = [c for t, c in ceiling_log if t >= cutoff]
    if final:
        eq = Counter(final).most_common(1)[0][0]
    else:
        eq = ceiling_log[-1][1]

    all_ceilings = [c for _, c in ceiling_log]
    print(f"\n  Equilibrium ceiling: {eq}")
    print(f"  Total 429s: {len(ceiling_log)}")
    print(f"  Ceiling range: {min(all_ceilings)}-{max(all_ceilings)}")

    # Floor interpretation.
    floor_ms = 1000.0 / 12.0
    print(f"\n  Floor: 12.0 tok/s = {floor_ms:.0f}ms ITL target")
    print(f"  Controller held ceiling at {eq} concurrent streams")
    print()


if __name__ == "__main__":
    asyncio.run(main())
