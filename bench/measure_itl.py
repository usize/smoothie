#!/usr/bin/env python3
"""
Direct ITL measurement at different concurrency levels.

Bypasses smoothie entirely — sends requests directly to llama-server
to measure how inter-token latency changes with batch size.

This answers the fundamental question: does ITL actually increase
measurably as we add concurrent sequences?
"""

import asyncio
import json
import statistics
import sys
import time

import aiohttp

URL = "http://127.0.0.1:8085"
MODEL = "qwen2.5:3b"
PROMPT = "Count from 1 to 100, one number per line."
MAX_TOKENS = 60
SETTLE_SECS = 5
MEASURE_SECS = 10

BATCH_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
TIMEOUT = aiohttp.ClientTimeout(total=300, sock_read=120)


async def measure_itl(
    session: aiohttp.ClientSession,
    stop: asyncio.Event,
    itl_collector: list[float],
) -> None:
    """Send requests in a loop, collecting ITL measurements."""
    while not stop.is_set():
        try:
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOKENS,
                "stream": True,
            }
            async with session.post(
                f"{URL}/v1/chat/completions",
                json=payload,
                timeout=TIMEOUT,
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
                        if not delta.get("content"):
                            continue

                        now = time.monotonic()
                        if last_token_time is not None:
                            itl_ms = (now - last_token_time) * 1000
                            itl_collector.append(itl_ms)
                        last_token_time = now
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(0.5)


async def run_at_concurrency(n: int) -> dict:
    """Run n concurrent streams and measure ITL."""
    itl_data: list[float] = []
    stop = asyncio.Event()
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(measure_itl(session, stop, itl_data))
            for _ in range(n)
        ]
        # Settle, then measure.
        await asyncio.sleep(SETTLE_SECS)
        itl_data.clear()
        await asyncio.sleep(MEASURE_SECS)
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    if not itl_data:
        return {"n": n, "p50": 0, "p95": 0, "mean": 0, "count": 0}

    itl_data.sort()
    p50 = itl_data[len(itl_data) // 2]
    p95 = itl_data[int(len(itl_data) * 0.95)]
    mean = statistics.mean(itl_data)

    return {"n": n, "p50": p50, "p95": p95, "mean": mean, "count": len(itl_data)}


async def main() -> None:
    batch_sizes = BATCH_SIZES
    if len(sys.argv) > 1:
        batch_sizes = [int(x) for x in sys.argv[1:]]

    print(f"\n{'=' * 66}")
    print(f"  Direct ITL vs Concurrency — {MODEL}")
    print(f"  {SETTLE_SECS}s settle + {MEASURE_SECS}s measure per level")
    print(f"{'=' * 66}")
    print(f"  {'batch':>6}  {'p50 ms':>8}  {'p95 ms':>8}  {'mean ms':>8}  {'samples':>8}  {'delta':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    prev_p50 = None
    for n in batch_sizes:
        result = await run_at_concurrency(n)
        delta = ""
        if prev_p50 and prev_p50 > 0 and result["p50"] > 0:
            pct = (result["p50"] - prev_p50) / prev_p50 * 100
            delta = f"{pct:+.1f}%"
        prev_p50 = result["p50"]
        print(
            f"  {result['n']:>6}  {result['p50']:>8.1f}  {result['p95']:>8.1f}"
            f"  {result['mean']:>8.1f}  {result['count']:>8}  {delta:>8}"
        )

    print()


if __name__ == "__main__":
    asyncio.run(main())
