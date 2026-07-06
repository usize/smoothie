# Smoothie Benchmarks

Characterize the ITL-vs-concurrency curve for a given model/hardware
combination, then verify that Smoothie's floor-based AIMD controller
finds the efficient operating region.

## Prerequisites

```
pip install aiohttp matplotlib
cargo build --release -p smoothie-server
```

## Quick Start

```bash
# 1. Start inference backend
llama-server \
  -m <model.gguf> \
  --parallel 64 \
  --port 8085 \
  --ctx-size 65536 \
  -ngl 99

# 2. Start smoothie (for controller test)
cargo run --release -p smoothie-server -- \
  -c examples/configs/smoothie-bench.yaml

# 3. Run full suite
python benchmarks/run_all.py
```

## Phases

| Phase | What it does | Requires |
|-------|-------------|----------|
| `--phase itl` | Measures ITL at each batch size directly against the backend | llama-server |
| `--phase controller` | Observes ceiling convergence through smoothie | llama-server + smoothie |
| `--phase charts` | Regenerates PNGs from saved JSON data | neither |

## Output

All output goes to `benchmarks/results/`:

- `itl_sweep.json` — raw ITL measurements per batch size
- `controller_timeline.json` — ceiling over time
- `itl_vs_batch.png` — ITL curve with p50/p95 band
- `itl_regimes.png` — annotated operating regimes (sequential / plateau / over-capacity)
- `controller_convergence.png` — ceiling timeline with equilibrium line

## Tuning the Config

The controller config lives in `examples/configs/smoothie-bench.yaml`.
Key parameters:

- **`floor_tps`**: Target per-stream decode rate. Set this above the
  plateau ITL but below the cliff ITL. Use the ITL sweep to find these.
- **`hysteresis_steps`**: Observations before acting. Higher = smoother
  but slower convergence. 200 works well for typical observation rates.
- **`beta`**: Multiplicative decrease factor. 0.8 = gentle (20% cut),
  0.5 = aggressive (50% cut). Gentler beta gives tighter oscillation
  but slower recovery from the cliff.
- **`headroom_ms`**: Dead zone width below the floor. Prevents
  oscillation at equilibrium.
