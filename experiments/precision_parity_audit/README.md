# Precision Parity Audit — what quantization breaks that the score hides

**Status:** experimental. Runnable wiring of `vqasynth.precision_parity`: run the
same VLM over the same benchmark items at a reference precision (FP16/BF16) and
under post-training quantization (INT8/NF4), then compare the two runs
**item-by-item** rather than reading two aggregate accuracies side by side.

Adapted from *GHOST-Q: Towards Studying Grounding Hallucinations Overlooked
Under Same-score TradeOffs in Quantized VLMS*
([arXiv:2609.29999](https://arxiv.org/abs/2609.29999)). The pure audit logic
(paired outcome flips, the same-score-tradeoff flag, the exact sign test and
Benjamini-Hochberg FDR correction) lives in `vqasynth.precision_parity` and is
unit-tested in `tests/test_precision_parity.py`; this package only owns I/O
(benchmark loading, two-precision inference, report printing). Precision
loading (INT8/NF4 via bitsandbytes) is exposed by `vqasynth.inference`
(`run_inference_on_benchmark(precision=...)`), and the audit plugs into the
existing evaluation stage as `BenchmarkRunner.score_parity`.

## What this is (and isn't)

This is an **adapted port** of the paper's core mechanism, not a reproduction
of its experiments:

- **Kept at full fidelity** — the paired FP16-vs-quantized item-by-item
  comparison; the decomposition into grounding successes lost vs failures
  rescued; the same-score-tradeoff criterion (aggregate accuracy held within
  ±2 percentage points while paired outcomes flip); the exact sign test on
  discordant pairs with BH-FDR correction across the per-category breakdown;
  the INT8/NF4 precision grid.
- **Substituted / out of scope** — the paper's A100 memory/latency profiling
  (needs a dedicated hardware bench) and its open-ended AMBER generation-budget
  audit (needs an open-ended-generation benchmark the repo does not host).
  The paper's three 8B model families are replaced by any HF model slug you
  pass; scoring reuses the repo's existing benchmark-native scorers.

## Prerequisites

- **Python 3.10+**, CUDA GPU + `bitsandbytes` for the quantized runs
- `vqasynth` installed (`pip install -e .` from the repo root) — provides
  `vqasynth.precision_parity`, `vqasynth.inference`, `vqasynth.benchmarks`

## 1. Compare two precisions end-to-end (maintainer-run, CUDA)

```bash
python -m experiments.precision_parity_audit.run compare \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --benchmark spatialscore \
    --baseline-precision fp16 \
    --quantized-precision nf4 \
    --limit 200 --output-dir parity_out --breakdown-by category
```

Runs inference twice over the same items, writes one prediction JSONL per
precision (`fp16_predictions.jsonl`, `nf4_predictions.jsonl`), then prints the
paired audit. `--output report.json` persists the summary.

## 2. Audit predictions you already collected (no GPU)

```bash
python -m experiments.precision_parity_audit.run audit \
    --baseline-predictions parity_out/fp16_predictions.jsonl \
    --quantized-predictions parity_out/nf4_predictions.jsonl \
    --benchmark spatialscore --breakdown-by category
```

Each JSONL is one `{"id": ..., "prediction": ...}` record per item. With
`--benchmark`, pairs are scored by that benchmark's native scorer; with
`--items items.jsonl` (id / question / answer), by the default gold-alignment
extractors shared with `vqasynth.visual_credit`.

### Reading the numbers

- **Correct under baseline only** — grounding successes quantization lost;
  GHOST-Q finds these cluster on hallucination-sensitive conditions even when
  the headline score is preserved.
- **Correct under quantized only** — failures the compression happened to fix.
  Net accuracy can be ~0 while both counts are large.
- **Same-score tradeoff** — flagged when the net accuracy delta stays within
  ±2pp while flips occurred: the aggregate score conceals the redistribution.
- **Flip asymmetry p / q(FDR)** — exact two-sided sign test on the discordant
  pairs; `q` is the Benjamini-Hochberg-corrected value across the breakdown,
  so per-category effects are comparable to the paper's FDR-thresholded grid.

## Testing

Structural tests for the audit logic and its integration with the
pre-existing evaluation stack run without CUDA or model weights:

```bash
pytest tests/test_precision_parity.py
```
