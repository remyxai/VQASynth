# Visual Credit Audit — does the image actually earn the answer?

**Status:** experimental. Runnable wiring of `vqasynth.visual_credit`: take a
VLM's answers on a set of spatial yes/no (or multi-choice) questions under the
real image *and* under no-image controls (text-only, blank), then compute
**dependence-credited correctness (D-CC)** and the **correct-but-uncredited**
rate — how many "right" answers the image genuinely earned versus how many the
model would have produced with no image at all.

Adapted from *Visual Credit Audit for Multimodal Spatial Reasoning*
([arXiv:2607.27069](https://arxiv.org/abs/2607.27069)). The pure audit logic
(decision extraction, gold alignment, the D-CC / uncredited decomposition, the
label-free decision-change audit) lives in `vqasynth.visual_credit` and is
unit-tested in `tests/test_visual_credit.py`; this package only owns I/O
(prediction-JSONL read, optional blank-image materialization, report printing).
Gold alignment and forced-choice decision extraction reuse the existing
extractors in `vqasynth.evaluation`; the per-question-type breakdown reuses
`vqasynth.evaluation.classify_question`. No changes to the `vqasynth/` core.

## What this is (and isn't)

This is an **adapted port** of the paper's core mechanism, not a reproduction
of its experiments:

- **Kept at full fidelity** — the D-CC decomposition (correct on the real image
  *and* not already correct under a no-image control → image-attributable gain),
  the correct-but-uncredited failure mode, and the label-free decision-change
  audit (does the image move the declared decision relative to a control?).
- **Substituted / out of scope** — the paper's MLLM-inference harness (running
  four open MLLMs) is replaced by consuming prediction JSONL you collect
  yourself, the same split as `experiments/prometheus_space_judge`. The paper's
  *validation* experiments (matched image permutation, fixed-pixel relation
  contrasts, the 3×3 evidence-source factorial) validate the metric rather than
  define it and are intentionally not ported here.

## Prerequisites

- **Python 3.10+**
- `vqasynth` installed (`pip install -e .` from the repo root) — provides
  `vqasynth.visual_credit` and `vqasynth.evaluation`

## Install

```bash
pip install -e .                       # VQASynth core (incl. vqasynth.visual_credit)
```

## 1. Collect predictions (maintainer-run)

Run your VLM once per item under each condition and collect the free-text
answers into a JSONL — one record per item:

```json
{"question": "Is the cup to the left of the book?", "gold": "Yes",
 "pred_real": "Yes, the cup is to the left.",
 "pred_text_only": "Yes.",
 "pred_blank": "No."}
```

`pred_real` is the answer on the benchmark image; `pred_text_only` /
`pred_blank` are the answers with no image / a blank image attached. At least
one control is required per item. Use the `controls` subcommand to emit the
control prompts (and optionally materialize blank PNGs) from an items file:

```bash
python -m experiments.visual_credit_audit.run controls \
    --items items.jsonl --image-dir blank_controls --emit-images
```

## 2. Audit

```bash
python -m experiments.visual_credit_audit.run audit \
    --predictions predictions.jsonl --breakdown
```

This prints D-CC, the correct-but-uncredited rate (overall and as a fraction of
correct decisions), image gain over the controls, and the label-free
decision-change rate; `--breakdown` additionally groups credit by question type
via `vqasynth.evaluation.classify_question`. Pass `--output report.json` to
write the summary as JSON.

### Reading the numbers

- **D-CC** is the share of decisions that are correct *because of* the image.
- **Correct but uncredited** is the share of decisions that are right on the
  image but the model would have answered identically with no image — the
  benchmark is rewarding them, but the image added no support (the paper finds
  12.73–26.25% of decisions land here).
- **Decision change rate (label-free)** needs no ground truth: it is the share
  of items where the image moved the declared decision away from a no-image
  control.

## Testing

Structural tests for the audit logic run without CUDA or model weights
(CPU-only, Python 3.10):

```bash
pytest tests/test_visual_credit.py
```
