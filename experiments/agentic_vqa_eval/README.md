# Agentic VQA Eval — how does the model hold up outside spatial-VQA datasets?

**Status:** experimental. Runnable wiring of `vqasynth.agentic_vqa`: register
[AgentVQA](https://arxiv.org/abs/2610.31494) as a fifth benchmark in the
multi-benchmark evaluation stage (`vqasynth.benchmarks`), then score a VLM on
it with the same `BenchmarkRunner` the SpatialScore / OmniSpatial / SpaCE-10 /
MindCube stage already uses.

AgentVQA is 1,000 human-verified multiple-choice questions curated from 23
source datasets across five agentic domains — **Web Agents, Robotics,
Egocentric Videos, Games, Spatial Planning**. The paper's cross-domain finding
is the reason to run it: across 22 models (best: 58.5%), errors are driven
primarily by failures to *ground and maintain the visual state* — localizing
UI elements, interpreting motion, tracking state changes, and inferring 3D
spatial relations — rather than by task misinterpretation or downstream logic.
The Spatial Planning / Robotics domains are exactly the failure surface
VQASynth's synthetic spatial data and CoT templates target, so the per-domain
breakdown is the interesting number, not the overall accuracy.

Adapted from *AgentVQA: A Unified Benchmark for Agentic Visual Understanding*
([arXiv:2610.31494](https://arxiv.org/abs/2610.31494)). The loader, scorer and
registry wiring live in `vqasynth.agentic_vqa` and are unit-tested in
`tests/test_agentic_vqa.py`; this package only owns I/O (CLI, prediction-JSONL
read, report printing). Items normalize to the common schema in
`vqasynth.benchmarks` and score through the existing
`vqasynth.evaluation.extract_option` path. No changes to the `vqasynth/` core —
`register_agentvqa()` adds the `"agentvqa"` entries to the existing registries
at runtime, so `BenchmarkRunner("all")` picks the benchmark up too.

## What this is (and isn't)

This is an **adapted port**, not a reproduction of the paper's experiments:

- **Kept at full fidelity** — the benchmark itself (all 23 subsets, the
  five-domain taxonomy, letter-exact multi-choice scoring), the pointing
  options that carry the paper's "localizing UI elements" failure mode
  (rendered as `(x, y)` text so letter prompting still works), and the
  per-domain / per-source-dataset breakdowns.
- **Substituted / out of scope** — the paper's own 22-model harness and its
  online-evaluation correlation analysis are replaced by this repo's existing
  inference path (`vqasynth.inference.run_inference_on_benchmark`) or by
  predictions you collect yourself. Video-only subsets (e.g. VSI-Bench,
  MarioQA) reference separate `media/` files rather than embedded stills and
  are dropped by default — single-image inference cannot consume them; pass
  `--include-video-items` to keep them. Fine-tuning-on-VQASynth-data-then-
  re-evaluating (the paper-driven experiment) is a follow-on, not part of this
  wiring.

Note that two AgentVQA subsets (`OmniSpatial`, `SpaCE-10`) are re-curated cuts
of benchmarks the eval stage already loads under their own names — use
`--subsets` to avoid double counting.

## Prerequisites

- **Python 3.10+**
- `vqasynth` installed (`pip install -e .` from the repo root)
- First run downloads the benchmark from the `AgentVQA/AgentVQA` HF dataset
  (~26 GB with media; the image-backed subsets are much smaller — load them
  selectively with `--domains` / `--subsets`). The AgentVQA repo lists no
  license of its own; each subset inherits its source dataset's terms.

## Usage

List the domain → subset taxonomy:

```bash
python -m experiments.agentic_vqa_eval.run --list-subsets
```

Score a HuggingFace VLM on the whole benchmark (or a slice):

```bash
python -m experiments.agentic_vqa_eval.run \
    --model remyxai/SpaceThinker-Qwen2.5VL-3B --max-items 200
```

Just the domains VQASynth's data targets:

```bash
python -m experiments.agentic_vqa_eval.run \
    --model <hf-slug> --domains "Spatial Planning,Robotics"
```

Or score predictions collected with any other harness — emit the item cut
first, then feed the answers back (one JSONL record per item, `id` plus any of
`prediction` / `pred` / `answer` / `output`):

```bash
python -m experiments.agentic_vqa_eval.run \
    --emit-items items.jsonl --predictions predictions.jsonl --output report.json
```

The report is the same shape as the other benchmark reports: overall accuracy,
`by_category` (agentic domain), `by_subcategory` (source dataset), plus
`per_item` when `--output` is set.

## Testing

Structural tests for the loader, scorer and registry wiring run without CUDA,
model weights, network, or a working `datasets` install (CPU-only, Python 3.10):

```bash
pytest tests/test_agentic_vqa.py
```
