# SpatialAnnotator — NOOA-based agent for open-ended spatial annotation

**Status:** experimental. Wraps VQASynth's tool inventory as a NOOA
[Object-Oriented Agent](https://github.com/NVIDIA-NeMo/labs-OO-Agents) so an
LLM can dynamically select + sequence tool calls per-sample instead of
following a pre-templated pipeline. Intended for offline labeling scenarios
where the input question isn't known at pipeline-design time.

Design context: [VQASynth Issue #106](https://github.com/remyxai/VQASynth/issues/106).

## Prerequisites

- **Python 3.12+** (NOOA constraint)
- LLM provider API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`)
- Model weights per tier (see below) — auto-downloaded on first use

## Resource tiers

The agent picks a tier automatically based on CUDA availability + VRAM. Force
with `VQASYNTH_AGENT_TIER=cpu|gpu`.

| Capability | CPU tier | GPU tier (≥12 GB VRAM) |
|---|---|---|
| Detection | Florence-2-base → Florence-2-large cascade | Florence-2 (same, GPU-served) |
| Segmentation | Florence-2 refer-segment (polygon summary) | Florence-2 (same, GPU-served) |
| Scene captioning | Florence-2 `<CAPTION>` / `<DETAILED_CAPTION>` | Florence-2 (same, GPU-served) |
| Full object detection | Florence-2 `<OD>` | Same |
| OCR | Florence-2 `<OCR>` / `<OCR_WITH_REGION>` | Same |
| Metric depth | **DepthPro** (`apple/DepthPro`, ~330M, metric + intrinsics) | **VGGT-1B** (via `vqasynth.scene_fusion.SpatialSceneConstructor`) |

The CPU-tier depth uses DepthPro rather than Depth Anything V2 because
DepthPro outputs metric depth (meters) + focal length natively — no
scale-calibration step to hallucinate around.

Future tiers not in the initial branch:
- Molmo-tier detection for questions requiring pointing rather than boxes
- SAM2 pixel-accurate masks (currently Florence-2 poly summary only)
- Cross-tier auto-escalation (fallback CPU → GPU if a run stalls)

## Install

```bash
# Python 3.12+ environment
pip install "nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main"
pip install -e .                                    # VQASynth itself
pip install depth_pro                               # CPU tier only
# Or install from source:
# pip install git+https://github.com/apple/ml-depth-pro
```

## Quick start

```bash
export OPENAI_API_KEY=sk-...
python -m experiments.nooa_agent.example_annotate \
    --image warehouse.jpg \
    --question "How far apart are the two workers in the foreground?"
```

For a CPU-only smoke test that skips depth (Florence-only questions like
"describe what's in the top-left quadrant"):

```bash
export VQASYNTH_AGENT_TIER=cpu
python -m experiments.nooa_agent.example_annotate \
    --image warehouse.jpg \
    --question "What products are visible on the middle shelf?"
```

## Tool inventory

The agent exposes these tools to the LLM (schemas auto-derived from method
signatures + docstrings — no OpenAI-JSON boilerplate):

- `detect_objects(image, phrase) -> list[Box]` — phrase-grounded detection
- `detect_all_objects(image) -> list[Box]` — full scene inventory (no phrase)
- `describe_region(image, box) -> str` — region caption
- `caption_scene(image, detail) -> str` — whole-image caption
- `dense_region_captions(image) -> list` — per-region captions covering the scene
- `read_text(image, with_regions) -> dict` — OCR ± locations
- `segment(image, referring_expression) -> list` — polygon summary for a phrase
- `pixel_relative_position(box_a, box_b) -> dict` — 2D direction + distance
- `metric_depth(image) -> DepthResult` — metric depth + intrinsics + point cloud
- `depth_at_pixel(depth, x, y) -> float` — meters at a pixel
- `distance_3d(depth, box_a, box_b) -> dict` — metric 3D Euclidean distance

The `annotate(image, question) -> SpatialAnswer` method is the LLM-driven
entry point: the model decides which tools to call and in what order to
ground its answer in real geometry.

## Testing

Structural smoke tests (no models required, work on Python 3.10):

```bash
pytest experiments/nooa_agent/tests/
```

End-to-end tests require Python 3.12 + a provider key. Not currently in CI.

## Not in scope for this branch

- Long-term memory (NOOA has `nooa-memory` — worth wiring for cross-image
  scene consistency in a future iteration)
- Sandbox for code-as-action (NOOA supports Python REPL execution; disabled
  by default here — enable with OS-level isolation only)
- Cross-tier fallback (CPU→GPU escalation when a run needs richer signal)
- Molmo point-based localization (Florence bbox output only for now)
- SAM2 pixel-accurate masks (Florence polygon summary only)
- Batch/streaming interfaces (single-image single-question only)

## Related session artifacts

- `/home/thorax/ecot_spacethinker_notes.md` — the 4-round smoke-test log
  (2026-07-15 through 2026-07-19) that validated the three operating modes
  (model-calls-tools, middleware-pre-inject, estimate-then-verify) and the
  Florence-2 confidence-gated cascade this branch is built on.
- `/home/thorax/ecot_smoke_output_aloha/florence2_tools_2026-07-18/` —
  original prototype Florence-2 tool code refactored into
  `tools/florence.py` here.
