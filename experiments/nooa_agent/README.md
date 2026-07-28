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
# Python 3.12+ environment (NOOA constraint)
pip install "numpy<2.0"                             # see note below
pip install "nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main"
pip install -e .                                    # VQASynth itself
```

**Why the numpy pin:** Florence-2 loads via `trust_remote_code=True`, and the
model repo's Python code has numpy 1.x assumptions that break on numpy 2.x
(silent tool errors like "cannot import name X" during inference). DepthPro
similarly has 1.x deps that haven't been patched at the time of writing.
Colab defaults to numpy 2.x and will need this override + a runtime restart.

**CPU tier — DepthPro** (only needed if not using the GPU tier). No PyPI
package exists; install from source + download weights:

```bash
pip install git+https://github.com/apple/ml-depth-pro
huggingface-cli download --local-dir checkpoints apple/DepthPro
```

The `depth_pro.create_model_and_transforms()` call defaults to loading
weights from `./checkpoints/depth_pro.pt`, so run subsequent commands
from the directory containing `checkpoints/`.

**GPU tier — VGGT** is installed transitively via `vqasynth[all]` — no
extra step; see the VQASynth root README for the CUDA install path.

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

## Multi-question workloads: `SceneContext`

For batch labeling — many questions per image — bind the image once and
memoize the expensive tool outputs (metric depth, Florence-2 forward passes):

```python
from experiments.nooa_agent.spatial_annotator import SpatialAnnotator, SceneContext

agent = SpatialAnnotator(llm=llm)
img = Image.open("warehouse.jpg").convert("RGB")

scene = SceneContext(agent, img)
await scene.warmup()                        # optional: eager depth + detect + caption
r1 = await scene.annotate("How far apart are the two workers?")
r2 = await scene.annotate("Which forklift is closest to the doorway?")
r3 = await scene.annotate("What products are on the middle shelf?")
```

The second and third calls reuse cached tool outputs from the first — depth
is by far the dominant per-image cost (VGGT-1B forward or DepthPro inference),
so this cuts wall-clock roughly proportional to the fraction of questions
that touch depth.

Binding a different image to the same agent auto-invalidates the cache.

## Trace capture for synthetic-VQA fine-tuning

Each `annotate()` call can be captured as a structured JSONL row consumable by
Qwen2.5-VL / Qwen3-VL fine-tuning (OpenAI-messages format with `<tool_call>`
and `<tool_response>` cycles preserved). Serializer is pluggable — add a
transform function for any other target model.

```python
from experiments.nooa_agent.trace import TraceWriter

with TraceWriter("/data/traces.jsonl") as writer:
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        scene = SceneContext(agent, img, trace_writer=writer, image_ref=img_path)
        for question in questions_for(img):
            await scene.annotate(question)          # auto-appends to writer
```

Each JSONL row contains a `messages` field (ready for
`tokenizer.apply_chat_template()`) and a `meta` block with `image_ref`,
`question`, `confidence`, `tool_calls_used`, `wall_clock_s` for downstream
filtering / quality gating.

**To target a different model family:**

```python
def anthropic_serialize(trace):
    """Emit Anthropic tool_use / tool_result blocks instead."""
    ...  # transform trace.events → Anthropic format
    return {"messages": ...}

writer = TraceWriter("/data/traces.jsonl", serializer=anthropic_serialize)
```

## Device + precision control

By default the CPU tier lands everything on `cpu` fp32 (~6-7 GB RAM peak
after cascade + depth are warm), and the GPU tier lands everything on
`cuda:0` fp32 (~8 GB VRAM peak). For multi-GPU pinning or fp16 to halve
VRAM, pass per-tool overrides:

```python
agent = SpatialAnnotator(
    tier="gpu",
    florence_device="cuda:1",     # detection on second GPU
    florence_dtype="fp16",         # ~1 GB → ~0.5 GB base, ~3 GB → ~1.5 GB large
    depth_dtype="fp16",            # halves DepthPro / VGGT VRAM likewise
)
```

`dtype` accepts `torch.float32` / `torch.float16` / `torch.bfloat16` or
their string aliases (`"fp32"` / `"fp16"` / `"bf16"`). VGGT device placement
is controlled by ``SpatialSceneConstructor`` — pin via
``CUDA_VISIBLE_DEVICES=<n>`` at process start when the multi-GPU case matters.

## End-to-end example: batch labeling with trace capture

Ties together the whole surface — auto-tier detection, per-image caching,
tool-driven answering, and JSONL-per-annotate capture in Qwen VL format:

```python
from PIL import Image
from experiments.nooa_agent.spatial_annotator import SpatialAnnotator, SceneContext
from experiments.nooa_agent.trace import TraceWriter
from nooa.unifiedllm.registry import get_llm_client

# Auto-tier (CPU→DepthPro, GPU→VGGT). Half-precision on GPU cuts VRAM ~2×.
llm = get_llm_client("gemini/gemini-2.5-pro")
agent = SpatialAnnotator(
    llm=llm,
    max_iterations=20,
    florence_dtype="fp16",   # ignored on CPU tier (no VRAM to save)
    depth_dtype="fp16",
)

QUESTIONS_PER_IMAGE = [
    "How far apart are the two workers in the foreground?",   # → detect + depth + distance_3d
    "Which forklift is closest to the doorway?",              # → detect + depth ordering
    "What items are visible on the middle shelves?",          # → dense_region_captions
    "Is the taller worker on the left or right?",             # → detect + pixel_relative_position
]

image_paths = ["/data/warehouse_01.jpg", "/data/warehouse_02.jpg"]

with TraceWriter("/data/spatial_traces.jsonl") as writer:
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        # Bind the image ONCE; each question below reuses cached Florence
        # outputs + metric depth from the first question that computed them.
        scene = SceneContext(agent, img, trace_writer=writer, image_ref=img_path)

        for q in QUESTIONS_PER_IMAGE:
            result = await scene.annotate(q)
            print(f"[{img_path}]  Q: {q}")
            print(f"  A: {result.answer}")
            print(f"  confidence: {result.confidence}, {result.tool_calls_used} tool calls")

        agent.clear_scene()   # optional — SceneContext binding auto-invalidates
```

Each `annotate()` call appends one Qwen-shaped JSONL row to
``/data/spatial_traces.jsonl``, ready to feed
``tokenizer.apply_chat_template()`` for fine-tuning Qwen2.5-VL or Qwen3-VL.

## Using as a lerobot `VlmClient`

`SpatialAnnotator` satisfies lerobot's steerable-annotation `VlmClient`
protocol via a thin adapter, so it drops into any annotation module that
consumes a `VlmClient` — the existing `Vqa`/`Plan` modules, the pending
`EcotReasoningModule` in [huggingface/lerobot#4036](https://github.com/huggingface/lerobot/pull/4036),
or any future module built on the same protocol. Every call routes through
the full CodeAct + tool-grounded loop (detect + metric depth + distance_3d
as the LLM composes them) before the JSON response is returned.

**Standalone example — no lerobot required:**

```python
from PIL import Image
from experiments.nooa_agent.spatial_annotator import SpatialAnnotator
from experiments.nooa_agent.lerobot_adapter import SpatialAnnotatorVlmClient
from nooa.unifiedllm.registry import get_llm_client

llm = get_llm_client("gemini/gemini-2.5-pro")
agent = SpatialAnnotator(llm=llm, max_iterations=12)
vlm = SpatialAnnotatorVlmClient(annotator=agent)

image = Image.open("frame.jpg").convert("RGB")
result = vlm.generate_json([[{
    "role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": (
            "Reply with strictly valid JSON: "
            '{"scene_perception": "<text>", "objects": [<name>, ...], '
            '"gripper_target_distance_cm": <number>}'
        )},
    ],
}]])
# → [{"scene_perception": "...", "objects": [...], "gripper_target_distance_cm": 8.3}]
```

The `generate_json` signature (`messages_batch → list[Any]`) is the entire
`VlmClient` protocol — the same surface every module in lerobot's steerable
pipeline calls. Whatever JSON schema the calling prompt specifies, the
adapter returns that JSON parsed. We intentionally hard-code **no**
schema-specific handling here: if PR #4036's ECoT 4-field format is
restructured in review, or if a different module wants a different schema
tomorrow, this adapter still works unchanged.

**To use with lerobot's pipeline:** configure the pipeline's `VlmClient`
factory to construct `SpatialAnnotatorVlmClient` instead of the default
Qwen backend. Every anchor's contact sheet + prompt goes through our
agent; the reasoning stays auditable via `agent.event_manager.values()`
and (if a `trace_writer` is bound to the `SceneContext`) streams to JSONL
alongside whatever the pipeline writes.

**Single-frame anchors are required for tool grounding.** Our metric-depth
tool (DepthPro / VGGT) produces geometrically meaningless output on a tiled
contact sheet — pixel coordinates don't map to a real camera geometry, so
`distance_3d` returns nonsense. Florence detection also degrades (labels
duplicate across tiles). When plugging into `EcotReasoningModule`, set
`contact_sheet_size=1` (or the equivalent config once PR #4036 finalizes)
so each anchor is a single frame. Temporal context is better recovered
through multi-turn chat history (see next section) than through tiled inputs.

## Aloha ECoT smoke test — bypass the pipeline, iterate frames directly

Fastest way to validate the agent on real robot data before doing the full
lerobot integration. See ``example_lerobot_aloha_ecot.py`` in this directory:

```bash
export GEMINI_API_KEY=...
python -m experiments.nooa_agent.example_lerobot_aloha_ecot \
    --repo-id lerobot/aloha_static_coffee \
    --episodes 0 \
    --anchor-stride 30 \
    --output /data/aloha_ecot.jsonl
```

Loads the aloha dataset, iterates at your chosen anchor cadence (every 30
frames at fps=50 → every 0.6s), runs our tool-grounded agent on each
single-frame anchor with the ECoT prompt, and writes one JSONL row per
anchor. Each row carries the 4-field ECoT JSON, per-anchor confidence,
tool-call count, wall clock, and supporting evidence. Ready to hand to a
reward-extraction script downstream.

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
