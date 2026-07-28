"""Aloha ECoT smoke: run SpatialAnnotator on aloha frames with the ECoT prompt.

Bypasses ``EcotReasoningModule`` from PR #4036 entirely — iterates a
``LeRobotDataset`` directly and emits one ECoT-style JSON blob per SINGLE
anchor frame via our tool-grounded agent. Faster path to validate the
agent produces useful ECoT reasoning on real robot data before wiring
through the full lerobot pipeline (which would need contact_sheet_size=1
config on their side, since our tools require per-frame images to give
meaningful outputs — metric depth on a tiled composite is geometrically
meaningless, Florence detection duplicates labels across tiles).

Requires: ``lerobot``, ``nooa``, and an LLM provider key (default: Gemini).
Model weights (Florence-2, DepthPro / VGGT) auto-download on first use.

Usage::

    export GEMINI_API_KEY=...
    python -m experiments.nooa_agent.example_lerobot_aloha_ecot \\
        --repo-id lerobot/aloha_static_coffee \\
        --episodes 0 1 \\
        --anchor-stride 30 \\
        --output /data/aloha_ecot.jsonl

Output: one JSONL row per anchor frame, containing the ECoT 4-field JSON
(scene_perception / object_identification / task_plan /
subtask_decomposition) plus metadata (episode_id, frame_idx, t_seconds,
tool_calls_used, wall_clock_s). Ready for downstream reward extraction
or direct import as ``language_persistent`` rows in lerobot's format.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


# Adapted from lerobot PR #4036's prompts/ecot.txt — reworded for single-frame
# rather than contact-sheet input, and instructed to use tool composition
# (detect + metric_depth + distance_3d) for grounded measurements in
# subtask_decomposition.
ECOT_PROMPT_TEMPLATE = """You are generating one structured Embodied Chain-of-Thought (ECoT) reasoning trace for a single frame of a robot manipulation episode.

Episode task: "{task}"
This frame is at t={t:.2f}s (frame {frame_idx} of {n_frames}) in episode {episode_id}.

Produce the four reasoning stages that ZR-0 shows generalize across robot embodiments:
  1. scene_perception — the workspace, the robot, and the overall scene state.
  2. object_identification — task-relevant objects and their salient properties / affordances.
  3. task_plan — the ordered high-level steps still needed to finish the task.
  4. subtask_decomposition — the immediate atomic action for THIS frame.

Ground each stage in tool outputs. For object_identification, USE detect_objects
or detect_all_objects to find real bounding boxes and describe_region to verify
labels. For subtask_decomposition, USE metric_depth + distance_3d to state
actual gripper-to-target distances in centimeters when the task involves reach
or grasp. Do not invent measurements — cite the tool call that produced each
number.

Output strictly valid JSON only, no prose, no fences, exactly this shape:
{{
  "scene_perception": "<text>",
  "object_identification": "<text>",
  "task_plan": "<text>",
  "subtask_decomposition": "<text>"
}}"""


class _JsonlWriter:
    """Simple append-only JSONL. TraceWriter-shaped but with the raw ECoT
    row rather than the Qwen chat-template rows the main pipeline writes."""

    def __init__(self, path: str):
        self.path = path
        self._f = None

    def __enter__(self):
        self._f = open(self.path, "a")
        return self

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, default=str) + "\n")
        self._f.flush()

    def __exit__(self, *_):
        if self._f is not None:
            self._f.close()


def _pil_from_tensor(tensor):
    """LeRobotDataset returns images as torch tensors (C, H, W) or (H, W, C);
    convert to PIL Image regardless of shape convention."""
    from torchvision.transforms.functional import to_pil_image
    import torch

    if isinstance(tensor, torch.Tensor):
        # to_pil_image handles both (C,H,W) and (H,W,C); may need dtype coerce
        if tensor.dtype != torch.uint8 and tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        return to_pil_image(tensor)
    # Fallback: assume it's already PIL or a numpy array
    from PIL import Image
    import numpy as np

    if isinstance(tensor, np.ndarray):
        return Image.fromarray(tensor)
    return tensor  # trust the caller


async def annotate_episode(
    agent, dataset, episode_id, anchor_stride, camera_key, writer
):
    """Iterate an episode's frames at anchor_stride cadence, emit ECoT rows."""
    from experiments.nooa_agent.lerobot_adapter import _strip_to_json

    # LeRobotDataset exposes episode boundaries via episode_data_index (dict
    # of tensors); coerce to int for indexing.
    ep_from = int(dataset.episode_data_index["from"][episode_id])
    ep_to = int(dataset.episode_data_index["to"][episode_id])

    # Task string lives on the sample or in the metadata; try both.
    task = ""
    try:
        task = dataset.meta.tasks[episode_id]
    except (AttributeError, KeyError, IndexError, TypeError):
        pass

    for anchor_frame_idx in range(ep_from, ep_to, anchor_stride):
        sample = dataset[anchor_frame_idx]
        image = _pil_from_tensor(sample[camera_key])
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        t_seconds = (anchor_frame_idx - ep_from) / dataset.fps
        # Sample-level task string (newer lerobot puts it on the sample dict)
        sample_task = sample.get("task") if isinstance(sample, dict) else None
        effective_task = task or sample_task or "unknown task"

        prompt = ECOT_PROMPT_TEMPLATE.format(
            task=effective_task,
            t=t_seconds,
            frame_idx=anchor_frame_idx - ep_from,
            n_frames=ep_to - ep_from,
            episode_id=episode_id,
        )

        t0 = time.time()
        try:
            result = await agent.annotate(image, prompt)
        except Exception as e:
            print(f"  ep {episode_id} frame {anchor_frame_idx}: annotate raised — {e}")
            continue
        elapsed = time.time() - t0

        try:
            ecot_json = _strip_to_json(result.answer)
        except ValueError as e:
            print(f"  ep {episode_id} frame {anchor_frame_idx}: JSON parse failed — {e}")
            ecot_json = {"_parse_error": str(e), "_raw_answer": result.answer[:500]}

        writer.write({
            "episode_id": int(episode_id),
            "anchor_frame_idx": int(anchor_frame_idx),
            "t_seconds": round(t_seconds, 3),
            "task": effective_task,
            "camera_key": camera_key,
            "ecot": ecot_json,
            "confidence": result.confidence,
            "tool_calls_used": result.tool_calls_used,
            "supporting_evidence": list(result.supporting_evidence),
            "wall_clock_s": round(elapsed, 2),
        })
        print(f"  ep {episode_id} frame {anchor_frame_idx:5d} (t={t_seconds:5.2f}s): "
              f"{result.tool_calls_used} calls, {elapsed:5.1f}s, conf={result.confidence}")


async def _amain(args):
    from experiments.nooa_agent.spatial_annotator import SpatialAnnotator
    from nooa.unifiedllm.registry import get_llm_client
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset {args.repo_id!r}...")
    dataset = LeRobotDataset(args.repo_id)
    n_episodes = len(dataset.episode_data_index["from"])
    print(f"  {len(dataset)} frames, {n_episodes} episodes, fps={dataset.fps}")

    # Auto-pick the first camera if none specified — aloha typically has
    # observation.images.top and observation.images.wrist.
    cameras = [k for k in dataset.features if k.startswith("observation.images.")]
    if not cameras:
        raise SystemExit(f"No observation.images.* features in {args.repo_id}")
    camera_key = args.camera or cameras[0]
    print(f"  camera: {camera_key}")

    print(f"Building agent (LLM={args.model})...")
    llm = get_llm_client(args.model)
    agent = SpatialAnnotator(llm=llm, max_iterations=args.max_iterations)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path}")
    with _JsonlWriter(str(out_path)) as writer:
        for episode_id in args.episodes:
            if episode_id >= n_episodes:
                print(f"Skipping episode {episode_id}: dataset has only {n_episodes} episodes")
                continue
            print(f"Episode {episode_id}...")
            await annotate_episode(
                agent, dataset, episode_id,
                anchor_stride=args.anchor_stride,
                camera_key=camera_key,
                writer=writer,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", default="lerobot/aloha_static_coffee")
    parser.add_argument(
        "--episodes", nargs="+", type=int, default=[0],
        help="Episode indices to annotate.",
    )
    parser.add_argument(
        "--anchor-stride", type=int, default=30,
        help="Frames between anchors. At fps=50, 30 → every 0.6s. Lower "
             "= denser anchors, more wall time. Match to your reward-shaping "
             "cadence downstream.",
    )
    parser.add_argument(
        "--camera", default=None,
        help="Camera feature key (e.g. observation.images.top). "
             "Auto-picks first observation.images.* if unset.",
    )
    parser.add_argument(
        "--model", default="gemini/gemini-2.5-pro",
        help="LiteLLM-compatible model id. Pro/Sonnet recommended over Flash "
             "for tool-composition quality.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=15,
        help="CodeAct iteration cap per annotate call.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output JSONL (append mode).",
    )
    args = parser.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
