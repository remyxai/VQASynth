"""3D object-detection QA-pair synthesis — standalone example.

Demonstrates the vqasynth.detection_3d stage on synthetic per-object point
clouds, so it runs with no CUDA, no VGGT depth model, no SAM2, no open3d, and no
numpy. It shows the same surface the real pipeline calls:

    # upstream (GPU host):
    from vqasynth.localize import Localizer
    from vqasynth.scene_fusion import SpatialSceneConstructor
    from vqasynth.detection_3d import Detection3DGenerator

    masks, _, captions = Localizer(captioner_type="florence").run(image)
    pcd_filepaths, canonicalized, _, _ = SpatialSceneConstructor().run(
        "warehouse_0", image, masks, output_dir="./scenes"
    )

    # this stage — consumes the per-object point clouds + captions:
    qa_pairs = Detection3DGenerator().run(captions, pcd_filepaths)

Here we skip the GPU upstream and feed synthetic point clouds directly, so the
QA-pair generation logic is observable end-to-end on any machine.

Refs: issue https://github.com/remyxai/VQASynth/issues/47
"""
from __future__ import annotations

import math
import os
import random
import sys

# Allow running from a source checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vqasynth.detection_3d import (
    Detection3DGenerator,
    compute_bounding_box,
    format_box_coordinates,
    format_point3d,
)


def synthetic_object_points():
    """Two synthetic objects standing in for scene_fusion's per-object .pcd output.

    - "wooden crate": an axis-aligned box (no oriented refinement needed).
    - "steel pipe":   a long, thin, 45-deg-rotated cylinder-ish cloud whose
                      axis-aligned box is wasteful -> oriented box attaches.
    """
    crate = [
        (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.8),
        (0.6, 0.5, 0.8), (0.6, 0.0, 0.8), (0.0, 0.5, 0.8), (0.6, 0.5, 0.0),
    ]
    # 45-degree rotated pipe, length ~2.0m, thin cross-section.
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    pipe = []
    for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for r in (0.03, -0.03):
            pipe.append((c * t - s * r, s * t + c * r, r))
    return crate, pipe


def main():
    random.seed(0)
    crate, pipe = synthetic_object_points()
    captions = ["wooden crate", "steel pipe"]
    # Single-example wrap, the way scene_fusion stores pointclouds in the dataset.
    pointclouds = [[crate, pipe]]

    generator = Detection3DGenerator(questions_per_object=2, fmt="point3d")
    qa_pairs = generator.run(captions, pointclouds)

    print("=" * 72)
    print("Per-object 3D bounding boxes")
    print("=" * 72)
    for label, pts in zip(captions, [crate, pipe]):
        box = compute_bounding_box(pts, label=label)
        print(f"\n{label}")
        print(f"  axis-aligned : {format_point3d(box)}")
        print(f"  bracket form : {format_box_coordinates(box)}")
        if box.oriented is not None:
            print("  oriented box : attached (axis-aligned form was a poor fit)")
            print(f"    obb extent : {tuple(round(e, 3) for e in box.oriented.extent)}")
        else:
            print("  oriented box : none (axis-aligned form was tight enough)")

    print("\n" + "=" * 72)
    print("Synthesized QA pairs (Molmo <point3d> pointing training data)")
    print("=" * 72)
    for pair in qa_pairs:
        question, answer = pair.split(" Answer: ", 1)
        print(f"\nQ: {question}")
        print(f"A: {answer}")

    # SpatialRGPT-style bracketed region descriptors are also available:
    print("\n" + "=" * 72)
    print("Bracketed region-descriptor format (SpatialRGPT convention)")
    print("=" * 72)
    bracket_pairs = Detection3DGenerator(
        questions_per_object=1, fmt="bracket"
    ).run(captions, pointclouds)
    for pair in bracket_pairs:
        print(f"  {pair}")

    # And the HuggingFace-compatible message schema (same shape prompt_stage emits):
    messages = generator.create_messages_from_prompts(qa_pairs)
    print("\n" + "=" * 72)
    print(f"Message schema: {len(messages)} messages, first user message carries")
    print("the image just like the rest of the VQASynth dataset.")
    print("=" * 72)
    print(f"  first content entry: {messages[0]['content'][0]}")


if __name__ == "__main__":
    main()
