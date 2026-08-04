"""Small runnable demo of the multi-view correspondence stage.

Loads one image, synthesizes a second view with a KNOWN perspective warp
(deterministic — no Ego4D download needed), then runs
``vqasynth.correspondence.CorrespondenceExtractor`` to recover point
correspondences and render a sample Molmo pointing-VLM QA row.

Run:
    python examples/correspondence_example.py
    python examples/correspondence_example.py --image path/to/view_a.jpg --out viz.png

CPU-only. Requires opencv-python + Pillow (both already in VQASynth's
requirements).
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image

from vqasynth.correspondence import (
    CorrespondenceExtractor,
    build_qa_message,
    correspondences_to_messages,
)


ASSET = os.path.join(
    os.path.dirname(__file__), "assets", "warehouse_rgb.jpg"
)


def make_synthetic_view(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Warp ``image`` by a known homography to stand in for a second view.

    Returning the ground-truth H lets the caller sanity-check that the
    extractor recovered a consistent geometry. Real usage feeds two real
    adjacent frames (e.g. from an Ego4D clip) instead.
    """
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    # A mild perspective + translation — typical of adjacent ego-camera frames.
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.05, h * 0.03], [w * 0.93, h * 0.06], [w * 0.97, h * 0.95], [w * 0.02, h * 0.92]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, H, (w, h))
    return Image.fromarray(warped), H


def draw_matches(view_a: Image.Image, view_b: Image.Image, result) -> np.ndarray:
    """Side-by-side canvas with correspondence lines (view A -> view B)."""
    a = np.array(view_a.convert("RGB"))
    b = np.array(view_b.convert("RGB"))
    h = max(a.shape[0], b.shape[0])
    canvas = np.full((h, a.shape[1] + b.shape[1], 3), 255, dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1]:] = b
    offset = a.shape[1]
    for m in result.matches:
        x1, y1 = int(m.pt_a[0]), int(m.pt_a[1])
        x2, y2 = int(m.pt_b[0]) + offset, int(m.pt_b[1])
        cv2.circle(canvas, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(canvas, (x2, y2), 4, (0, 128, 0), -1)
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 165, 0), 1)
    return canvas


def main(image_path: str, out_path: str | None) -> None:
    view_a = Image.open(image_path).convert("RGB")
    view_b, H_truth = make_synthetic_view(view_a)

    extractor = CorrespondenceExtractor()
    result = extractor.extract(view_a, view_b)
    print(result)  # compact repr
    print(f"ground-truth homography recovered? {result.homography is not None}")

    if not result.matches:
        print("No correspondences found — try a less textured image or relax --ratio.")
        return

    # One standalone QA row (multi-view: image index 0 = A, index 1 = B).
    sample = build_qa_message(
        result.matches[0], result.view_a_size, result.view_b_size
    )
    print("\nSample pointing-VLM QA row:")
    for msg in sample["messages"]:
        print(f"  [{msg['role']}]")
        for block in msg["content"]:
            if block["type"] == "text":
                print(f"    text: {block['text']}")
            else:
                print(f"    image index={block['index']}")

    # The full multi-turn conversation shape that lands in the dataset.
    conversation = correspondences_to_messages(result, max_turns=4)
    print(f"\nAssembled {len(conversation)} messages from "
          f"{min(4, len(result.matches))} correspondences.")

    if out_path:
        canvas = draw_matches(view_a, view_b, result)
        Image.fromarray(canvas).save(out_path)
        print(f"\nVisualization saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=ASSET, help="Path to view A image")
    parser.add_argument("--out", default=None, help="Optional path to save a match visualization")
    args = parser.parse_args()
    main(args.image, args.out)
