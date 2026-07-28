"""Florence-2 detection + segmentation + confidence-gated escalation.

Class-based refactor of the 2026-07-18 smoke-test scripts
(``florence2_tools.py`` + ``florence2_cascade.py``). NOOA's Agent auto-derives
tool schemas from these method signatures + docstrings, so no separate
OpenAI-schema JSON registration is needed.

Escalation policy: if Florence-2-base returns a "degenerate" detection
(single box covering >60% of the image, the observed whole-image-fallback
failure mode) escalate to Florence-2-large.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PIL import Image

# Trigger from the 2026-07-18 cascade experiment
DEGENERATE_AREA_THRESHOLD = 0.6


@dataclass
class Box:
    """Bounding box in pixel coordinates: x1, y1 top-left; x2, y2 bottom-right."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class FlorenceDetector:
    """CPU-tier detector: Florence-2-base with Florence-2-large escalation.

    Escalation triggers when the base tier's detection is degenerate (single
    box > 60% of image area — the observed whole-image-fallback pattern).
    """

    BASE_MODEL = "microsoft/Florence-2-base"
    LARGE_MODEL = "microsoft/Florence-2-large"

    def __init__(self, device: str = "cpu", enable_cascade: bool = True):
        self.device = device
        self.enable_cascade = enable_cascade
        self._base = None
        self._large = None

    # -- lazy model loaders (only pay the load cost when a method is called)

    def _load(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        ).to(self.device)
        model.eval()
        return processor, model

    def _base_backend(self):
        if self._base is None:
            self._base = self._load(self.BASE_MODEL)
        return self._base

    def _large_backend(self):
        if self._large is None:
            self._large = self._load(self.LARGE_MODEL)
        return self._large

    # -- florence runner used by every task-specific method

    def _run(self, backend, image: Image.Image, task: str, extra: str = "") -> dict:
        import torch

        processor, model = backend
        prompt = task + extra
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        return processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height)
        )

    # -- degenerate-detection heuristic drives cascade

    def _is_degenerate(self, boxes: list[Box], image: Image.Image) -> bool:
        if not boxes:
            return False  # empty is not degenerate — it's just "not found"
        img_area = image.width * image.height
        return any(b.area / img_area > DEGENERATE_AREA_THRESHOLD for b in boxes)

    # -- tool-shaped methods NOOA will auto-derive schemas from

    def detect_objects(self, image: Image.Image, phrase: str) -> list[Box]:
        """Detect objects matching a natural-language phrase in the image.

        Returns bounding boxes. Empty list if no match. On CPU tier, escalates
        from Florence-2-base to Florence-2-large if the base tier returns a
        degenerate detection (single box covering the whole image).

        Args:
            image: RGB image to search.
            phrase: Short referring expression, e.g. 'the coffee capsule',
                'the person walking'. Definite articles ('the ...') tend to
                localize better than bare phrases.
        """
        boxes = self._detect_at_tier(self._base_backend(), image, phrase)
        if self.enable_cascade and self._is_degenerate(boxes, image):
            boxes = self._detect_at_tier(self._large_backend(), image, phrase)
        return boxes

    def _detect_at_tier(self, backend, image: Image.Image, phrase: str) -> list[Box]:
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        parsed = self._run(backend, image, task, phrase)
        result = parsed.get(task, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("labels", []) or [""] * len(bboxes)
        return [
            Box(x1=round(b[0], 1), y1=round(b[1], 1),
                x2=round(b[2], 1), y2=round(b[3], 1), label=lbl)
            for b, lbl in zip(bboxes, labels)
        ]

    def caption(self, image: Image.Image, detail: str = "short") -> str:
        """Generate a natural-language caption for the whole image.

        Use for scene-level description without asking about specific objects.
        Cheap first-pass to give the annotator scene context.

        Args:
            image: RGB image.
            detail: 'short' (<CAPTION>) or 'detailed' (<DETAILED_CAPTION>).
        """
        task = "<DETAILED_CAPTION>" if detail == "detailed" else "<CAPTION>"
        parsed = self._run(self._base_backend(), image, task)
        result = parsed.get(task, "")
        return result if isinstance(result, str) else str(result)

    def detect_all_objects(self, image: Image.Image) -> list[Box]:
        """Detect every object Florence-2 knows about in the image.

        Unlike ``detect_objects`` (which requires a phrase), this returns a full
        inventory — useful for surveying the scene before targeted queries.
        """
        task = "<OD>"
        parsed = self._run(self._base_backend(), image, task)
        result = parsed.get(task, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("labels", []) or [""] * len(bboxes)
        return [
            Box(x1=round(b[0], 1), y1=round(b[1], 1),
                x2=round(b[2], 1), y2=round(b[3], 1), label=lbl)
            for b, lbl in zip(bboxes, labels)
        ]

    def dense_region_captions(self, image: Image.Image) -> list[dict[str, Any]]:
        """Return a caption + bbox for every notable region in the image.

        Covers the whole scene with per-region descriptions. Use when the
        annotator needs comprehensive scene understanding without knowing
        what to look for.

        Returns a list of ``{caption: str, box: Box}``.
        """
        task = "<DENSE_REGION_CAPTION>"
        parsed = self._run(self._base_backend(), image, task)
        result = parsed.get(task, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("labels", [])
        return [
            {
                "caption": lbl,
                "box": Box(x1=round(b[0], 1), y1=round(b[1], 1),
                           x2=round(b[2], 1), y2=round(b[3], 1), label=lbl),
            }
            for b, lbl in zip(bboxes, labels)
        ]

    def read_text(self, image: Image.Image, with_regions: bool = False) -> dict[str, Any]:
        """Extract text from the image via OCR.

        Args:
            image: RGB image.
            with_regions: if True, also returns quadrilateral bboxes per
                detected text region (<OCR_WITH_REGION>). Otherwise returns
                a single concatenated text string (<OCR>).

        Returns:
            ``{"text": str}`` when with_regions=False, or
            ``{"regions": [{"text": str, "quad_box": [x1,y1,...,x4,y4]}, ...]}``
            when with_regions=True.
        """
        if with_regions:
            task = "<OCR_WITH_REGION>"
            parsed = self._run(self._base_backend(), image, task)
            result = parsed.get(task, {})
            quads = result.get("quad_boxes", [])
            labels = result.get("labels", []) or [""] * len(quads)
            return {
                "regions": [
                    {"text": lbl, "quad_box": [round(v, 1) for v in q]}
                    for q, lbl in zip(quads, labels)
                ]
            }
        task = "<OCR>"
        parsed = self._run(self._base_backend(), image, task)
        text = parsed.get(task, "")
        return {"text": text if isinstance(text, str) else str(text)}

    def describe_region(self, image: Image.Image, box: Box) -> str:
        """Return a short caption describing a specific region of the image.

        Use to verify what a detected box actually contains, or to caption
        arbitrary regions the annotator wants to reason about.

        Args:
            image: RGB image.
            box: The region to describe (pixel coordinates).
        """
        task = "<REGION_TO_DESCRIPTION>"
        W, H = image.width, image.height
        region_str = (
            f"<loc_{int(box.x1 * 999 / W)}><loc_{int(box.y1 * 999 / H)}>"
            f"<loc_{int(box.x2 * 999 / W)}><loc_{int(box.y2 * 999 / H)}>"
        )
        parsed = self._run(self._base_backend(), image, task, region_str)
        caption = parsed.get(task, "")
        if isinstance(caption, dict):
            for v in caption.values():
                if isinstance(v, str):
                    return v
        return caption if isinstance(caption, str) else str(caption)


class FlorenceSegmenter:
    """Referring-expression segmentation via Florence-2. CPU-tier.

    Returns polygon summaries (bbox of polygon + point count). For pixel-
    accurate masks in production, use ``Sam2Segmenter`` on GPU tier instead.
    """

    def __init__(self, detector: FlorenceDetector):
        # Share the loaded model — no separate load
        self._detector = detector

    def segment(self, image: Image.Image, referring_expression: str) -> list[dict[str, Any]]:
        """Segment an object matching a short referring expression.

        Returns a list of regions, each with:
            - label: str
            - polygon_bbox: [x1, y1, x2, y2] of the polygon's bounding box
            - polygon_points: int, number of vertices

        For pixel-accurate mask output, use the GPU-tier SAM2 segmenter.

        Args:
            image: RGB image.
            referring_expression: Short referring phrase.
        """
        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        parsed = self._detector._run(
            self._detector._base_backend(), image, task, referring_expression
        )
        result = parsed.get(task, {})
        polygons = result.get("polygons", [])
        labels = result.get("labels", []) or [""] * len(polygons)
        out = []
        for poly_group, label in zip(polygons, labels):
            for region in poly_group:
                if len(region) < 6:
                    continue
                xs, ys = region[0::2], region[1::2]
                out.append({
                    "label": label,
                    "polygon_bbox": [round(min(xs), 1), round(min(ys), 1),
                                     round(max(xs), 1), round(max(ys), 1)],
                    "polygon_points": len(xs),
                })
        return out


def relative_position_2d(box_a: Box, box_b: Box) -> dict[str, Any]:
    """Pure-geometry direction + pixel distance between two boxes.

    No model call. Use after ``detect_objects`` to ground spatial-relation
    claims (left/right/above/below/aligned) in real geometry instead of
    guessing from the image.

    For metric 3D distance (meters), use ``distance_3d_meters`` on the depth
    tool tier instead.

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        Dict with:
            - b_is: str, direction of b relative to a
              (e.g., "left of and above a")
            - distance_px: pixel distance between box centers
            - dx_px, dy_px: signed pixel offsets
            - center_a, center_b: box centers as [x, y] lists
    """
    cx_a, cy_a = box_a.center
    cx_b, cy_b = box_b.center
    dx = cx_b - cx_a
    dy = cy_b - cy_a
    dist = math.hypot(dx, dy)
    horiz = "left of" if dx < -10 else "right of" if dx > 10 else "horizontally aligned with"
    vert = "above" if dy < -10 else "below" if dy > 10 else "vertically aligned with"
    return {
        "b_is": f"{horiz} and {vert} a",
        "distance_px": round(dist, 1),
        "dx_px": round(dx, 1),
        "dy_px": round(dy, 1),
        "center_a": [round(cx_a, 1), round(cy_a, 1)],
        "center_b": [round(cx_b, 1), round(cy_b, 1)],
    }


__all__ = [
    "Box",
    "FlorenceDetector",
    "FlorenceSegmenter",
    "relative_position_2d",
    "DEGENERATE_AREA_THRESHOLD",
]
