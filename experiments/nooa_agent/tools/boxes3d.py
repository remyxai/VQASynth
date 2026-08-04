"""3D bounding-box tool wrapper for the SpatialAnnotator agent.

Wraps :class:`vqasynth.detection_3d.Detection3DGenerator` (PR #129) as a NOOA
agent tool, so the SpatialAnnotator can call "give me the 3D box of each object"
as a discrete step in a dynamically-composed pipeline — instead of only through
the batch Docker stage (``docker/detection_3d_stage/``).

The two inputs this needs are already produced by other NOOA tools:
``depth.DepthProEstimator`` / ``depth.VggtEstimator`` yield the metric point
cloud (``DepthResult.point_cloud_xyz``), and ``florence.FlorenceSegmenter`` /
SAM2 yield per-object masks. This tool composes those two into an explicit 3D
bounding-box primitive — ``(center, extent)`` per object — which the agent then
pairs with ``depth.distance_3d_meters`` and ``florence.relative_position_2d``
for spatial reasoning that needs the box *shape*, not just a center point.

Same tool ABI as :mod:`experiments.nooa_agent.tools.orientation` (PR #134):
compact-``__repr__`` result dataclass + backend class with lazy imports +
module-level singleton + one standalone tool function.

Scope discipline (issue #47 / the detection_3d brief):
  - Does NOT reimplement point-cloud -> box extraction. Box math is delegated
    to ``Detection3DGenerator.compute_boxes`` (which wraps open3d's
    ``get_axis_aligned_bounding_box`` as :func:`vqasynth.detection_3d.compute_aabb`).
  - Does NOT re-run depth or segmentation — both already have NOOA tools
    (``depth.py``, ``florence.py``); this tool takes their outputs as inputs.
  - Does NOT attach the raw point cloud to the returned :class:`Box3D`. The box
    is the summary; the point cloud is a large internal artifact.

Heavy imports stay inside methods, mirroring :mod:`depth` / :mod:`florence` /
:mod:`orientation` so importing this module never drags in ``open3d`` (a
runtime-only dep of the underlying stage's ``.pcd`` I/O path) or triggers a
weight download. ``vqasynth.detection_3d`` is imported lazily for uniformity
with the other tool backends, even though it is pure-Python at import time.
``numpy`` is reached only on the real point-cloud slicing path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# DepthResult is the depth-tool contract this tool consumes; ``_unproject`` is
# reused (not reimplemented) for the no-point-cloud fallback, the same way
# ``distance_3d_meters`` falls back to it.
from experiments.nooa_agent.tools.depth import DepthResult, _unproject

logger = logging.getLogger(__name__)

# The box math is the pure-Python equivalent of open3d's
# ``get_axis_aligned_bounding_box`` (see :func:`vqasynth.detection_3d.compute_aabb`).
DEFAULT_BACKEND = "open3d_aabb"


@dataclass
class Box3D:
    """Axis-aligned 3D bounding box for one detected object (agent-facing summary).

    Fields:
        center: ``(cx, cy, cz)`` box center in meters, camera frame.
        extent: ``(dx, dy, dz)`` full per-axis side lengths in meters.
        label: class label carried from the segmentation/captioning stage.
        mask_id: index into the caller's mask list, so the agent can correlate
            a box back to the mask that produced it (``None`` if untracked).
        confidence: ``0..1`` heuristic. The underlying
            :class:`~vqasynth.detection_3d.Detection3DGenerator` does not emit a
            confidence today, so this defaults to ``1.0``; a future density-based
            heuristic could populate it without changing the tool surface.
        backend: e.g. ``"open3d_aabb"`` — leaves room for an oriented-box backend.
    """

    center: tuple[float, float, float]
    extent: tuple[float, float, float]
    label: str
    mask_id: int | None = None
    confidence: float = 1.0
    backend: str = DEFAULT_BACKEND

    def __repr__(self) -> str:
        # Compact one-liner — mirrors DepthResult/OrientationResult's guard
        # against dumping large artifacts in a NOOA trace event (one fires per
        # tool call). The raw floats already live on the named fields and the
        # underlying stage may attach a point cloud internally; this repr
        # deliberately surfaces only the summary, formatted to 2 decimals.
        cx, cy, cz = self.center
        dx, dy, dz = self.extent
        return (
            f"Box3D(label={self.label!r}, "
            f"center=({cx:.2f},{cy:.2f},{cz:.2f}), "
            f"extent=({dx:.2f},{dy:.2f},{dz:.2f}))"
        )


class Detection3DGeneratorAgent:
    """Backend composing :class:`vqasynth.detection_3d.Detection3DGenerator`.

    Does NOT reimplement point-cloud -> box extraction — it delegates the box
    math to the underlying stage's :meth:`compute_boxes`. This class adds the
    agent-facing concerns the batch stage doesn't have: per-mask point-cloud
    slicing from a :class:`DepthResult`, empty-mask dropping with ``mask_id``
    tracking, and lifting the stage's :class:`BoundingBox3D` into the compact
    :class:`Box3D` summary.

    ``backend`` labels the box-extraction strategy (default ``"open3d_aabb"``)
    so an oriented-box backend can be slotted in later without changing the
    tool surface.
    """

    BACKEND = DEFAULT_BACKEND

    def __init__(self, backend: str | None = None):
        self.backend = backend or self.BACKEND
        # Constructed lazily so importing this module never imports
        # vqasynth.detection_3d (uniform with the other tool backends).
        self._generator = None

    def _ensure_loaded(self):
        if self._generator is not None:
            return
        from vqasynth.detection_3d import Detection3DGenerator

        self._generator = Detection3DGenerator()

    def _resolve_point_cloud(self, depth: DepthResult) -> np.ndarray:
        """The (H, W, 3) metric point cloud, falling back to unprojection.

        Prefers ``depth.point_cloud_xyz`` (always populated by the real
        DepthPro / VGGT / FoundationGeo estimators). If a caller hands a
        :class:`DepthResult` without one, unproject ``depth_m`` via the
        intrinsics — reusing ``depth._unproject`` rather than reimplementing
        pinhole unprojection (same fallback ``distance_3d_meters`` uses).
        """
        xyz = depth.point_cloud_xyz
        if xyz is not None:
            return np.asarray(xyz, dtype=np.float64)
        return _unproject(
            np.asarray(depth.depth_m, dtype=np.float64),
            np.asarray(depth.intrinsics_3x3, dtype=np.float64),
        )

    def detect(
        self,
        masks: Sequence,
        depth: DepthResult,
        labels: Sequence[str],
    ) -> tuple[list[Box3D], int]:
        """One :class:`Box3D` per mask with enough valid depth points.

        Args:
            masks: per-object ``HxW`` bool/uint8 arrays.
            depth: :class:`DepthResult` whose point cloud is sliced per mask.
            labels: class labels aligned by index with ``masks`` (may be empty,
                in which case each box is labeled ``"object_{i}"``).

        Returns:
            ``(boxes, dropped)`` — ``boxes`` is the lifted :class:`Box3D` list
            (``mask_id`` aligned to the caller's mask list), ``dropped`` is the
            count of masks that resolved to no valid point cloud.
        """
        self._ensure_loaded()
        pcd = self._resolve_point_cloud(depth)
        H, W = pcd.shape[:2]

        # Slice the point cloud per mask; drop masks with no valid points rather
        # than emitting a zero-extent box. mask_id tracks the ORIGINAL mask index
        # so the caller can correlate a surviving box back to its mask even when
        # earlier masks were dropped.
        kept: list[tuple[int, str, np.ndarray]] = []
        dropped = 0
        for i, mask in enumerate(masks):
            m = np.asarray(mask).astype(bool)
            if m.shape[:2] != (H, W):
                # Defensive: the tool already checks mask-vs-image shape, but
                # the actual indexing target is the depth point-cloud grid.
                raise ValueError(
                    f"mask {i} shape {m.shape[:2]} != point-cloud grid ({H}, {W})"
                )
            pts = pcd[m]  # (N, 3)
            # Drop non-finite points (NaN/inf from invalid depth pixels) so they
            # don't poison the AABB min/max.
            if pts.size:
                pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) == 0:
                dropped += 1
                continue
            label = labels[i] if i < len(labels) else f"object_{i}"
            kept.append((i, label, pts))

        if not kept:
            return [], dropped

        # Delegate the box math to the underlying stage — do NOT reimplement it.
        # compute_boxes aligns captions/clouds defensively and skips empties,
        # but we've already filtered empties, so its output is 1:1 with `kept`.
        captions = [label for _, label, _ in kept]
        clouds = [pts for _, _, pts in kept]
        stage_boxes = self._generator.compute_boxes(captions, clouds)

        boxes = [
            self._lift(box, mask_id=mask_id, label=label)
            for (mask_id, label, _pts), box in zip(kept, stage_boxes)
        ]
        return boxes, dropped

    def _lift(self, box, *, mask_id: int, label: str) -> Box3D:
        """Lift a :class:`vqasynth.detection_3d.BoundingBox3D` into a :class:`Box3D`.

        Surfaces the axis-aligned center/extent only. The oriented refinement
        the stage may attach (``box.oriented``) is an internal detail not
        carried on the compact agent-facing summary — the box IS the summary;
        the raw point cloud stays behind in the stage's internal artifact.
        """
        return Box3D(
            center=tuple(float(v) for v in box.center),
            extent=tuple(float(v) for v in box.extent),
            label=label or box.label,
            mask_id=mask_id,
            confidence=1.0,  # underlying stage emits no confidence today
            backend=self.backend,
        )


# Module-level singleton so N tool calls in one agent session share one
# Detection3DGenerator — mirrors the singleton pattern the other tool backends
# (depth / florence / orientation) use.
_DEFAULT_ESTIMATOR: Detection3DGeneratorAgent | None = None


def _get_default_estimator() -> Detection3DGeneratorAgent:
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None:
        _DEFAULT_ESTIMATOR = Detection3DGeneratorAgent()
    return _DEFAULT_ESTIMATOR


def detect_3d_boxes(
    image,
    masks,
    depth: DepthResult,
    *,
    labels: Sequence[str] | None = None,
) -> list[Box3D]:
    """Agent tool: one axis-aligned 3D :class:`Box3D` per object.

    Composes the two upstream NOOA tools' outputs into an explicit box
    primitive the agent can pair with :func:`depth.distance_3d_meters` and
    :func:`florence.relative_position_2d` for shape-aware spatial reasoning.

    Args:
        image: RGB ``PIL.Image`` the masks + depth were computed from (used only
            to validate the mask pixel grid).
        masks: list of ``HxW`` bool/uint8 arrays (matches SAM /
            :class:`florence.FlorenceSegmenter` mask output shape).
        depth: a :class:`~experiments.nooa_agent.tools.depth.DepthResult` from
            ``DepthProEstimator`` / ``VggtEstimator`` — needs
            ``depth.point_cloud_xyz`` (falls back to unprojection).
        labels: optional list of class labels aligned by index with ``masks``
            (from :class:`florence.FlorenceDetector` output). Defaults to
            ``"object_{i}"`` when omitted.

    Returns:
        One :class:`Box3D` per mask that had enough non-masked depth points.
        Masks resolving to no valid point cloud are dropped (not emitted as
        zero-extent boxes); the drop count is logged at INFO for trace
        visibility.

    Raises:
        ValueError: if ``image`` isn't a PIL image, a mask's shape != the image
            shape, or ``labels`` length != ``masks`` length.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected a PIL image but got {type(image)}")

    masks = list(masks)
    # Validate the mask pixel grid against the image BEFORE doing any work —
    # the tool boundary is where a mis-sized mask is the caller's bug, not an
    # indexing accident inside the agent.
    expected = (image.height, image.width)
    for i, mask in enumerate(masks):
        shape = np.asarray(mask).shape
        if len(shape) != 2 or (shape[0], shape[1]) != expected:
            raise ValueError(
                f"mask {i} shape {(shape[0], shape[1])} != image shape {expected}"
            )

    if labels is not None:
        labels = list(labels)
        if len(labels) != len(masks):
            raise ValueError(
                f"labels length {len(labels)} != masks length {len(masks)}"
            )
    else:
        labels = []

    boxes, dropped = _get_default_estimator().detect(masks, depth, labels)
    if dropped:
        logger.info(
            "detect_3d_boxes: dropped %d of %d masks (no valid depth points)",
            dropped,
            len(masks),
        )
    return boxes


__all__ = [
    "Box3D",
    "Detection3DGeneratorAgent",
    "detect_3d_boxes",
    "DEFAULT_BACKEND",
]
