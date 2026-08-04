"""Structural smoke tests for the NOOA 3D bounding-box tool wrapper.

Verifies the :class:`Box3D` dataclass shape + compact ``__repr__``, the
mask/depth -> per-object box orchestration, empty-mask dropping, and the
tool-boundary validation guards — all against synthetic numpy inputs that feed
the *real* ``vqasynth.detection_3d`` stage (or a stub returning known boxes).
No CUDA, no SAM, no open3d. Real end-to-end (VGGT depth -> SAM2 masks) belongs
on a GPU host.

Imports from the pre-existing package modules this tool composes —
``vqasynth.detection_3d`` (the stage whose ``compute_boxes`` the agent
delegates to) and ``experiments.nooa_agent.tools.depth`` (the
:class:`DepthResult` contract) — so this is a cross-module integration check,
not a self-test of ``boxes3d`` alone. Mirrors the philosophy of
``tests/test_orientation_tool.py``.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from PIL import Image

from experiments.nooa_agent.tools.boxes3d import (
    Box3D,
    Detection3DGeneratorAgent,
    detect_3d_boxes,
)
# Pre-existing package modules — exercised here so the suite isn't a self-test
# of only the new tool: the agent must lift real BoundingBox3D objects produced
# by vqasynth.detection_3d, and DepthResult is the depth-tool contract the tool
# consumes.
from vqasynth.detection_3d import BoundingBox3D, compute_aabb
from experiments.nooa_agent.tools.depth import DepthResult


def _depth_result(point_cloud_xyz: np.ndarray) -> DepthResult:
    """Build a minimal DepthResult carrying the given (H, W, 3) point cloud.

    depth_m / intrinsics are unused on the point-cloud path (the tool reads
    ``point_cloud_xyz`` directly); they're populated only so the dataclass
    shape is honest.
    """
    H, W = point_cloud_xyz.shape[:2]
    depth_m = np.zeros((H, W), dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = 100.0
    return DepthResult(
        depth_m=depth_m,
        focal_px=100.0,
        intrinsics_3x3=K,
        point_cloud_xyz=point_cloud_xyz.astype(np.float32),
        backend="test",
    )


# ---------------------------------------------------------------------------
# Box3D.__repr__ — compact, bounded, no raw point cloud
# ---------------------------------------------------------------------------
def test_box3d_repr_matches_brief_format():
    """The repr is the one-line summary from the brief:
    ``Box3D(label='crate', center=(0.30,0.25,0.40), extent=(0.60,0.50,0.80))``."""
    box = Box3D(
        center=(0.30, 0.25, 0.40),
        extent=(0.60, 0.50, 0.80),
        label="crate",
        mask_id=2,
        confidence=1.0,
        backend="open3d_aabb",
    )
    assert repr(box) == (
        "Box3D(label='crate', center=(0.30,0.25,0.40), extent=(0.60,0.50,0.80))"
    )


def test_box3d_repr_is_compact_and_one_line():
    # A NOOA trace event fires per tool call, so the repr must stay one line and
    # bounded — no full float dump, no raw point cloud (the underlying stage may
    # attach one internally; Box3D deliberately carries none).
    box = Box3D(
        center=(0.123456789, -1.987654321, 100.5),
        extent=(0.0001, 2.0, 3.3333333),
        label="x" * 20,
    )
    text = repr(box)
    assert "\n" not in text
    assert len(text) < 160
    # 2-decimal formatting — the raw floats must not leak through.
    assert "0.123456789" not in text
    assert "3.3333333" not in text


def test_box3d_carries_no_point_cloud_field():
    # The box is the summary; the point cloud is a large internal artifact the
    # tool must NOT surface on the agent-facing result.
    box = Box3D(center=(0, 0, 0), extent=(1, 1, 1), label="x")
    assert not hasattr(box, "points")
    assert not hasattr(box, "point_cloud")
    assert not hasattr(box, "point_cloud_xyz")
    assert "array" not in repr(box)


# ---------------------------------------------------------------------------
# _lift — composes the pre-existing vqasynth.detection_3d BoundingBox3D
# ---------------------------------------------------------------------------
def test_agent_lifts_real_stage_bounding_box():
    """Drive the lift directly with a real vqasynth.detection_3d BoundingBox3D
    (built via the pre-existing compute_aabb) — the agent must surface
    center/extent/label and must NOT carry the raw point cloud."""
    src = compute_aabb([(0, 0, 0), (2, 3, 4)], label="crate")
    assert isinstance(src, BoundingBox3D)  # real pre-existing type, not a stub

    agent = Detection3DGeneratorAgent()
    box = agent._lift(src, mask_id=1, label="crate")

    assert isinstance(box, Box3D)
    assert box.center == pytest.approx((1.0, 1.5, 2.0))
    assert box.extent == pytest.approx((2.0, 3.0, 4.0))
    assert box.label == "crate"
    assert box.mask_id == 1
    assert box.confidence == 1.0  # stage emits no confidence -> default
    assert box.backend == "open3d_aabb"


# ---------------------------------------------------------------------------
# detect_3d_boxes — synthetic (image, mask, depth) inputs feed the real stage
# ---------------------------------------------------------------------------
def test_detect_3d_boxes_lifts_real_stage_output():
    """End-to-end through the real agent + the real vqasynth.detection_3d stage,
    fed synthetic points. Two non-overlapping objects -> two boxes with the
    expected axis-aligned center/extent, labels, and mask_id."""
    H, W = 4, 4
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    # Object A: top-left 2x2 -> an axis-aligned 2x2x2 box, center (1,1,1).
    pcd[0, 0] = (0, 0, 0)
    pcd[0, 1] = (2, 0, 0)
    pcd[1, 0] = (0, 2, 0)
    pcd[1, 1] = (2, 2, 2)
    # Object B: bottom-right 2x2 -> a 1x1x1 box, center (3.5,3.5,3.5).
    pcd[2, 2] = (3, 3, 3)
    pcd[2, 3] = (4, 3, 3)
    pcd[3, 2] = (3, 4, 3)
    pcd[3, 3] = (4, 4, 4)

    mask_a = np.zeros((H, W), dtype=bool)
    mask_a[0:2, 0:2] = True
    mask_b = np.zeros((H, W), dtype=bool)
    mask_b[2:4, 2:4] = True

    image = Image.new("RGB", (W, H))
    boxes = detect_3d_boxes(
        image, [mask_a, mask_b], _depth_result(pcd), labels=["crate", "ball"]
    )

    assert len(boxes) == 2
    by_id = {b.mask_id: b for b in boxes}
    a = by_id[0]
    assert a.label == "crate"
    assert a.center == pytest.approx((1.0, 1.0, 1.0))
    assert a.extent == pytest.approx((2.0, 2.0, 2.0))
    assert a.backend == "open3d_aabb"
    b = by_id[1]
    assert b.label == "ball"
    assert b.center == pytest.approx((3.5, 3.5, 3.5))
    assert b.extent == pytest.approx((1.0, 1.0, 1.0))


def test_detect_3d_boxes_matches_pre_existing_compute_aabb():
    """The lifted Box3D must equal a box computed directly via the pre-existing
    vqasynth.detection_3d.compute_aabb on the same points — the tool delegates
    box extraction to the stage, it doesn't reinvent it."""
    H, W = 2, 2
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    pcd[0, 0] = (0, 0, 0)
    pcd[0, 1] = (2, 0, 0)
    pcd[1, 0] = (0, 3, 0)
    pcd[1, 1] = (2, 3, 4)

    image = Image.new("RGB", (W, H))
    mask = np.ones((H, W), dtype=bool)
    boxes = detect_3d_boxes(image, [mask], _depth_result(pcd), labels=["crate"])

    expected = compute_aabb([(0, 0, 0), (2, 0, 0), (0, 3, 0), (2, 3, 4)], label="crate")
    assert len(boxes) == 1
    assert boxes[0].center == pytest.approx(expected.center)
    assert boxes[0].extent == pytest.approx(expected.extent)
    assert boxes[0].label == "crate"


def test_detect_3d_boxes_default_labels_when_omitted():
    H, W = 2, 2
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    pcd[0, 0] = (0, 0, 0)
    pcd[0, 1] = (1, 1, 1)
    pcd[1, 0] = (2, 2, 2)
    pcd[1, 1] = (3, 3, 3)
    image = Image.new("RGB", (W, H))
    boxes = detect_3d_boxes(image, [np.ones((H, W), dtype=bool)], _depth_result(pcd))
    assert boxes[0].label == "object_0"


# ---------------------------------------------------------------------------
# Empty-mask handling — dropped, not emitted as a zero-extent box
# ---------------------------------------------------------------------------
def test_detect_3d_boxes_drops_empty_mask_and_logs(caplog):
    H, W = 2, 2
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    pcd[0, 0] = (0, 0, 0)
    pcd[0, 1] = (1, 1, 1)
    pcd[1, 0] = (2, 2, 2)
    pcd[1, 1] = (3, 3, 3)
    full = np.ones((H, W), dtype=bool)
    empty = np.zeros((H, W), dtype=bool)  # zero non-zero pixels -> dropped

    image = Image.new("RGB", (W, H))
    with caplog.at_level(logging.INFO, logger="experiments.nooa_agent.tools.boxes3d"):
        boxes = detect_3d_boxes(
            image, [full, empty], _depth_result(pcd), labels=["a", "b"]
        )

    # Empty mask dropped — no zero-extent box emitted.
    assert len(boxes) == 1
    assert boxes[0].mask_id == 0  # ORIGINAL index preserved across the drop
    assert boxes[0].label == "a"
    # Drop count surfaced in the info log for trace visibility.
    assert any("dropped 1 of 2" in r.getMessage() for r in caplog.records)


def test_detect_3d_boxes_drops_mask_with_only_invalid_points():
    # A mask whose pixels are all non-finite (invalid depth) resolves to no
    # valid point cloud -> dropped too.
    H, W = 2, 2
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    pcd[0, 0] = (0, 0, 0)
    pcd[0, 1] = (1, 1, 1)
    pcd[1, 0] = (np.nan, np.nan, np.nan)  # invalid
    pcd[1, 1] = (np.inf, 0, 0)            # invalid
    good = np.zeros((H, W), dtype=bool)
    good[0, :] = True
    bad = np.zeros((H, W), dtype=bool)
    bad[1, :] = True

    image = Image.new("RGB", (W, H))
    boxes = detect_3d_boxes(image, [good, bad], _depth_result(pcd))
    assert len(boxes) == 1
    assert boxes[0].mask_id == 0


# ---------------------------------------------------------------------------
# Tool-boundary validation guards
# ---------------------------------------------------------------------------
def test_detect_3d_boxes_rejects_non_pil_image():
    H, W = 2, 2
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="PIL"):
        detect_3d_boxes(
            "not-an-image", [np.zeros((H, W), dtype=bool)], _depth_result(pcd)
        )  # type: ignore[arg-type]


def test_detect_3d_boxes_rejects_mask_wrong_shape():
    H, W = 4, 4
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    image = Image.new("RGB", (W, H))  # 4x4
    bad_mask = np.zeros((3, 3), dtype=bool)  # != image shape
    with pytest.raises(ValueError, match="mask 0 shape"):
        detect_3d_boxes(image, [bad_mask], _depth_result(pcd))


def test_detect_3d_boxes_rejects_labels_length_mismatch():
    H, W = 2, 2
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    image = Image.new("RGB", (W, H))
    masks = [np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=bool)]
    with pytest.raises(ValueError, match="labels length"):
        detect_3d_boxes(image, masks, _depth_result(pcd), labels=["only-one"])


# ---------------------------------------------------------------------------
# Stub delegation — the tool defers to the module-level singleton
# ---------------------------------------------------------------------------
def test_detect_3d_boxes_delegates_to_default_estimator(monkeypatch):
    """Patch the singleton to a stub returning known box params, so the
    tool->agent delegation boundary is tested independent of the numpy
    point-extraction path (no CUDA, no open3d)."""
    image = Image.new("RGB", (2, 2))
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1

    class _StubAgent:
        def detect(self, masks, depth, labels):
            return [
                Box3D(
                    center=(0.0, 0.0, 1.0),
                    extent=(0.5, 0.5, 0.5),
                    label="stub",
                    mask_id=0,
                )
            ], 0

    monkeypatch.setattr(
        "experiments.nooa_agent.tools.boxes3d._get_default_estimator",
        lambda: _StubAgent(),
    )
    depth = _depth_result(np.zeros((2, 2, 3), dtype=np.float32))
    boxes = detect_3d_boxes(image, [mask], depth)
    assert len(boxes) == 1
    assert boxes[0].label == "stub"
    assert boxes[0].extent == (0.5, 0.5, 0.5)
