"""Tests for vqasynth.pointmap_calibration — FoundationGeo-style pixel-wise
metric calibration (Stage-2 scale + ray-direction correction fields) applied
as an analytic post-hoc pass on VGGT's back-projection.

Capability tests exercise the calibration directly (numpy-only, no VGGT, no
CUDA). The wiring test confirms the gated call-site edit landed in the
existing ``SpatialSceneConstructor`` module.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from vqasynth.pointmap_calibration import (
    _parse_intrinsic,
    _ray_direction_correction,
    _scale_field,
    calibrate_point_map,
)


def _plain_pinhole(depth, fx, fy, cx, cy):
    """Reference z-depth pinhole back-projection (what VGGT's unproject does)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    return np.stack([depth * xn, depth * yn, depth], axis=-1)


# --------------------------------------------------------------------------- #
# scale field (spatially-varying metric alignment) — parameter-free proxy
# --------------------------------------------------------------------------- #

def test_scale_field_reference_match_is_identity():
    """When the inferred focal equals the reference focal, the scale field is 1
    everywhere — i.e. no detected focal-length OOD bias ⇒ no correction."""
    xn = np.array([[0.0, 0.5, 1.0]])
    yn = np.zeros_like(xn)
    s = _scale_field(xn, yn, focal_px=100.0, reference_focal=100.0, strength=0.5)
    assert np.allclose(s, 1.0)


def test_scale_field_is_unity_on_axis_and_grows_radially():
    """Off-axis rays are more sensitive to focal error, so the correction must be
    unity at the principal point and grow toward the image border."""
    xn = np.array([[0.0, 1.0]])
    yn = np.zeros_like(xn)
    # focal 2× the reference ⇒ log_dev = log(2)
    s = _scale_field(xn, yn, focal_px=200.0, reference_focal=100.0, strength=0.5)
    assert np.isclose(s[0, 0], 1.0)            # on-axis: no correction
    assert np.isclose(s[0, 1], np.sqrt(2.0))   # border: exp(0.5 * log 2)
    assert s[0, 1] > s[0, 0]


def test_scale_field_is_a_field_not_a_global_scalar():
    """The paper's whole point: metric alignment must vary spatially. With a focal
    deviation, the field takes more than one distinct value across the image."""
    xn = np.array([[0.0, 0.5, 1.0]])
    yn = np.zeros_like(xn)
    s = _scale_field(xn, yn, focal_px=150.0, reference_focal=100.0, strength=0.5)
    assert len(np.unique(np.round(s, 6))) == 3   # genuinely spatially varying


# --------------------------------------------------------------------------- #
# ray-direction correction field — parameter-free proxy
# --------------------------------------------------------------------------- #

def test_ray_correction_zero_is_identity():
    xn = np.array([[1.0]])
    yn = np.array([[2.0]])
    xc, yc = _ray_direction_correction(xn, yn, k=0.0)
    assert np.allclose(xc, xn) and np.allclose(yc, yn)


def test_ray_correction_bends_off_axis_rays():
    """A non-zero radial coefficient bends the normalized ray direction outward."""
    xn = np.array([[1.0]])
    yn = np.array([[2.0]])
    xc, yc = _ray_direction_correction(xn, yn, k=0.1)
    # r² = 5, factor = 1 + 0.1*5 = 1.5
    assert np.isclose(xc, 1.5)
    assert np.isclose(yc, 3.0)


# --------------------------------------------------------------------------- #
# end-to-end calibrate_point_map
# --------------------------------------------------------------------------- #

def _toy_depth(h=6, w=8, value=2.0):
    return np.full((h, w), value, dtype=np.float64)


def _toy_intrinsic(fx=100.0, fy=100.0, cx=4.0, cy=3.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def test_calibrate_shape_and_dtype():
    depth = _toy_depth()
    pts = calibrate_point_map(depth, None, _toy_intrinsic(), reference_focal=100.0)
    assert pts.shape == (6, 8, 3)
    assert pts.dtype == np.float64


def test_calibrate_identity_matches_plain_pinhole():
    """With no distortion and focal == reference, calibration == raw pinhole
    unprojection (both calibration fields are the identity when no bias is
    detected — correct behavior, not a no-op baseline)."""
    depth = _toy_depth()
    K = _toy_intrinsic()
    pts = calibrate_point_map(depth, None, K, reference_focal=100.0)
    expected = _plain_pinhole(depth, 100.0, 100.0, 4.0, 3.0)
    assert np.allclose(pts, expected)


def test_calibrate_scale_field_changes_border_not_center():
    """focal > reference ⇒ scale field rescales metric depth, growing toward the
    border while leaving the principal-point ray on-axis (unchanged)."""
    depth = _toy_depth()
    K = _toy_intrinsic(fx=100.0)               # inferred focal
    pts = calibrate_point_map(depth, None, K, reference_focal=50.0)  # OOD: 2× ref

    # principal point (u=cx=4, v=cy=3) is on-axis ⇒ scale 1 ⇒ depth unchanged
    assert np.isclose(pts[3, 4, 2], 2.0)
    # a corner pixel is at max radius ⇒ scale = sqrt(2) ⇒ metric depth larger
    assert pts[0, 0, 2] > 2.0
    assert np.isclose(pts[0, 0, 2], 2.0 * np.sqrt(2.0))


def test_calibrate_ray_correction_changes_xy_not_z():
    """Distortion bends ray directions (X,Y) without touching the metric depth
    (Z) when the scale field is identity (focal == reference)."""
    depth = _toy_depth()
    K = _toy_intrinsic()
    base = calibrate_point_map(depth, None, K, reference_focal=100.0, distortion=0.0)
    bent = calibrate_point_map(depth, None, K, reference_focal=100.0, distortion=0.05)
    # off-axis pixel: X/Y differ
    assert not np.allclose(base[0, 0, 0], bent[0, 0, 0])
    # metric depth (Z) unchanged by the ray-direction field
    assert np.allclose(base[..., 2], bent[..., 2])


def test_calibrate_extrinsic_none_is_camera_space():
    depth = _toy_depth()
    pts = calibrate_point_map(depth, None, _toy_intrinsic(), reference_focal=100.0)
    expected = _plain_pinhole(depth, 100.0, 100.0, 4.0, 3.0)
    assert np.allclose(pts, expected)


def test_calibrate_extrinsic_rotation_3x3():
    """A 3x3 rotation extrinsic rotates camera-space points into world space."""
    depth = _toy_depth()
    K = _toy_intrinsic()
    cam = calibrate_point_map(depth, None, K, reference_focal=100.0)
    # 90° rotation about Z: (x, y, z) -> (-y, x, z)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world = calibrate_point_map(depth, R, K, reference_focal=100.0)
    expected = np.stack([-cam[..., 1], cam[..., 0], cam[..., 2]], axis=-1)
    assert np.allclose(world, expected)


def test_calibrate_extrinsic_3x4_applies_translation():
    depth = _toy_depth()
    K = _toy_intrinsic()
    t = np.array([1.0, 2.0, 3.0])
    Rt = np.column_stack([np.eye(3), t])  # identity rotation, translation t
    cam = calibrate_point_map(depth, None, K, reference_focal=100.0)
    world = calibrate_point_map(depth, Rt, K, reference_focal=100.0)
    assert np.allclose(world, cam + t)


def test_parse_intrinsic_accepts_matrix_and_batched():
    fx, fy, cx, cy = _parse_intrinsic([[100.0, 0, 4.0], [0, 110.0, 3.0], [0, 0, 1]])
    assert (fx, fy, cx, cy) == (100.0, 110.0, 4.0, 3.0)
    # batched (1,3,3) -> trailing 3x3
    K = np.array([[[100.0, 0, 4.0], [0, 110.0, 3.0], [0, 0, 1]]])
    fx2, fy2, cx2, cy2 = _parse_intrinsic(K)
    assert (fx2, fy2, cx2, cy2) == (100.0, 110.0, 4.0, 3.0)


def test_calibrate_accepts_torch_input():
    """The call site passes a (possibly bf16/cuda) VGGT depth tensor; the module
    must coerce torch tensors without importing torch at module load."""
    torch = pytest.importorskip("torch")
    depth = torch.full((6, 8), 2.0)
    K = torch.tensor([[100.0, 0, 4.0], [0, 100.0, 3.0], [0, 0, 1]])
    pts = calibrate_point_map(depth, None, K, reference_focal=100.0)
    expected = _plain_pinhole(_toy_depth(), 100.0, 100.0, 4.0, 3.0)
    assert isinstance(pts, np.ndarray)
    assert np.allclose(pts, expected)


# --------------------------------------------------------------------------- #
# integration: the gated call-site edit landed in the existing module
# --------------------------------------------------------------------------- #

def test_scene_fusion_wires_metric_calibration():
    """The non-new SpatialSceneConstructor module must expose the opt-in flag and
    route the back-projection through calibrate_point_map when it is set.

    Verified by AST rather than import because scene_fusion pulls in the full
    VGGT stack at module load; the capability itself is covered above.
    """
    import vqasynth  # non-new package (light __init__)

    scene_fusion_src = Path(vqasynth.__file__).with_name("scene_fusion.py").read_text()
    tree = ast.parse(scene_fusion_src)

    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "SpatialSceneConstructor"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert "metric_calibration" in {a.arg for a in init.args.args}

    cpc = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_point_cloud_from_model"
    )
    body = ast.unparse(cpc)
    assert "self.metric_calibration" in body          # gated
    assert "calibrate_point_map" in body              # routes through the capability
    assert "unproject_depth_map_to_point_map" in body  # default path preserved
