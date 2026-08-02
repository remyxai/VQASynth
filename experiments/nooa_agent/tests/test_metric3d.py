"""Smoke tests for the Metric3D v2 backend (tools/metric3d.py).

Mirrors the test_shapes.py philosophy: validates the wiring + the
parameter-free canonicalization geometry without CUDA, without the metric3d
package, and without model weights. Real end-to-end metric-depth + normal
inference belongs on a GPU host with the upstream package installed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# Imports from the EXISTING depth module (not the new metric3d module) — this is
# the integration surface. depth.py re-exports Metric3DEstimator, so reaching it
# through depth proves the new backend is wired onto the shared tool surface.
from experiments.nooa_agent.tools.depth import DepthResult, Metric3DEstimator
from experiments.nooa_agent.tools.metric3d import (
    CANONICAL_INTRINSICS,
    _resize_depth,
    _resize_image,
    _resize_normal,
    horizontal_fov_deg,
    select_canonical_intrinsic,
)


# ── device / dtype plumbing (matches the DepthPro/VGGT plumbing tests) ──

def test_metric3d_estimator_accepts_device_and_dtype():
    m = Metric3DEstimator(device="cuda:1", dtype="fp16")
    assert m.device == "cuda:1"
    assert m.dtype == "fp16"
    # Model loads lazily — construction must not pull in torch or the package.
    assert m._model is None


def test_metric3d_estimator_accepts_fov_override():
    m = Metric3DEstimator(device="cpu", dtype="fp32", fov_x_deg=62.0)
    assert m.fov_x_deg == 62.0


# ── canonicalization geometry (the parameter-free core; tested directly) ──

def test_horizontal_fov_known_values():
    # 2*arctan(W/(2*fx)): when W == 2*fx the half-angle is 45°, so FoV is 90°.
    assert horizontal_fov_deg(2 * 920.0, 920.0) == pytest.approx(90.0)
    # Very long focal → narrow (telephoto) FoV approaching 0.
    assert horizontal_fov_deg(1.0, 1e6) < 0.5
    # Wide FoV sanity: a 24mm-equivalent wide lens on a 36mm-wide "sensor".
    assert 70.0 < horizontal_fov_deg(36.0, 24.0) < 76.0  # ≈ 73.7°


def test_select_canonical_picks_nearest_fov():
    # An input FoV exactly matching one bin should select that bin.
    target_w, _target_h, target_fx = CANONICAL_INTRINSICS[0]
    target_fov = horizontal_fov_deg(target_w, target_fx)
    assert select_canonical_intrinsic(target_fov) == CANONICAL_INTRINSICS[0]

    # A small perturbation must not flip to a different bin (nearest wins).
    assert select_canonical_intrinsic(target_fov + 1.0) == CANONICAL_INTRINSICS[0]


def test_select_canonical_realistic_camera():
    # A typical phone main camera has ~65° horizontal FoV; the selected bin
    # must be the one whose FoV is closest to 65° (and within the bin spread).
    chosen = select_canonical_intrinsic(65.0)
    chosen_gap = abs(horizontal_fov_deg(chosen[0], chosen[2]) - 65.0)
    all_gaps = [abs(horizontal_fov_deg(c[0], c[2]) - 65.0) for c in CANONICAL_INTRINSICS]
    assert chosen_gap == pytest.approx(min(all_gaps))


def test_canonical_intrinsics_all_valid_pinhole():
    # Every canonical bin must be a real, finite, positive pinhole intrinsic.
    for w, h, fx in CANONICAL_INTRINSICS:
        assert w > 0 and h > 0 and fx > 0
        fov = horizontal_fov_deg(w, fx)
        assert math.isfinite(fov)
        assert 10.0 < fov < 120.0  # sensible camera range


# ── canonicalization resampling (image/depth/normal warps; no model needed) ──

def test_resize_image_changes_grid_keeps_channels():
    arr = (np.random.rand(20, 30, 3) * 255).astype(np.uint8)
    out = _resize_image(arr, width=50, height=40)
    assert out.shape == (40, 50, 3)
    assert out.dtype == np.uint8


def test_resize_depth_preserves_metric_range():
    # Canonicalization resamples the spatial grid but must NOT renormalize the
    # depth values — metric meters survive the warp.
    depth = np.linspace(1.0, 5.0, num=20 * 30, dtype=np.float32).reshape(20, 30)
    out = _resize_depth(depth, width=15, height=10)
    assert out.shape == (10, 15)
    assert out.min() >= 1.0 - 1e-3 and out.max() <= 5.0 + 1e-3


def test_resize_normal_yields_unit_vectors():
    normals = np.zeros((20, 30, 3), dtype=np.float32)
    normals[..., 2] = 1.0  # all facing camera
    out = _resize_normal(normals, width=60, height=40)
    assert out.shape == (40, 60, 3)
    # Every vector must be unit-length after the re-normalization step.
    norms = np.linalg.norm(out, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


# ── DepthResult.normal_map integration (the new field on the shared surface) ──

def test_depth_result_carries_normal_map():
    """The Metric3D normal head's output rides the shared DepthResult so
    downstream tools get it for free — verify the field exists and the compact
    repr reports it without dumping the array."""
    H, W = 8, 12
    depth = np.ones((H, W), dtype=np.float32) * 3.5
    normals = np.zeros((H, W, 3), dtype=np.float32)
    normals[..., 2] = 1.0  # facing the camera (+Z)
    K = np.eye(3, dtype=np.float32) * 100.0
    K[2, 2] = 1.0
    r = DepthResult(
        depth_m=depth,
        focal_px=100.0,
        intrinsics_3x3=K,
        point_cloud_xyz=None,
        normal_map=normals,
        backend="metric3d",
    )
    assert r.normal_map is not None
    assert r.normal_map.shape == (H, W, 3)
    assert r.backend == "metric3d"
    text = repr(r)
    assert "has_normals=True" in text
    assert len(text) < 200  # repr stays compact (regression guard from test_shapes)


def test_depth_result_backwards_compatible_without_normals():
    """Existing backends don't produce normals — normal_map must default to
    None so every pre-existing DepthResult constructor is unaffected."""
    r = DepthResult(
        depth_m=np.zeros((4, 4), dtype=np.float32),
        focal_px=50.0,
        intrinsics_3x3=np.eye(3, dtype=np.float32),
        point_cloud_xyz=None,
        backend="depthpro",
    )
    assert r.normal_map is None
    assert "has_normals=False" in repr(r)
    # Positional construction (the form used throughout test_shapes.py) survives.
    r2 = DepthResult(
        np.zeros((2, 2), dtype=np.float32), 50.0, np.eye(3, dtype=np.float32), None, "vggt"
    )
    assert r2.normal_map is None
