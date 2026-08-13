"""Structural tests for the Metric3D v2 depth + learned-normal backend.

Mirrors ``test_surface_normals_tool.py``: verify the backend's pure adapters
and the learned-normal wiring against synthetic arrays, with no GPU, no model
download, and no torch import. The integration anchors are the *pre-existing*
``experiments.nooa_agent.tools.depth`` (the DepthResult contract + ``_unproject``
the backend joins) and ``experiments.nooa_agent.tools.surface_normals`` (the
normal surface whose ``confidence`` hook Metric3D v2's learned head fills): the
depth map flows through ``_unproject`` and the learned normals flow through
``surface_normals(..., learned_normals=...)`` — so this is not a self-test of
the new file alone. If the DepthResult contract or the surface_normals wiring
changes, these assertions move with it.
"""
from __future__ import annotations

import numpy as np

# Pre-existing modules — exercised here so the test isn't a self-test of only
# the new metric3d code. _unproject is the same geometry depth.py uses; the
# surface_normals() override is the existing surface that consumes learned
# normals and invokes the new module.
from experiments.nooa_agent.tools.depth import DepthResult, _unproject
from experiments.nooa_agent.tools.metric3d import (
    Metric3DEstimator,
    _sanitize_confidence,
    _split_normal_4ch,
    build_depth_result,
    build_normal_result,
)
from experiments.nooa_agent.tools.surface_normals import NormalResult, surface_normals


def _frontal_4ch(H, W, *, normal=(0.0, 0.0, -1.0), conf=0.9):
    """A synthetic Metric3D v2 prediction_normal: (H, W, 4) frontal plane."""
    out = np.zeros((H, W, 4), dtype=np.float32)
    out[..., 0] = normal[0]
    out[..., 1] = normal[1]
    out[..., 2] = normal[2]
    out[..., 3] = conf
    return out


# ---------------------------------------------------------------------------
# build_depth_result — joins the existing DepthResult contract
# ---------------------------------------------------------------------------
def test_build_depth_result_joins_depth_contract():
    H, W = 12, 16
    depth_m = np.full((H, W), 5.0, dtype=np.float32)

    dr = build_depth_result(depth_m)

    assert isinstance(dr, DepthResult)
    assert dr.backend == "metric3d"
    assert dr.depth_m.shape == (H, W)
    # Canonical focal default; intrinsics + point cloud built the same way the
    # other backends build them (so distance_3d_meters consumes this unchanged).
    assert dr.focal_px == 1000.0
    assert dr.intrinsics_3x3.shape == (3, 3)
    assert dr.point_cloud_xyz.shape == (H, W, 3)
    # Point cloud came through depth.py's own _unproject at the canonical K —
    # constant-depth frontal plane ⇒ z == 5.0 everywhere.
    np.testing.assert_allclose(dr.point_cloud_xyz[..., 2], 5.0, atol=1e-4)


def test_build_depth_result_focal_override_and_invalid_fill():
    H, W = 8, 8
    depth_m = np.full((H, W), 4.0, dtype=np.float32)
    depth_m[0, 0] = np.nan  # a masked region Metric3D v2 may emit

    dr = build_depth_result(depth_m, focal_px=700.0)

    assert dr.focal_px == 700.0
    # Invalid pixel is filled with the median of the valid depths, not left NaN
    # (mirrors FoundationGeo's handling so downstream sampling doesn't hit NaN).
    assert np.isfinite(dr.depth_m).all()
    assert dr.depth_m[0, 0] == 4.0


# ---------------------------------------------------------------------------
# build_normal_result — fills the NormalResult.confidence hook (the contribution)
# ---------------------------------------------------------------------------
def test_build_normal_result_populates_confidence_and_convention():
    H, W = 10, 12
    result = build_normal_result(_frontal_4ch(H, W, conf=0.9))

    assert isinstance(result, NormalResult)
    assert result.backend == "metric3d+learned_normals"
    assert result.normals_xyz.shape == (H, W, 3)
    # Metric3D v2 toward-camera convention preserved: frontal plane -> (0, 0, -1).
    np.testing.assert_allclose(result.normals_xyz[H // 2, W // 2], [0, 0, -1.0], atol=1e-5)
    # The hook surface_normals.py leaves None for its geometry proxy is filled.
    assert result.confidence is not None
    assert result.confidence.shape == (H, W)
    np.testing.assert_allclose(result.confidence, 0.9, atol=1e-5)


def test_build_normal_result_unit_normalizes_garbage_and_zeros_confidence():
    # A non-finite / zero learned normal: defensive unit-normalize keeps it a
    # zero vector (detectable downstream), and there's no confidence channel to
    # trust — but the 4ch input's own confidence still flows through.
    arr = _frontal_4ch(4, 4, conf=0.5)
    arr[0, 0, :3] = [0.0, 0.0, 0.0]  # degenerate normal
    result = build_normal_result(arr)

    assert np.linalg.norm(result.normals_xyz[0, 0]) < 1e-6  # zero, not NaN
    assert np.isfinite(result.normals_xyz).all()
    # Confidence on that pixel still comes from channel 3 (0.5), clamped clean.
    assert result.confidence[0, 0] == 0.5


# ---------------------------------------------------------------------------
# _split_normal_4ch / _sanitize_confidence — robustness to tensor layout
# ---------------------------------------------------------------------------
def test_split_normal_4ch_handles_channel_first_and_last():
    H, W = 5, 6
    last = _frontal_4ch(H, W)  # (H, W, 4)
    first = np.transpose(last, (2, 0, 1))  # (4, H, W)

    n_last, c_last = _split_normal_4ch(last)
    n_first, c_first = _split_normal_4ch(first)

    np.testing.assert_allclose(n_last, n_first, atol=1e-6)
    np.testing.assert_allclose(c_last, c_first, atol=1e-6)
    assert n_last.shape == (H, W, 3)


def test_split_normal_4ch_synthesizes_confidence_for_three_channel():
    # A checkpoint variant that drops the confidence channel still yields a
    # usable map (1.0 finite, 0.0 non-finite) so the hook is never empty.
    normals = _frontal_4ch(3, 3)[..., :3]
    n, c = _split_normal_4ch(normals)
    assert c.shape == (3, 3)
    assert (c == 1.0).all()


def test_sanitize_confidence_clamps_and_nans_to_zero():
    raw = np.array([0.2, 1.5, -0.3, np.nan, np.inf], dtype=np.float32)
    out = _sanitize_confidence(raw)
    assert out.max() == 1.0
    assert out.min() == 0.0
    # NaN *and* inf are "no signal" -> 0; the finite 1.5 clamps to 1.0, -0.3 to 0.0.
    np.testing.assert_allclose(out, [0.2, 1.0, 0.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Wiring: surface_normals (existing module) consumes learned normals via metric3d
# ---------------------------------------------------------------------------
def test_surface_normals_prefers_learned_and_fills_confidence():
    # Build a DepthResult the way depth.py's backends do, then ask the existing
    # surface_normals() for normals — once via the geometry proxy, once handing
    # it Metric3D v2's learned head. The learned path must invoke metric3d,
    # fill confidence, and tag the backend so the two are distinguishable.
    H, W = 8, 8
    focal = 100.0
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32)
    depth_m = np.full((H, W), 3.0, dtype=np.float32)
    dr = DepthResult(
        depth_m=depth_m,
        focal_px=focal,
        intrinsics_3x3=K,
        point_cloud_xyz=_unproject(depth_m, K),
        backend="metric3d",
    )

    geometry = surface_normals(dr)
    assert geometry.backend == "metric3d+pointcloud_normals"
    assert geometry.confidence is None  # proxy leaves the hook open

    learned = surface_normals(dr, learned_normals=_frontal_4ch(H, W, conf=0.8))
    assert learned.backend == "metric3d+learned_normals"
    assert learned.confidence is not None  # learned head fills it
    np.testing.assert_allclose(learned.confidence, 0.8, atol=1e-5)
    np.testing.assert_allclose(learned.normals_xyz[H // 2, W // 2], [0, 0, -1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Metric3DEstimator — constructible + lazy (no torch / no hub on the test host)
# ---------------------------------------------------------------------------
def test_estimator_constructs_without_loading_model():
    est = Metric3DEstimator(device="cpu", dtype="fp16", focal_px=712.0)
    assert est.model_name == "metric3d_vit_large"
    assert est.focal_px == 712.0
    assert est._model is None  # lazy: _load() not triggered by construction
