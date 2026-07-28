"""Structural smoke tests for the SpatialAnnotator tools.

Runs without CUDA, without models, without NOOA — validates the pure-geometry
helpers and the tier-detection heuristic. End-to-end tests requiring model
weights and Python 3.12 live outside this file.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from experiments.nooa_agent.tools import detect_tier
from experiments.nooa_agent.tools.florence import Box, relative_position_2d
from experiments.nooa_agent.tools.depth import (
    DepthResult,
    _unproject,
    depth_at_point,
    distance_3d_meters,
)


# ── Box ────────────────────────────────────────────────────────────────

def test_box_area():
    assert Box(x1=0, y1=0, x2=10, y2=5).area == 50

def test_box_center():
    assert Box(x1=0, y1=0, x2=10, y2=20).center == (5, 10)

def test_box_to_list():
    assert Box(x1=1, y1=2, x2=3, y2=4, label="x").to_list() == [1, 2, 3, 4]


# ── relative_position_2d ───────────────────────────────────────────────

def test_relative_position_right_of_and_above():
    a = Box(x1=100, y1=200, x2=110, y2=210)  # center 105, 205
    b = Box(x1=300, y1=100, x2=310, y2=110)  # center 305, 105
    r = relative_position_2d(a, b)
    assert "right of" in r["b_is"]
    assert "above" in r["b_is"]
    assert r["distance_px"] > 0

def test_relative_position_aligned():
    a = Box(x1=100, y1=100, x2=110, y2=110)
    b = Box(x1=103, y1=105, x2=113, y2=115)  # <10px center offset in each axis
    r = relative_position_2d(a, b)
    assert "horizontally aligned with" in r["b_is"]
    assert "vertically aligned with" in r["b_is"]


# ── _unproject ─────────────────────────────────────────────────────────

def test_unproject_shape_and_z():
    depth = np.ones((4, 6), dtype=np.float32) * 2.5   # 2.5m everywhere
    K = np.array([[500, 0, 3], [0, 500, 2], [0, 0, 1]], dtype=np.float32)
    xyz = _unproject(depth, K)
    assert xyz.shape == (4, 6, 3)
    # Z channel should match input depth
    assert np.allclose(xyz[..., 2], 2.5)
    # Center pixel (u=3, v=2) with cx=3, cy=2 should have x=0, y=0
    assert np.allclose(xyz[2, 3, :2], 0.0)


# ── depth_at_point ─────────────────────────────────────────────────────

def test_depth_at_point_samples_correctly():
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)
    K = np.eye(3, dtype=np.float32)
    dr = DepthResult(depth_m=depth, focal_px=1.0, intrinsics_3x3=K,
                     point_cloud_xyz=None, backend="test")
    # depth[2, 3] = 2*6 + 3 = 15
    assert depth_at_point(dr, 3, 2) == 15.0

def test_depth_at_point_clamps_out_of_bounds():
    depth = np.zeros((4, 6), dtype=np.float32)
    depth[3, 5] = 99.0
    dr = DepthResult(depth_m=depth, focal_px=1.0,
                     intrinsics_3x3=np.eye(3, dtype=np.float32),
                     point_cloud_xyz=None, backend="test")
    # Way out of bounds should clamp to the last valid pixel (3, 5)
    assert depth_at_point(dr, 1000, 1000) == 99.0


# ── distance_3d_meters ─────────────────────────────────────────────────

def test_distance_3d_uses_point_cloud_when_available():
    # Two objects at known 3D coords via a fake point cloud
    H, W = 4, 6
    xyz = np.zeros((H, W, 3), dtype=np.float32)
    xyz[1, 1] = [0.0, 0.0, 0.0]
    xyz[3, 5] = [3.0, 4.0, 0.0]   # distance from origin = 5m
    dr = DepthResult(
        depth_m=np.ones((H, W), dtype=np.float32),
        focal_px=100, intrinsics_3x3=np.eye(3, dtype=np.float32),
        point_cloud_xyz=xyz, backend="test",
    )
    box_a = Box(x1=0.5, y1=0.5, x2=1.5, y2=1.5)  # center (1, 1)
    box_b = Box(x1=4.5, y1=2.5, x2=5.5, y2=3.5)  # center (5, 3)
    r = distance_3d_meters(dr, box_a, box_b)
    assert r["distance_m"] == 5.0
    assert r["dx_m"] == 3.0
    assert r["dy_m"] == 4.0
    assert r["backend"] == "test"


# ── detect_tier ────────────────────────────────────────────────────────

def test_detect_tier_env_override_cpu():
    with patch.dict(os.environ, {"VQASYNTH_AGENT_TIER": "cpu"}):
        assert detect_tier() == "cpu"

def test_detect_tier_env_override_gpu():
    with patch.dict(os.environ, {"VQASYNTH_AGENT_TIER": "gpu"}):
        assert detect_tier() == "gpu"

def test_detect_tier_no_cuda_returns_cpu():
    # Regardless of env, if CUDA isn't available we should get cpu
    with patch.dict(os.environ, {}, clear=True):
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if not has_cuda:
            assert detect_tier() == "cpu"
        else:
            # On a real GPU machine, tier depends on VRAM — just check it's valid
            assert detect_tier() in ("cpu", "gpu")


# ── FlorenceDetector construction (no model load — just import path) ───

def test_florence_detector_imports_cleanly():
    from experiments.nooa_agent.tools.florence import FlorenceDetector
    d = FlorenceDetector(device="cpu")
    # Backends are lazy-loaded — should not have loaded anything yet
    assert d._base is None
    assert d._large is None

def test_florence_segmenter_shares_detector():
    from experiments.nooa_agent.tools.florence import FlorenceDetector, FlorenceSegmenter
    d = FlorenceDetector(device="cpu")
    s = FlorenceSegmenter(detector=d)
    assert s._detector is d
