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


# ── device / dtype plumbing ────────────────────────────────────────────

def test_florence_detector_accepts_device_and_dtype():
    from experiments.nooa_agent.tools.florence import FlorenceDetector
    d = FlorenceDetector(device="cuda:1", dtype="fp16")
    assert d.device == "cuda:1"
    assert d.dtype == "fp16"

def test_depth_pro_accepts_device_and_dtype():
    from experiments.nooa_agent.tools.depth import DepthProEstimator
    d = DepthProEstimator(device="cuda", dtype="fp16")
    assert d.device == "cuda"
    assert d.dtype == "fp16"

def test_vggt_accepts_device_and_dtype_advisory():
    """VGGT's device/dtype are stored but advisory — SpatialSceneConstructor
    controls the actual placement. Verify the constructor accepts them so
    call-site signatures stay uniform across tiers."""
    from experiments.nooa_agent.tools.depth import VggtEstimator
    v = VggtEstimator(device="cuda:2", dtype="fp16")
    assert v.device == "cuda:2"
    assert v.dtype == "fp16"

def test_resolve_torch_dtype_aliases():
    import torch
    from experiments.nooa_agent.tools.florence import _resolve_torch_dtype
    assert _resolve_torch_dtype("fp16") is torch.float16
    assert _resolve_torch_dtype("float16") is torch.float16
    assert _resolve_torch_dtype("half") is torch.float16
    assert _resolve_torch_dtype("bf16") is torch.bfloat16
    assert _resolve_torch_dtype("fp32") is torch.float32
    assert _resolve_torch_dtype(torch.float16) is torch.float16
    assert _resolve_torch_dtype(None) is None

def test_resolve_torch_dtype_rejects_unknown_string():
    from experiments.nooa_agent.tools.florence import _resolve_torch_dtype
    with pytest.raises(ValueError, match="fp32"):
        _resolve_torch_dtype("not-a-dtype")


# ── DepthResult repr — pinned compact so NOOA traces don't bloat ───────

def test_depth_result_repr_is_compact_not_full_array():
    """Regression: default dataclass repr would dump the entire depth array
    + point cloud (~7 MB text for 768×768). NOOA logs return values into
    trace events; a heavy repr scales trace size linearly with question count.
    """
    depth = np.zeros((768, 768), dtype=np.float32)
    xyz = np.zeros((768, 768, 3), dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    r = DepthResult(depth_m=depth, focal_px=1450.7, intrinsics_3x3=K,
                    point_cloud_xyz=xyz, backend="vggt")
    text = repr(r)
    # A compact summary should be well under 200 chars regardless of image size
    assert len(text) < 200, f"repr is {len(text)} chars — probably dumping arrays"
    assert "vggt" in text
    assert "1450.7" in text
    assert "(768, 768)" in text
    assert "has_pointcloud=True" in text


# ── _scene_cached decorator ────────────────────────────────────────────

def test_scene_cache_hits_on_repeated_call():
    from experiments.nooa_agent.spatial_annotator import _scene_cached

    call_count = 0

    class Fake:
        @_scene_cached
        def compute(self, image, x):
            nonlocal call_count
            call_count += 1
            return x * 2

    fake = Fake()
    img = object()
    assert fake.compute(img, 5) == 10
    assert fake.compute(img, 5) == 10
    assert call_count == 1  # second call hit cache

def test_scene_cache_misses_on_different_args():
    from experiments.nooa_agent.spatial_annotator import _scene_cached

    call_count = 0

    class Fake:
        @_scene_cached
        def compute(self, image, x):
            nonlocal call_count
            call_count += 1
            return x * 2

    fake = Fake()
    img = object()
    fake.compute(img, 5)
    fake.compute(img, 6)   # different arg → miss
    assert call_count == 2

def test_scene_cache_invalidates_on_new_image():
    from experiments.nooa_agent.spatial_annotator import _scene_cached

    call_count = 0

    class Fake:
        @_scene_cached
        def compute(self, image, x):
            nonlocal call_count
            call_count += 1
            return x * 2

    fake = Fake()
    # Bind both objects to locals so neither is GC'd — a freed object() can
    # be re-issued at the same id, which would give a false cache hit.
    img_a = object()
    img_b = object()
    fake.compute(img_a, 5)
    fake.compute(img_b, 5)   # different object → miss
    assert call_count == 2
    assert fake._scene_image_ref is img_b   # strong ref pins the current image

def test_scene_cache_id_reuse_footgun_avoided():
    """Regression: if the decorator compared by id() alone, freeing img A and
    allocating img B could reuse A's address → false cache hit. The strong
    ref should prevent this.
    """
    from experiments.nooa_agent.spatial_annotator import _scene_cached

    call_count = 0

    class Fake:
        @_scene_cached
        def compute(self, image, x):
            nonlocal call_count
            call_count += 1
            return x * 2

    fake = Fake()
    fake.compute(object(), 5)   # ephemeral A — GC'd immediately
    fake.compute(object(), 5)   # ephemeral B — may reuse A's id
    # Even with id reuse, we should have missed cache on B (identity, not id)
    assert call_count == 2

def test_scene_cache_preserves_signature_for_nooa():
    """NOOA introspects tool signatures via inspect.signature; verify the
    decorator doesn't shadow the wrapped function's signature/annotations.

    Note: this module uses ``from __future__ import annotations`` (PEP 563)
    so annotations are stored as strings, not the actual types. NOOA-side
    resolution via ``typing.get_type_hints`` would materialize them; we just
    verify parameter names + docstring survive the decorator.
    """
    import inspect
    from experiments.nooa_agent.spatial_annotator import _scene_cached

    class Fake:
        @_scene_cached
        def compute(self, image: object, x: int) -> int:
            """A tool docstring NOOA will use."""
            return x * 2

    sig = inspect.signature(Fake.compute)
    assert list(sig.parameters) == ["self", "image", "x"]
    assert sig.parameters["x"].annotation in (int, "int")   # PEP 563 tolerant
    assert Fake.compute.__doc__ == "A tool docstring NOOA will use."
