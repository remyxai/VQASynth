"""Structural tests for the NOOA surface-normals tool wrapper.

Mirrors ``test_orientation_tool.py``'s philosophy: verify the normal
derivation + facing/angle prose against synthetic geometry, with no GPU, no
model download, and no torch import. The integration anchor is the existing
``experiments.nooa_agent.tools.depth`` module — fixtures build a real
:class:`DepthResult` and (for the frontal-plane case) unproject it through
the *pre-existing* ``_unproject`` helper, so this is not a self-test of the
new file alone: if the DepthResult contract or the unprojection geometry
changes, these normals move with it.
"""
from __future__ import annotations

import numpy as np

# Pre-existing depth module — exercised here so the test isn't a self-test of
# only the new surface_normals code. _unproject is the same geometry the depth
# tool uses internally, so the frontal-plane fixture flows through the real
# DepthResult → point-cloud → normals path the agent would hit.
from experiments.nooa_agent.tools.depth import DepthResult, _unproject
from experiments.nooa_agent.tools.surface_normals import (
    NormalResult,
    normal_at_pixel,
    surface_angle,
    surface_facing,
    surface_normals,
)


def _depth_result(xyz, *, backend="depthpro"):
    """Bundle a point cloud into a DepthResult the way depth.py's backends do."""
    H, W = xyz.shape[:2]
    focal = 100.0
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32)
    return DepthResult(
        depth_m=np.full((H, W), 5.0, dtype=np.float32),
        focal_px=focal,
        intrinsics_3x3=K,
        point_cloud_xyz=xyz.astype(np.float32),
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Integration: DepthResult (existing module) → surface_normals → normals
# ---------------------------------------------------------------------------
def test_frontal_plane_normals_face_camera_via_real_unproject():
    # Frontal plane: constant depth. Unproject through depth.py's OWN helper so
    # the fixture isn't a parallel reimplementation of the geometry.
    H, W = 16, 20
    focal = 100.0
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32)
    depth_m = np.full((H, W), 5.0, dtype=np.float32)
    dr = DepthResult(
        depth_m=depth_m,
        focal_px=focal,
        intrinsics_3x3=K,
        point_cloud_xyz=_unproject(depth_m, K),
        backend="vggt",
    )

    result = surface_normals(dr)

    assert isinstance(result, NormalResult)
    assert result.normals_xyz.shape == (H, W, 3)
    # Metric3Dv2 convention: viewer-facing surface → normal ≈ (0, 0, -1).
    sample = result.normals_xyz[H // 2, W // 2]
    np.testing.assert_allclose(sample, [0.0, 0.0, -1.0], atol=1e-4)
    # Every interior pixel is unit-length and camera-facing.
    interior = result.normals_xyz[1:-1, 1:-1]
    norms = np.linalg.norm(interior, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)
    assert result.backend == "vggt+pointcloud_normals"


def test_floor_plane_normals_face_upward():
    # Hand-built point cloud for a horizontal surface: x increases right, z
    # decreases down-pixel, y held constant → normal (0, -1, 0) = faces up.
    H, W = 10, 12
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xyz = np.stack([j.astype(np.float32), np.zeros_like(j, dtype=np.float32),
                    -i.astype(np.float32)], axis=-1)
    dr = _depth_result(xyz, backend="depthpro")

    result = surface_normals(dr, smooth=0)  # raw cross product — synthetic input
    sample = result.normals_xyz[H // 2, W // 2]
    np.testing.assert_allclose(sample, [0.0, -1.0, 0.0], atol=1e-4)
    assert surface_facing(sample) == "faces upward"


def test_surface_normals_unprojects_when_pointcloud_missing():
    # DepthPro-style DepthResult carries no point cloud; surface_normals must
    # fall back to _unproject(depth_m, K) and still produce camera-facing normals.
    H, W = 8, 8
    focal = 80.0
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32)
    dr = DepthResult(
        depth_m=np.full((H, W), 3.0, dtype=np.float32),
        focal_px=focal,
        intrinsics_3x3=K,
        point_cloud_xyz=None,
        backend="depthpro",
    )
    result = surface_normals(dr)
    np.testing.assert_allclose(result.normals_xyz[H // 2, W // 2], [0, 0, -1.0], atol=1e-4)
    assert result.backend == "depthpro+pointcloud_normals"


# ---------------------------------------------------------------------------
# normal_at_pixel — same x=col / y=row convention as depth_at_point
# ---------------------------------------------------------------------------
def test_normal_at_pixel_clips_and_samples():
    H, W = 6, 7
    xyz = np.stack(
        [np.full((H, W), 0.5), np.zeros((H, W)), np.full((H, W), 4.0)], axis=-1
    ).astype(np.float32)
    result = surface_normals(_depth_result(xyz), smooth=0)
    # Out-of-bounds coords clip to the corner, not raise.
    n = normal_at_pixel(result, x=-100, y=9999)
    assert n.shape == (3,)
    # x=col / y=row: column 0 + constant-x cloud → finite sample returned.
    assert np.isfinite(n).all()


# ---------------------------------------------------------------------------
# surface_facing — deterministic prose mapping (unit-tested in isolation)
# ---------------------------------------------------------------------------
def test_surface_facing_buckets_each_axis():
    assert surface_facing(np.array([0.0, 0.0, -1.0])) == "faces the camera"
    assert surface_facing(np.array([0.0, 0.0, 1.0])) == "faces away from the camera"
    assert surface_facing(np.array([0.0, -1.0, 0.0])) == "faces upward"
    assert surface_facing(np.array([0.0, 1.0, 0.0])) == "faces downward"
    assert surface_facing(np.array([-1.0, 0.0, 0.0])) == "faces left"
    assert surface_facing(np.array([1.0, 0.0, 0.0])) == "faces right"


def test_surface_facing_handles_degenerate_input():
    assert surface_facing(np.zeros(3)) == "unknown surface"
    assert surface_facing(np.array([np.nan, 0.0, 0.0])) == "unknown surface"
    # Magnitude is ignored — direction is what matters.
    assert surface_facing(np.array([0.0, 0.0, -7.5])) == "faces the camera"


# ---------------------------------------------------------------------------
# surface_angle — parallel / anti-parallel, the "resting flat" signal
# ---------------------------------------------------------------------------
def test_surface_angle_parallel_and_antiparallel():
    up = np.array([0.0, -1.0, 0.0])
    flat = surface_angle(up, up)
    assert flat["angle_deg"] == 0.0
    assert flat["parallel"] is True
    assert flat["anti_parallel"] is False
    assert "parallel" in flat["relation"]

    # Tabletop faces up (0,-1,0); an object resting on it has its bottom face
    # pointing down (0,1,0) → normals anti-parallel → "resting flush".
    resting = surface_angle(np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert resting["angle_deg"] == 180.0
    assert resting["anti_parallel"] is True
    assert "flush" in resting["relation"]


def test_surface_angle_orthogonal_is_in_between():
    d = surface_angle(np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]))
    assert 80.0 <= d["angle_deg"] <= 100.0
    assert d["parallel"] is False and d["anti_parallel"] is False


# ---------------------------------------------------------------------------
# NormalResult.__repr__ — compact, mirrors DepthResult.__repr__'s guard
# ---------------------------------------------------------------------------
def test_normal_result_repr_is_compact():
    r = NormalResult(normals_xyz=np.zeros((8, 8, 3), dtype=np.float32), backend="x")
    text = repr(r)
    assert len(text) < 120
    assert "pointcloud_normals" not in text  # full array must NOT be dumped
    assert "NormalResult" in text
    assert "\n" not in text
