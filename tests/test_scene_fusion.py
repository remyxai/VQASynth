"""CPU tests for the multi-view fusion logic in vqasynth.scene_fusion.

Mirrors the philosophy of ``tests/test_vggt_speedups.py`` /
``tests/test_correspondence.py``: verify the fusion/registration mechanics
with VGGT fully mocked — no GPU, no model download, no real ``vggt`` install
(the package is installed by the stage image, not requirements.txt). A
reference pinhole unprojection implementing VGGT's documented camera model
(OpenCV camera-from-world extrinsics inverted analytically, pixel-division
intrinsics) stands in for ``unproject_depth_map_to_point_map``, so what is
under test is scene_fusion's OWN logic: per-frame extrinsic indexing,
frame-major concatenation into one registered cloud, cross-frame object
merging, metadata surfacing, and the single-image back-compat path.

The synthetic scene is a world-space plane z=3 viewed by two cameras with
known poses — a 3D point visible in both frames must map to ~the same world
coordinate from each, which is exactly the registration property multi-view
fusion is supposed to deliver. Real multi-frame VGGT runs are validated on a
GPU host via the bundled PR, not here.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

# ────────────────────────────────────────────────────────────────────────
# vggt stubs — installed before vqasynth.scene_fusion is imported
# ────────────────────────────────────────────────────────────────────────

_CURRENT_SCENE = {}


def _reference_unproject(depth_map, extrinsics_cam, intrinsics_cam):
    """Reference implementation of VGGT's documented unprojection.

    world = R^T (p_cam - t) for OpenCV camera-from-world [R|t] extrinsics,
    p_cam = ((u-cu)*d/fu, (v-cv)*d/fv, d) for pixel (u, v) of depth d.
    """
    depth_map = np.asarray(depth_map, dtype=np.float64)
    extrinsics_cam = np.asarray(extrinsics_cam, dtype=np.float64)
    intrinsics_cam = np.asarray(intrinsics_cam, dtype=np.float64)
    S, H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    out = np.zeros((S, H, W, 3))
    for s in range(S):
        K = intrinsics_cam[s]
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        d = depth_map[s]
        p_cam = np.stack(
            [(u - cu) * d / fu, (v - cv) * d / fv, d], axis=-1
        )
        R = extrinsics_cam[s][:, :3]
        t = extrinsics_cam[s][:, 3]
        out[s] = (p_cam - t) @ R  # row form of R^T @ (p_cam - t)
    return out


def _install_vggt_stubs():
    """Fake the three vggt imports of vqasynth/scene_fusion.py.

    The fake heads ignore their token inputs and replay the geometry the
    test planted in ``_CURRENT_SCENE`` (the camera head "predicts" the known
    poses; the depth head replays synthetic depth + confidence), so
    ``create_point_cloud_from_model``'s real orchestration code runs on
    controlled geometry.
    """
    vggt = types.ModuleType("vggt")
    vggt_models = types.ModuleType("vggt.models.vggt")
    pose_enc = types.ModuleType("vggt.utils.pose_enc")
    geometry = types.ModuleType("vggt.utils.geometry")

    class _StubVGGT:
        @classmethod
        def from_pretrained(cls, repo_id):
            return cls()

        def to(self, device=None, dtype=None):
            return self

        def eval(self):
            return self

        def aggregator(self, images):
            return images, None

        def camera_head(self, tokens):
            return [tokens]

        def depth_head(self, tokens, images, ps_idx):
            scene = _CURRENT_SCENE
            depth = torch.from_numpy(scene["depth"]).unsqueeze(0)  # (1,S,H,W)
            conf = torch.from_numpy(scene["conf"]).unsqueeze(0)
            return depth, conf

    def _pose_encoding_to_extri_intri(pose_enc, image_size_hw=None, **kwargs):
        scene = _CURRENT_SCENE
        extr = torch.from_numpy(scene["extrinsics"]).unsqueeze(0)  # (1,S,3,4)
        intr = torch.from_numpy(scene["intrinsics"]).unsqueeze(0)  # (1,S,3,3)
        return extr, intr

    geometry.unproject_depth_map_to_point_map = _reference_unproject
    pose_enc.pose_encoding_to_extri_intri = _pose_encoding_to_extri_intri
    vggt_models.VGGT = _StubVGGT

    for name, mod in [
        ("vggt", vggt),
        ("vggt.models", types.ModuleType("vggt.models")),
        ("vggt.models.vggt", vggt_models),
        ("vggt.utils", types.ModuleType("vggt.utils")),
        ("vggt.utils.pose_enc", pose_enc),
        ("vggt.utils.geometry", geometry),
    ]:
        sys.modules[name] = mod


_install_vggt_stubs()

from vqasynth.scene_fusion import (  # noqa: E402
    SpatialSceneConstructor,
    calibrate_metric_scale,
    compute_object_depth_confidence,
    preprocess_images,
)

# ────────────────────────────────────────────────────────────────────────
# Synthetic two-camera scene: world plane z=3, known poses
# ────────────────────────────────────────────────────────────────────────

PLANE_Z = 3.0


def _frame_image(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))


# The preprocessed grid the model (and the planted depth maps) live on.
_grid = preprocess_images([_frame_image()])
GRID_H, GRID_W = int(_grid.shape[-2]), int(_grid.shape[-1])
HW = GRID_H * GRID_W
FU = float(GRID_W)
CU, CV = GRID_W / 2.0, GRID_H / 2.0


def _rot_y(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _extrinsic(R, camera_center):
    """OpenCV camera-from-world [R|t] for a camera at ``camera_center``."""
    t = -R @ np.asarray(camera_center, dtype=np.float64)
    return np.hstack([R, t.reshape(3, 1)])


CAM0 = _extrinsic(np.eye(3), np.zeros(3))
CAM1 = _extrinsic(_rot_y(8.0), np.array([0.3, 0.05, 0.1]))


def _plane_depth(extrinsic):
    """Depth of the world plane z=PLANE_Z along each pixel ray of a camera."""
    R = extrinsic[:, :3]
    t = extrinsic[:, 3]
    center = -R.T @ t
    u, v = np.meshgrid(np.arange(GRID_W), np.arange(GRID_H))
    dcam = np.stack(
        [(u - CU) / FU, (v - CV) / FU, np.ones_like(u, dtype=np.float64)], axis=-1
    )
    dw = dcam @ R  # R^T @ d, row form
    return (PLANE_Z - center[2]) / dw[..., 2]


def _set_scene(extrinsics, conf_values):
    intr = np.array([[FU, 0.0, CU], [0.0, FU, CV], [0.0, 0.0, 1.0]])
    _CURRENT_SCENE.clear()
    _CURRENT_SCENE.update(
        extrinsics=np.asarray(extrinsics, dtype=np.float64),
        intrinsics=np.stack([intr] * len(extrinsics)),
        depth=np.stack([_plane_depth(e) for e in extrinsics]),
        conf=np.stack(
            [np.full((GRID_H, GRID_W), float(c)) for c in conf_values]
        ),
    )
    return dict(_CURRENT_SCENE)


@pytest.fixture
def constructor(monkeypatch):
    monkeypatch.delenv("VQASYNTH_AVGGT_STEP1", raising=False)
    return SpatialSceneConstructor()


# ────────────────────────────────────────────────────────────────────────
# create_point_cloud_from_model — single-image back-compat
# ────────────────────────────────────────────────────────────────────────


def test_single_image_returns_original_three_tuple(constructor):
    scene = _set_scene([CAM0], conf_values=[1.5])
    result = constructor.create_point_cloud_from_model(_frame_image())
    assert len(result) == 3
    pcd, depth_map_np, focal_val = result
    # (H, W) depth, scalar focal — the historical contract (see
    # experiments/nooa_agent/tools/depth.py::VggtEstimator.metric_depth).
    assert depth_map_np.shape == (GRID_H, GRID_W)
    assert np.allclose(depth_map_np, scene["depth"][0])
    assert focal_val == FU
    ref = _reference_unproject(scene["depth"], scene["extrinsics"], scene["intrinsics"])
    assert np.allclose(np.asarray(pcd.points), ref.reshape(-1, 3), atol=1e-6)


def test_single_frame_list_matches_lone_image(constructor):
    scene = _set_scene([CAM0], conf_values=[1.5])
    lone = constructor.create_point_cloud_from_model(_frame_image(1))
    listed = constructor.create_point_cloud_from_model([_frame_image(1)])
    assert np.allclose(np.asarray(lone[0].points), np.asarray(listed[0].points))
    assert np.allclose(lone[1], listed[1])
    assert lone[2] == listed[2]


# ────────────────────────────────────────────────────────────────────────
# create_point_cloud_from_model — multi-view fusion + registration
# ────────────────────────────────────────────────────────────────────────


def test_fused_cloud_is_registered_and_spans_all_frames(constructor):
    scene = _set_scene([CAM0, CAM1], conf_values=[1.5, 2.5])
    pcd, depth_map_np, focal_val, metadata = constructor.create_point_cloud_from_model(
        [_frame_image(0), _frame_image(1)], return_metadata=True
    )

    assert metadata.num_frames == 2
    assert metadata.frame_shape == (GRID_H, GRID_W)
    assert np.allclose(metadata.extrinsics, scene["extrinsics"])
    assert np.allclose(metadata.intrinsics, scene["intrinsics"])
    assert metadata.focals == [FU, FU]
    assert focal_val == FU  # primary frame keeps the historical scalar

    # Stacked per-frame depth for S>1 (documented return shape).
    assert depth_map_np.shape == (2, GRID_H, GRID_W)

    coords = np.asarray(pcd.points)
    assert coords.shape == (2 * HW, 3)

    # The fused cloud spans all frames, in frame order, each chunk placed by
    # its OWN extrinsic — using the wrong pose for a frame shows up here.
    ref = _reference_unproject(scene["depth"], scene["extrinsics"], scene["intrinsics"])
    assert np.allclose(coords[:HW], ref[0].reshape(-1, 3), atol=1e-6)
    assert np.allclose(coords[HW:], ref[1].reshape(-1, 3), atol=1e-6)

    # Registration consistency: the world point at frame 0's center pixel is
    # exactly (0, 0, PLANE_Z); project it into frame 1 and the fused cloud
    # must place both observations at ~the same world coordinate.
    u0 = v0 = int(CU)
    idx0 = 0 * HW + v0 * GRID_W + u0
    q = coords[idx0]
    assert np.allclose(q, [0.0, 0.0, PLANE_Z], atol=1e-6)

    R1, t1 = CAM1[:, :3], CAM1[:, 3]
    center1 = -R1.T @ t1
    p1_cam = R1 @ (q - center1)
    u1 = int(round(FU * p1_cam[0] / p1_cam[2] + CU))
    v1 = int(round(FU * p1_cam[1] / p1_cam[2] + CV))
    idx1 = 1 * HW + v1 * GRID_W + u1
    assert np.allclose(q, coords[idx1], atol=0.05), (
        "the same 3D point observed in two frames must land at ~the same "
        f"world coordinate from both (got {q} vs {coords[idx1]})"
    )


# ────────────────────────────────────────────────────────────────────────
# Phase-1a/1b stand-in helpers
# ────────────────────────────────────────────────────────────────────────


def test_calibrate_metric_scale_is_median_depth():
    assert calibrate_metric_scale([3.0, 2.0, 4.0, 10.0]) == 3.5


def test_compute_object_depth_confidence_is_mean():
    assert compute_object_depth_confidence([1.0, 2.0, 3.0]) == 2.0
    assert compute_object_depth_confidence([]) == 0.0


# ────────────────────────────────────────────────────────────────────────
# run — per-frame masks, cross-frame object merge, stats
# ────────────────────────────────────────────────────────────────────────


def _rect_mask(r0, r1, c0, c1):
    m = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
    m[r0:r1, c0:c1] = 255
    return m


RECT0 = (200, 240, 250, 290)  # object 0 in frame 0 (40x40)
RECT1 = (150, 170, 260, 300)  # object 0 in frame 1 (20x40)
RECT2 = (50, 90, 50, 90)      # object 1 in frame 0 only (40x40)


def _rect(depth_frame, rect):
    r0, r1, c0, c1 = rect
    return depth_frame[r0:r1, c0:c1].ravel()


def test_run_merges_object_across_frames(constructor, tmp_path):
    scene = _set_scene([CAM0, CAM1], conf_values=[1.5, 2.5])
    per_frame_masks = [
        [_rect_mask(*RECT0), _rect_mask(*RECT2)],  # frame 0: objects 0 and 1
        [_rect_mask(*RECT1)],                      # frame 1: object 0 only
    ]

    result = constructor.run(
        "scene", [_frame_image(0), _frame_image(1)], per_frame_masks, str(tmp_path),
        return_scene_stats=True,
    )
    assert len(result) == 5
    filepaths, canonicalized, depth_map_np, focal_val, scene_stats = result

    assert len(filepaths) == 2
    assert all(os.path.exists(f) for f in filepaths)
    assert scene_stats["num_frames"] == 2
    assert np.asarray(scene_stats["extrinsics"]).shape == (2, 3, 4)
    assert depth_map_np.shape == (2, GRID_H, GRID_W)

    obj0, obj1 = scene_stats["objects"]
    n0 = (RECT0[1] - RECT0[0]) * (RECT0[3] - RECT0[2])
    n1 = (RECT1[1] - RECT1[0]) * (RECT1[3] - RECT1[2])

    # Object 0 appears in both frames -> its merged cloud carries points
    # from each, with scale/conf aggregated over ALL of its pixels.
    assert obj0["frames_seen"] == [0, 1]
    assert obj0["point_count"] == n0 + n1
    expected_scale = np.median(np.concatenate(
        [_rect(scene["depth"][0], RECT0), _rect(scene["depth"][1], RECT1)]
    ))
    assert obj0["metric_scale"] == pytest.approx(expected_scale, rel=1e-6)
    expected_conf = (n0 * 1.5 + n1 * 2.5) / (n0 + n1)
    assert obj0["depth_confidence"] == pytest.approx(expected_conf, rel=1e-6)

    # Object 1 was only segmented in frame 0 -> merged from that frame alone.
    assert obj1["frames_seen"] == [0]
    assert obj1["point_count"] == (RECT2[1] - RECT2[0]) * (RECT2[3] - RECT2[2])


def test_run_flat_masks_stick_to_primary_frame(constructor, tmp_path):
    """The historical flat mask list applies to the primary (first) frame."""
    _set_scene([CAM0, CAM1], conf_values=[1.5, 2.5])
    flat_masks = [_rect_mask(*RECT0)]  # not nested -> single-frame masks

    _, _, _, _, scene_stats = constructor.run(
        "scene", [_frame_image(0), _frame_image(1)], flat_masks, str(tmp_path),
        return_scene_stats=True,
    )
    (obj0,) = scene_stats["objects"]
    assert obj0["frames_seen"] == [0]
    assert obj0["point_count"] == (RECT0[1] - RECT0[0]) * (RECT0[3] - RECT0[2])


def test_run_original_resolution_masks_resample_onto_grid(constructor, tmp_path):
    """Pipeline masks arrive at the ORIGINAL image resolution (here 32x32) —
    they must be resampled onto the 518 grid the fused cloud lives on."""
    _set_scene([CAM0, CAM1], conf_values=[1.5, 2.5])
    small = np.zeros((32, 32), dtype=np.uint8)
    small[8:24, 8:24] = 255  # 16x16 source rect -> ~1/4 of the grid's area

    _, _, _, _, scene_stats = constructor.run(
        "scene", [_frame_image(0), _frame_image(1)], [[small], [small]], str(tmp_path),
        return_scene_stats=True,
    )
    (obj0,) = scene_stats["objects"]
    assert obj0["frames_seen"] == [0, 1]
    # Same source rect in both frames -> ~equal contributions, ~2x either.
    per_frame = obj0["point_count"] / 2.0
    expected = (16 * 518 / 32) ** 2  # 16x16 source pixels upscaled 32->518
    assert per_frame == pytest.approx(expected, rel=0.05)


def test_run_single_image_keeps_four_tuple(constructor, tmp_path):
    """README / examples unpack exactly four values from run()."""
    _set_scene([CAM0], conf_values=[1.5])
    result = constructor.run(
        "scene", _frame_image(0), [_rect_mask(*RECT0)], str(tmp_path)
    )
    assert len(result) == 4
    filepaths, canonicalized, depth_map_np, focal_val = result
    assert len(filepaths) == 1
    assert depth_map_np.shape == (GRID_H, GRID_W)
    assert focal_val == FU
    assert isinstance(canonicalized, bool)


# ────────────────────────────────────────────────────────────────────────
# apply_transform — dataset column surfacing
# ────────────────────────────────────────────────────────────────────────


def test_apply_transform_unbatched_multiview(constructor, tmp_path):
    _set_scene([CAM0, CAM1], conf_values=[1.5, 2.5])
    example = {
        "image": [_frame_image(0), _frame_image(1)],
        "masks": [
            [_rect_mask(*RECT0).tolist(), _rect_mask(*RECT2).tolist()],
            [_rect_mask(*RECT1).tolist()],
        ],
    }
    out = constructor.apply_transform(example, 0, str(tmp_path), "image")

    # One scene of two fused frames: per-frame camera poses surfaced as a
    # frame-major column, per-object stats aligned with the saved clouds.
    assert np.asarray(out["camera_extrinsics"]).shape == (1, 2, 3, 4)
    assert np.asarray(out["camera_intrinsics"]).shape == (1, 2, 3, 3)
    assert len(out["pointclouds"][0]) == 2
    assert len(out["object_metric_scale"][0]) == 2
    assert len(out["object_depth_confidence"][0]) == 2
    assert out["focallength"] == [[FU, FU]]  # one row, per-frame focals
    assert np.asarray(out["depth_map"][0]).shape == (2, GRID_H, GRID_W)


def test_apply_transform_batched_single_images(constructor, tmp_path):
    """Batched single-image rows keep their historical column shapes."""
    _set_scene([CAM0], conf_values=[1.5])
    example = {
        "image": [_frame_image(0), _frame_image(1)],
        "masks": [[_rect_mask(*RECT0).tolist()], [_rect_mask(*RECT2).tolist()]],
    }
    out = constructor.apply_transform(example, [0, 1], str(tmp_path), "image")

    assert len(out["pointclouds"]) == 2
    assert np.asarray(out["camera_extrinsics"]).shape == (2, 1, 3, 4)
    assert np.asarray(out["depth_map"][0]).shape == (GRID_H, GRID_W)
    assert out["focallength"] == [FU, FU]  # scalars, as before
