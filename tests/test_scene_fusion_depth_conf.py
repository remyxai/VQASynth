"""CPU tests for VGGT depth-confidence aggregation in vqasynth.scene_fusion.

Mirrors the philosophy of ``tests/test_detection_3d.py`` and
``tests/test_correspondence.py``: exercise the aggregation logic on synthetic
inputs — no GPU, no VGGT weights, no model download, no SAM. Per the design
brief, this CPU test is the acceptance bar; real end-to-end validation
(VGGT ``depth_conf`` -> SAM2 masks -> per-object confidence column) belongs on
a GPU host.

``vqasynth.scene_fusion`` imports the ``vggt`` package at module top but only
needs it when VGGT actually runs, so this suite installs minimal stub modules
for it before the import when the real package is absent. Everything else the
module needs at import time (torch, open3d, PIL) is importorskip-guarded,
following ``tests/test_judge_dataset.py``.

The alignment test replays the exact mask-preprocessing lines from
``SpatialSceneConstructor.run`` through the REAL pre-existing
``preprocess_images``, so the conf-map-vs-mask grid contract is checked against
the module's own code path rather than a re-implementation.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("open3d")  # vqasynth.scene_fusion imports it at module top
pytest.importorskip("PIL")


def _install_vggt_stubs():
    """Stand in for the vggt package so scene_fusion imports without it.

    Only installed when the real package is absent; the stubs are never
    exercised — these tests never instantiate VGGT.
    """
    if importlib.util.find_spec("vggt") is not None:
        return
    models_vggt = types.ModuleType("vggt.models.vggt")
    models_vggt.VGGT = object
    pose_enc = types.ModuleType("vggt.utils.pose_enc")
    pose_enc.pose_encoding_to_extri_intri = lambda *a, **k: None
    geometry = types.ModuleType("vggt.utils.geometry")
    geometry.unproject_depth_map_to_point_map = lambda *a, **k: None
    for name, mod in [
        ("vggt", types.ModuleType("vggt")),
        ("vggt.models", types.ModuleType("vggt.models")),
        ("vggt.models.vggt", models_vggt),
        ("vggt.utils", types.ModuleType("vggt.utils")),
        ("vggt.utils.pose_enc", pose_enc),
        ("vggt.utils.geometry", geometry),
    ]:
        sys.modules[name] = mod


_install_vggt_stubs()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from vqasynth.scene_fusion import (  # noqa: E402
    SpatialSceneConstructor,
    compute_object_depth_confidence,
    preprocess_images,
)


def _region_conf_map(height, width, hi=0.95, lo=0.05):
    """Synthetic conf map: left 40% high confidence, right 40% low."""
    conf = np.full((height, width), 0.5)
    conf[:, : int(width * 0.4)] = hi
    conf[:, int(width * 0.6):] = lo
    return conf


# ---------------------------------------------------------------------------
# Region reflection: high-conf mask -> high scalar, low-conf mask -> low scalar
# ---------------------------------------------------------------------------

def test_object_confidence_reflects_mask_region():
    conf = _region_conf_map(64, 64)

    high_mask = np.zeros((64, 64), dtype=bool)
    high_mask[:, :16] = True
    low_mask = np.zeros((64, 64), dtype=bool)
    low_mask[:, 48:] = True

    high_conf = compute_object_depth_confidence(conf, high_mask)
    low_conf = compute_object_depth_confidence(conf, low_mask)

    assert high_conf == pytest.approx(0.95)
    assert low_conf == pytest.approx(0.05)
    assert high_conf > low_conf


def test_object_confidence_averages_within_mask():
    """A mask straddling both regions gets the mean of its pixels, not the map max."""
    conf = _region_conf_map(8, 8)
    straddling = np.zeros((8, 8), dtype=bool)
    straddling[:, :2] = True  # 0.95 side
    straddling[:, 6:] = True  # 0.05 side

    # 16 pixels at 0.95 + 16 pixels at 0.05 => exactly 0.5
    assert compute_object_depth_confidence(conf, straddling) == pytest.approx(0.5)


def test_accepts_torch_mask_like_run_does():
    """run() passes preprocessed torch bool masks — the helper must handle those."""
    conf = _region_conf_map(16, 16)
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[:, :4] = True

    assert compute_object_depth_confidence(conf, mask) == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Monotonic degradation: noise that lowers the conf map lowers the scalar
# ---------------------------------------------------------------------------

def test_confidence_degrades_monotonically_with_multiplicative_decay():
    conf = _region_conf_map(64, 64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[:, :16] = True

    scalars = [
        compute_object_depth_confidence(conf * alpha, mask)
        for alpha in (1.0, 0.75, 0.5, 0.25)
    ]
    assert all(a > b for a, b in zip(scalars, scalars[1:]))
    assert scalars == sorted(scalars, reverse=True)


def test_confidence_degrades_monotonically_with_injected_noise():
    """Subtract |N(0, sigma)| from the conf map; growing sigma must not raise confidence.

    The same seed is reused for every sigma, so each noise draw is the previous
    one scaled up — the degradation is strictly monotone by construction.
    """
    conf = _region_conf_map(64, 64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[:, :16] = True

    scalars = []
    for sigma in (0.0, 0.1, 0.25, 0.5, 1.0):
        rng = np.random.default_rng(42)
        noise = (
            np.abs(rng.normal(0.0, sigma, size=conf.shape))
            if sigma > 0
            else np.zeros_like(conf)
        )
        degraded = np.clip(conf - noise, 0.0, 1.0)
        scalars.append(compute_object_depth_confidence(degraded, mask))

    assert all(a > b for a, b in zip(scalars, scalars[1:]))
    # The clean map's confidence is the ceiling.
    assert scalars[0] == pytest.approx(0.95)
    assert scalars[-1] < 0.5


# ---------------------------------------------------------------------------
# Grid alignment: masks preprocessed exactly like run() does, via the real
# preprocess_images, must land on the same (H, W) grid as the conf map
# ---------------------------------------------------------------------------

def test_masks_align_with_conf_grid_through_real_preprocessing():
    src_h, src_w = 224, 336

    # Two synthetic object masks in the source-image layout run() receives:
    # object A on the left third, object B on the right third (255 = in-object).
    masks_np = np.zeros((2, src_h, src_w), dtype=np.uint8)
    masks_np[0, :, : src_w // 3] = 255
    masks_np[1, :, -src_w // 3:] = 255

    # The exact alignment lines from SpatialSceneConstructor.run.
    masks = torch.tensor(masks_np) / 255.0
    masks = preprocess_images(masks.unsqueeze(0), mode="crop").squeeze(0)
    masks = masks >= 0.5
    assert masks.shape[0] == 2

    # The conf map lives on the grid preprocess_images produces for the image
    # itself (same mode), exactly as create_point_cloud_from_model derives it.
    preprocessed = preprocess_images(
        [Image.new("RGB", (src_w, src_h))], mode="crop"
    )
    grid_h, grid_w = preprocessed.shape[-2:]
    assert masks.shape[1:] == (grid_h, grid_w)

    conf = _region_conf_map(grid_h, grid_w)
    conf_a = compute_object_depth_confidence(conf, masks[0])
    conf_b = compute_object_depth_confidence(conf, masks[1])

    # Object A sat in the high-confidence region, object B in the low one.
    assert conf_a > 0.8
    assert conf_b < 0.2


# ---------------------------------------------------------------------------
# Plumbing: run()/apply_transform() surface the per-object confidence column.
# Following tests/test_vggt_speedups.py, a stub constructor stands in for the
# VGGT-loaded original — same signatures, synthetic scene outputs.
# ---------------------------------------------------------------------------

class _StubConstructor(SpatialSceneConstructor):
    """SpatialSceneConstructor with the VGGT call replaced by synthetic output.

    Skips model loading entirely (no ``super().__init__``"); the overridden
    method matches the real signature, including the return_depth_conf opt-in.
    """

    def __init__(self, conf_map):
        self._conf_map = conf_map

    def create_point_cloud_from_model(self, pil_image, mode="crop", *, return_depth_conf=False):
        preprocessed = preprocess_images([pil_image], mode=mode)
        grid_h, grid_w = preprocessed.shape[-2:]

        rng = np.random.default_rng(7)
        coords = rng.uniform(0.0, 1.0, size=(grid_h * grid_w, 3))
        colors = np.zeros_like(coords)

        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        depth_map_np = np.full((grid_h, grid_w), 2.0)
        if return_depth_conf:
            return pcd, depth_map_np, 500.0, self._conf_map
        return pcd, depth_map_np, 500.0


def _stub_scene(tmp_path):
    """Build a stub constructor + two masks: one solid, one empty (skipped)."""
    src_h, src_w = 224, 336
    masks_np = np.zeros((2, src_h, src_w), dtype=np.uint8)
    masks_np[0, :, : src_w // 3] = 255  # solid, over the high-conf region
    masks_np[1] = 0  # empty -> run() skips it with a warning

    preprocessed = preprocess_images([Image.new("RGB", (src_w, src_h))], mode="crop")
    grid_h, grid_w = preprocessed.shape[-2:]
    constructor = _StubConstructor(_region_conf_map(grid_h, grid_w))

    image = Image.new("RGB", (src_w, src_h))
    return constructor, image, masks_np


def test_run_surfaces_aligned_object_confidences(tmp_path):
    constructor, image, masks_np = _stub_scene(tmp_path)

    filepaths, canonicalized, depth_map, focal, confidences = constructor.run(
        "scene", image, masks_np, str(tmp_path), return_depth_conf=True
    )

    # The empty mask was skipped: exactly one cloud saved, one confidence,
    # aligned index-for-index.
    assert len(filepaths) == 1
    assert len(confidences) == len(filepaths)
    # The surviving mask sat over the high-confidence region.
    assert confidences[0] > 0.8

    # Legacy call shape (examples/detection_3d_example.py) still unpacks 4.
    legacy = constructor.run("scene2", image, masks_np, str(tmp_path))
    assert len(legacy) == 4


def test_apply_transform_emits_depth_confidence_column(tmp_path):
    constructor, image, masks_np = _stub_scene(tmp_path)

    example = {"image": image, "masks": masks_np}
    result = constructor.apply_transform(
        example, idx=0, output_dir=str(tmp_path), images="image"
    )

    # apply_transform swallows exceptions into None columns; assert real values.
    assert result["pointclouds"] is not None
    assert result["depth_confidence"] is not None
    # Single-example wrap: one per-image entry, holding the per-object list
    # aligned with the per-object pointcloud list.
    assert len(result["depth_confidence"]) == 1
    assert len(result["depth_confidence"][0]) == len(result["pointclouds"][0]) == 1
    assert result["depth_confidence"][0][0] > 0.8


# ---------------------------------------------------------------------------
# Defensive contract
# ---------------------------------------------------------------------------

def test_empty_mask_raises():
    conf = _region_conf_map(8, 8)
    with pytest.raises(ValueError, match="empty mask"):
        compute_object_depth_confidence(conf, np.zeros((8, 8), dtype=bool))


def test_shape_mismatch_raises():
    conf = _region_conf_map(8, 8)
    with pytest.raises(ValueError, match="does not match"):
        compute_object_depth_confidence(conf, np.ones((4, 4), dtype=bool))
