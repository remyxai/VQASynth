"""CPU acceptance tests for metric scale calibration in vqasynth.prompts.

Builds a synthetic known-geometry scene — a pinhole camera with known
intrinsics and two object patches at known 3D positions a known distance
apart — then drives the calibrated distance path end to end:
``unproject_depth_map`` (intrinsics) -> per-object clouds ->
``PromptGenerator`` distance/size predicates -> parsed answers.

Acceptance bar (per the design brief): the calibrated path reports
ground-truth metric values within a tight tolerance, and the retired ``* 10``
fudge does not. Also checks size correctness (width/height/volume) on a known
box, per-scene scale recovery via a DepthPro-style metric anchor
(``calibrate_metric_scale``), and the [A]/[B]/[X] template convention shared
with the pre-existing vqasynth.prompt_templates vocabulary — so this suite is
a cross-module integration check, not a self-test of the new code alone.

No VGGT, no CUDA, no model downloads (mirrors the philosophy of
tests/test_detection_3d.py and tests/test_correspondence.py). Real end-to-end
validation on VGGT/DepthPro outputs belongs on a GPU host.
"""
from __future__ import annotations

import math
import random
import re

import numpy as np
import open3d as o3d
import pytest

from vqasynth import prompt_templates
from vqasynth.prompts import (
    PromptGenerator,
    calibrate_metric_scale,
    depth_map_to_meters,
    unproject_depth_map,
)

# ---------------------------------------------------------------------------
# Synthetic known-geometry scene (pure numpy, exact by construction)
# ---------------------------------------------------------------------------
# Pinhole grid chosen so every patch edge lands exactly on a pixel: with
# focal 100 px and depth 5 m, one pixel step is 0.05 m on the image plane.
FOCAL_PX = 100.0
DEPTH_M = 5.0
WIDTH_PX, HEIGHT_PX = 161, 121

# patch A spans x in [-0.6, -0.2] m, y in [ 0.0,  0.4] m  -> AABB center (-0.4, +0.2, 5)
# patch B spans x in [ 0.2,  0.6] m, y in [-0.4,  0.0] m  -> AABB center (+0.4, -0.2, 5)
GT_SIZE_M = 0.4        # each patch's x and y extent
GT_VERTICAL_M = 0.4    # |A_center_y - B_center_y|
GT_HORIZONTAL_M = 0.8  # xz-plane distance between AABB centers

_UNITS_TO_METERS = {
    "meters": 1.0,
    "centimeters": 0.01,
    "inches": 0.0254,
    "feet": 0.3048,
}
_ANSWER_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(meters|centimeters|inches|feet)"
)


def _scene_grid_meters():
    """x/y position (meters) of every pixel on the DEPTH_M plane."""
    us, vs = np.meshgrid(np.arange(WIDTH_PX), np.arange(HEIGHT_PX))
    cx, cy = (WIDTH_PX - 1) / 2.0, (HEIGHT_PX - 1) / 2.0
    return (us - cx) / FOCAL_PX * DEPTH_M, (vs - cy) / FOCAL_PX * DEPTH_M


def _patch_masks():
    x_m, y_m = _scene_grid_meters()
    eps = 1e-9
    mask_a = (
        (x_m >= -0.6 - eps) & (x_m <= -0.2 + eps)
        & (y_m >= -eps) & (y_m <= 0.4 + eps)
    )
    mask_b = (
        (x_m >= 0.2 - eps) & (x_m <= 0.6 + eps)
        & (y_m >= -0.4 - eps) & (y_m <= eps)
    )
    assert mask_a.sum() == 81 and mask_b.sum() == 81  # 9x9 pixels each
    return mask_a, mask_b


def _pairwise_ground_truth_meters():
    """Mean nearest-neighbor distance A -> B, from the grid geometry alone.

    For an A point at (x, y): every B x is >= 0.2 > x, so the nearest B x is
    the left edge 0.2 (dx = 0.2 - x); every B y is <= 0 <= y, so the nearest
    B y is 0 (dy = y). Exact, independent of the code under test.
    """
    x_m, y_m = _scene_grid_meters()
    mask_a, _ = _patch_masks()
    dx = 0.2 - x_m[mask_a]
    dy = y_m[mask_a]
    return float(np.mean(np.hypot(dx, dy)))


def _cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


@pytest.fixture
def scene():
    """Metric depth map + per-patch clouds for the two-patch scene."""
    depth_map = np.full((HEIGHT_PX, WIDTH_PX), DEPTH_M, dtype=np.float64)
    mask_a, mask_b = _patch_masks()
    points = unproject_depth_map(depth_map, FOCAL_PX)
    cloud_a = _cloud(points[mask_a.ravel()])
    cloud_b = _cloud(points[mask_b.ravel()])
    return depth_map, cloud_a, cloud_b


def answer_to_meters(result):
    """Parse the first `<value> <unit>` pair in a result string -> (meters, unit)."""
    assert "[X]" not in result and "[A]" not in result, (
        f"unsubstituted placeholder in {result!r}"
    )
    match = _ANSWER_RE.search(result)
    assert match, f"no distance + unit found in {result!r}"
    unit = match.group(2)
    return float(match.group(1)) * _UNITS_TO_METERS[unit], unit


def assert_metric_close(result, gt_meters, rel_tol=0.05):
    """A formatted answer matches GT: within rel_tol, or within the rounding
    granularity of whichever unit human_like_distance happened to pick
    (it rounds to 0-2 decimals, so whole feet can be off by half a foot)."""
    got_meters, unit = answer_to_meters(result)
    assert abs(got_meters - gt_meters) <= max(
        rel_tol * gt_meters, 0.5 * _UNITS_TO_METERS[unit]
    ), f"{result!r}: parsed {got_meters:.4f} m, expected {gt_meters:.4f} m"


# ---------------------------------------------------------------------------
# Intrinsics unprojection (the metric-accurate path)
# ---------------------------------------------------------------------------

def test_unproject_depth_map_is_exact():
    depth_map = np.full((HEIGHT_PX, WIDTH_PX), DEPTH_M, dtype=np.float64)
    points = unproject_depth_map(depth_map, FOCAL_PX)

    assert points.shape == (HEIGHT_PX * WIDTH_PX, 3)
    # OpenCV pinhole convention: z holds the depth, x/y follow the pixel grid.
    assert np.allclose(points[:, 2], DEPTH_M)
    x_m, y_m = _scene_grid_meters()
    assert np.allclose(points[:, 0], x_m.ravel())
    assert np.allclose(points[:, 1], y_m.ravel())


def test_unproject_depth_map_drops_invalid_pixels():
    depth_map = np.full((6, 6), 2.0)
    depth_map[0, 0] = np.nan
    depth_map[1, 1] = 0.0
    depth_map[2, 2] = -1.0
    assert unproject_depth_map(depth_map, 50.0).shape == (33, 3)


def test_unproject_depth_map_rejects_bad_intrinsics():
    with pytest.raises(ValueError):
        unproject_depth_map(np.ones((4, 4)), 0.0)
    with pytest.raises(ValueError):
        unproject_depth_map(np.ones((4, 4, 3)), 100.0)  # not a 2D map


def test_cloud_extents_are_metric(scene):
    """The per-object clouds span exactly their known metric footprints."""
    _, cloud_a, cloud_b = scene
    extent_a = cloud_a.get_axis_aligned_bounding_box().get_extent()
    extent_b = cloud_b.get_axis_aligned_bounding_box().get_extent()
    assert extent_a[0] == pytest.approx(GT_SIZE_M)
    assert extent_a[1] == pytest.approx(GT_SIZE_M)
    assert extent_b[0] == pytest.approx(GT_SIZE_M)
    assert extent_b[1] == pytest.approx(GT_SIZE_M)
    center_a = cloud_a.get_axis_aligned_bounding_box().get_center()
    center_b = cloud_b.get_axis_aligned_bounding_box().get_center()
    assert center_a[0] == pytest.approx(-0.4) and center_a[1] == pytest.approx(0.2)
    assert center_b[0] == pytest.approx(0.4) and center_b[1] == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# Acceptance bar: predicates report ground-truth meters; the x10 path does not
# ---------------------------------------------------------------------------

def test_distance_predicates_match_ground_truth(scene):
    random.seed(0)
    _, cloud_a, cloud_b = scene
    gen = PromptGenerator()  # metric cloud -> scale 1.0 is the correct calibration
    A, B = ("red cube", cloud_a), ("blue ball", cloud_b)

    for _ in range(10):  # every template + unit choice must stay metric
        assert_metric_close(gen.vertical_distance_data(A, B), GT_VERTICAL_M)
        assert_metric_close(gen.horizontal_distance_data(A, B), GT_HORIZONTAL_M)
        assert_metric_close(gen.width_data(A), GT_SIZE_M)
        assert_metric_close(gen.height_data(A), GT_SIZE_M)
        assert_metric_close(gen.height_data(B), GT_SIZE_M)

    # The pairwise predicate (compute_point_cloud_distance().mean()) that feeds
    # the 5x distance QA block in evaluate_predicates_on_pairs.
    gt_pairwise = _pairwise_ground_truth_meters()
    raw = float(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b)).mean())
    assert raw == pytest.approx(gt_pairwise, abs=1e-9)
    for _ in range(10):
        assert_metric_close(gen.human_like_distance(gen._metric(raw)), gt_pairwise)


def test_old_x10_fudge_fails_the_same_bar(scene):
    """The retired behavior (raw distance * 10, then humanize) must NOT pass:
    that is the whole point of replacing it with intrinsics calibration."""
    random.seed(0)
    _, cloud_a, cloud_b = scene
    gen = PromptGenerator()
    raw = float(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b)).mean())
    gt_pairwise = _pairwise_ground_truth_meters()

    old_style = gen.human_like_distance(10 * raw)
    got_meters, unit = answer_to_meters(old_style)
    assert abs(got_meters - gt_pairwise) > 0.5 * gt_pairwise, (
        f"x10 path unexpectedly close to ground truth: {old_style!r}"
    )
    # Sanity: the x10 value is ~10x off, not just outside tolerance.
    assert got_meters > 5 * gt_pairwise


def test_evaluate_predicates_on_pairs_dispatch_is_metric(scene):
    """End-to-end through the real dispatch used by run(): every unit-bearing
    answer in the emitted prompts is metric-correct for this scene."""
    random.seed(1)
    _, cloud_a, cloud_b = scene
    gen = PromptGenerator()
    pairs = [(("red cube", cloud_a), ("blue ball", cloud_b))]
    results = gen.evaluate_predicates_on_pairs(pairs, is_canonicalized=True)

    gt_values = [
        GT_VERTICAL_M,
        GT_HORIZONTAL_M,
        GT_SIZE_M,
        _pairwise_ground_truth_meters(),
    ]
    unit_results = [r for r in results if _ANSWER_RE.search(r)]
    assert len(unit_results) >= 6  # the distance/size predicates all survived
    for result in unit_results:
        got_meters, _ = answer_to_meters(result)
        assert any(
            abs(got_meters - gt) <= 0.05 * gt
            or any(
                abs(got_meters - gt) <= 0.5 * _UNITS_TO_METERS[unit]
                for unit in _UNITS_TO_METERS
            )
            for gt in gt_values
        ), f"{result!r}: parsed {got_meters:.4f} m matches no ground truth {gt_values}"


# ---------------------------------------------------------------------------
# Known-box size check (width / height / volume)
# ---------------------------------------------------------------------------

def _box_cloud(size_m, step_m, scale=1.0):
    """Grid over the 6 faces of a centered cube (scale != 1 emulates a
    relative/up-to-scale cloud)."""
    half = size_m / 2.0
    n = max(2, int(round(size_m / step_m)) + 1)
    t = np.linspace(-half, half, n)
    a, b = np.meshgrid(t, t)
    faces = [
        np.stack([a.ravel(), b.ravel(), np.full(a.size, half)], axis=1),
        np.stack([a.ravel(), b.ravel(), np.full(a.size, -half)], axis=1),
        np.stack([a.ravel(), np.full(a.size, half), b.ravel()], axis=1),
        np.stack([a.ravel(), np.full(a.size, -half), b.ravel()], axis=1),
        np.stack([np.full(a.size, half), a.ravel(), b.ravel()], axis=1),
        np.stack([np.full(a.size, -half), a.ravel(), b.ravel()], axis=1),
    ]
    return _cloud(np.vstack(faces) * scale)


def test_known_box_size_and_volume():
    random.seed(2)
    size, volume = 0.4, 0.4 ** 3

    metric_box = _box_cloud(size, step_m=0.05)
    extent = metric_box.get_axis_aligned_bounding_box().get_extent()
    assert extent[0] == pytest.approx(size)
    assert extent[1] == pytest.approx(size)  # height_data reads the Y axis
    assert extent[2] == pytest.approx(size)
    # The exact primitive big/small predicates compare (AABB volume).
    assert extent.prod() == pytest.approx(volume)

    # A cloud in units 8x too small (a relative-depth scene): uncalibrated it
    # understates every linear size 8x and the volume 512x; threaded through
    # the per-scene metric scale it reports the true box.
    relative_box = _box_cloud(size, step_m=0.05, scale=1.0 / 8.0)
    rel_extent = relative_box.get_axis_aligned_bounding_box().get_extent()
    assert rel_extent.prod() == pytest.approx(volume / 512)

    gen = PromptGenerator(metric_scale=8.0)
    A = ("crate", relative_box)
    for _ in range(10):
        assert_metric_close(gen.width_data(A), size)
        assert_metric_close(gen.height_data(A), size)


# ---------------------------------------------------------------------------
# Per-scene scale recovery from a metric anchor (DepthPro-style)
# ---------------------------------------------------------------------------

def test_calibrate_metric_scale_recovers_scale():
    metric = np.full((16, 16), 6.0)
    true_scale = 7.5
    relative = metric / true_scale
    assert calibrate_metric_scale(relative, metric) == pytest.approx(
        true_scale, rel=1e-12
    )

    # Robust to outlier pixels (both maps have local errors): 10% of pixels
    # corrupted by +/-30% still leaves the median ratio intact.
    rng = np.random.default_rng(3)
    noisy_relative = relative.copy()
    corrupt = rng.random(noisy_relative.shape) < 0.1
    noisy_relative[corrupt] *= rng.choice([0.7, 1.3], size=corrupt.sum())
    assert calibrate_metric_scale(noisy_relative, metric) == pytest.approx(
        true_scale, rel=0.01
    )

    with pytest.raises(ValueError):
        calibrate_metric_scale(np.ones((4, 4)), np.ones((5, 5)))  # grid mismatch
    with pytest.raises(ValueError):
        calibrate_metric_scale(np.zeros((4, 4)), np.zeros((4, 4)))  # nothing valid


def test_depth_map_to_meters_decodes_uint16_mm():
    # vqasynth.depth stores DepthPro output as millimeter uint16 (PIL I;16).
    from PIL import Image

    metric_m = np.array([[0.5, 1.5], [2.0, 3.0]])
    mm_uint16 = (metric_m * 1000).astype(np.uint16)
    assert np.allclose(depth_map_to_meters(mm_uint16), metric_m)
    # PIL round-trip (mode inferred from the uint16 dtype — explicit "I;16"
    # is rejected by newer Pillow but infers to the same 16-bit image).
    depth_pil = Image.fromarray(mm_uint16)
    assert np.asarray(depth_pil).dtype == np.uint16
    assert np.allclose(depth_map_to_meters(depth_pil), metric_m)
    assert np.allclose(depth_map_to_meters(metric_m), metric_m)  # float passthrough


def test_relative_scene_reports_metric_through_anchor(scene):
    """Full story: cloud unprojected from a VGGT-style *relative* depth map
    (arbitrary scale), calibrated against a DepthPro-style metric anchor ->
    predicates report the true metric distances."""
    random.seed(4)
    depth_map, _, _ = scene
    true_scale = 7.5
    relative_depth = depth_map / true_scale  # VGGT-style up-to-scale map
    metric_anchor = depth_map.copy()  # DepthPro-style metric map (meters)

    scale = calibrate_metric_scale(relative_depth, metric_anchor)
    assert scale == pytest.approx(true_scale, rel=1e-12)

    mask_a, mask_b = _patch_masks()
    points = unproject_depth_map(relative_depth, FOCAL_PX)
    cloud_a = _cloud(points[mask_a.ravel()])
    cloud_b = _cloud(points[mask_b.ravel()])

    gen = PromptGenerator(metric_scale=scale)
    A, B = ("red cube", cloud_a), ("blue ball", cloud_b)
    for _ in range(10):
        assert_metric_close(gen.vertical_distance_data(A, B), GT_VERTICAL_M)
        assert_metric_close(gen.horizontal_distance_data(A, B), GT_HORIZONTAL_M)
        assert_metric_close(gen.width_data(A), GT_SIZE_M)

    # Same predicates WITHOUT the calibration report the relative units
    # (true_scale x too small) — i.e. the anchor, not luck, does the work.
    uncalibrated = PromptGenerator()
    got_meters, _ = answer_to_meters(uncalibrated.vertical_distance_data(A, B))
    assert got_meters == pytest.approx(GT_VERTICAL_M / true_scale, rel=0.05)


# ---------------------------------------------------------------------------
# human_like_distance formatting + metric-scale plumbing
# ---------------------------------------------------------------------------

def test_human_like_distance_humanizes_without_fudge():
    random.seed(5)
    gen = PromptGenerator()
    for _ in range(100):
        assert_metric_close(gen.human_like_distance(0.85), 0.85)
        assert_metric_close(gen.human_like_distance(2.5), 2.5)

    # Every unit band is reachable and converts back exactly.
    seen_units = set()
    for value in (0.05, 0.85, 2.5, 12.0):
        for _ in range(200):
            _, unit = answer_to_meters(gen.human_like_distance(value))
            seen_units.add(unit)
    assert {"meters", "centimeters", "inches", "feet"} <= seen_units

    assert "nan" in gen.human_like_distance(float("nan"))
    assert gen.human_like_distance(0.0).startswith("0.0")


def test_metric_scale_resolution():
    gen = PromptGenerator(metric_scale=2.0)
    assert gen._resolve_metric_scale({}) == 2.0
    assert gen._resolve_metric_scale({"metric_scale": [7.5]}) == 7.5  # per-row list
    assert gen._resolve_metric_scale({"metric_scale": 3.0}) == 3.0
    with pytest.raises(ValueError):
        gen._resolve_metric_scale({"metric_scale": [1.0, 2.0]})  # one per example
    with pytest.raises(ValueError):
        gen._resolve_metric_scale({"metric_scale": [0.0]})

    # Column resolution feeds run() without cross-example leakage: an example
    # without the column falls back to the generator default, not the previous
    # example's scale.
    gen._resolve_metric_scale({"metric_scale": [7.5]})
    assert gen._resolve_metric_scale({}) == 2.0

    with pytest.raises(ValueError):
        PromptGenerator(metric_scale=-1.0)


# ---------------------------------------------------------------------------
# Cross-module convention check against the pre-existing template vocabulary
# ---------------------------------------------------------------------------

def test_template_vocabulary_keeps_placeholder_convention():
    answer_groups = [
        prompt_templates.distance_template_answers,
        prompt_templates.vertical_distance_answers,
        prompt_templates.horizontal_distance_answers,
        prompt_templates.width_answers,
        prompt_templates.height_answers,
    ]
    for group in answer_groups:
        assert any("[X]" in answer for answer in group), (
            "calibrated values substitute into [X]; a group without it would "
            "emit answers with no distance"
        )
    assert all("[A]" in q for q in prompt_templates.distance_template_questions)
