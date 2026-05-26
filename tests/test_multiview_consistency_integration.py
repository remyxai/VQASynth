"""Tests for vqasynth.multiview_consistency_integration."""

import numpy as np
import pytest

from vqasynth.multiview_consistency_integration import (
    MultiViewConsistencyConfig,
    MultiViewConsistencyEvaluator,
    aggregate_pair_scores,
    count_geometric_inliers,
    dense_support_ratio,
    extract_orb_features,
    match_descriptors,
    pairwise_consistency_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic textured images that ORB can lock onto.
# ---------------------------------------------------------------------------


def _textured_image(seed: int, size: int = 256) -> np.ndarray:
    """Random but reproducible textured RGB image with enough corners
    for ORB to find ~hundreds of keypoints."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    # Stamp a few high-contrast rectangles for stable corners
    for _ in range(20):
        y0 = int(rng.integers(0, size - 32))
        x0 = int(rng.integers(0, size - 32))
        h = int(rng.integers(8, 32))
        w = int(rng.integers(8, 32))
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        img[y0 : y0 + h, x0 : x0 + w] = color
    return img


def _shifted(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate an image by (dx, dy) px, filling exposed regions with zero.
    A pure translation keeps point correspondences exactly recoverable."""
    h, w = image.shape[:2]
    out = np.zeros_like(image)
    src_y0 = max(0, -dy)
    src_x0 = max(0, -dx)
    dst_y0 = max(0, dy)
    dst_x0 = max(0, dx)
    copy_h = h - abs(dy)
    copy_w = w - abs(dx)
    if copy_h <= 0 or copy_w <= 0:
        return out
    out[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = image[
        src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
    ]
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_match_paper_settings():
    cfg = MultiViewConsistencyConfig()
    assert cfg.orb_n_features == 2048
    assert 0.0 < cfg.lowe_ratio < 1.0
    assert cfg.ransac_pixel_threshold > 0
    assert 0.0 < cfg.ransac_confidence < 1.0
    assert cfg.min_inlier_matches >= 8  # below this, F-matrix RANSAC is unreliable
    assert cfg.pair_aggregation in {"mean", "min", "median"}
    assert cfg.backbone in {"vggt", "mast3r", "dust3r", "fast3r"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_extract_orb_features_finds_keypoints_on_textured_image():
    img = _textured_image(seed=0)
    kps, desc = extract_orb_features(img, n_features=1024)
    assert len(kps) > 50
    assert desc is not None
    assert desc.shape[0] == len(kps)


def test_extract_orb_features_accepts_grayscale_and_rgba():
    img_rgb = _textured_image(seed=1)
    img_gray = img_rgb.mean(axis=-1).astype(np.uint8)
    img_rgba = np.dstack([img_rgb, np.full(img_rgb.shape[:2], 255, np.uint8)])

    for variant in (img_rgb, img_gray, img_rgba):
        kps, desc = extract_orb_features(variant, n_features=256)
        assert len(kps) > 0
        assert desc is not None


def test_match_descriptors_returns_empty_on_missing_input():
    assert match_descriptors(None, None) == []
    assert match_descriptors(np.zeros((0, 32), np.uint8), None) == []


def test_match_descriptors_matches_identical_images():
    img = _textured_image(seed=2)
    kps_a, desc_a = extract_orb_features(img, n_features=512)
    kps_b, desc_b = extract_orb_features(img.copy(), n_features=512)
    matches = match_descriptors(desc_a, desc_b, ratio=0.75)
    # Most descriptors should pass the ratio test against themselves
    assert len(matches) > 20


def test_count_geometric_inliers_high_for_translated_view():
    img = _textured_image(seed=3)
    shifted = _shifted(img, dx=12, dy=7)

    kps_a, desc_a = extract_orb_features(img, n_features=1024)
    kps_b, desc_b = extract_orb_features(shifted, n_features=1024)
    matches = match_descriptors(desc_a, desc_b, ratio=0.85)

    inliers = count_geometric_inliers(kps_a, kps_b, matches)
    assert inliers >= 15


def test_count_geometric_inliers_low_for_unrelated_images():
    img_a = _textured_image(seed=4)
    img_b = _textured_image(seed=999)  # entirely different random seed

    kps_a, desc_a = extract_orb_features(img_a, n_features=1024)
    kps_b, desc_b = extract_orb_features(img_b, n_features=1024)
    matches = match_descriptors(desc_a, desc_b, ratio=0.75)

    inliers = count_geometric_inliers(kps_a, kps_b, matches)
    # The paper's whole point: unrelated views should NOT yield meaningful
    # geometric support, even when the backbone hallucinates dense output.
    assert inliers < 15


def test_dense_support_ratio_bounds():
    assert dense_support_ratio(0, 100) == 0.0
    assert dense_support_ratio(50, 100) == 0.5
    assert dense_support_ratio(100, 100) == 1.0
    # Clamped to [0, 1] even if caller passes a degenerate count
    assert dense_support_ratio(150, 100) == 1.0
    assert dense_support_ratio(10, 0) == 0.0


@pytest.mark.parametrize(
    "agg,expected",
    [("mean", 2.0), ("min", 1.0), ("median", 2.0)],
)
def test_aggregate_pair_scores(agg, expected):
    assert aggregate_pair_scores([1.0, 2.0, 3.0], aggregation=agg) == expected


def test_aggregate_pair_scores_empty():
    assert aggregate_pair_scores([]) == 0.0


def test_aggregate_pair_scores_unknown_raises():
    with pytest.raises(ValueError):
        aggregate_pair_scores([1.0], aggregation="bogus")


# ---------------------------------------------------------------------------
# Pairwise metric + evaluator scaffold
# ---------------------------------------------------------------------------


def test_pairwise_metrics_single_view_yields_zero_pairs():
    img = _textured_image(seed=5)
    out = pairwise_consistency_metrics([img])
    assert out["n_views"] == 1
    assert out["pairs"] == []
    assert out["registered"] is False


def test_pairwise_metrics_consistent_views_register():
    img = _textured_image(seed=6)
    views = [img, _shifted(img, dx=8, dy=4), _shifted(img, dx=-6, dy=10)]
    cfg = MultiViewConsistencyConfig(lowe_ratio=0.85)
    out = pairwise_consistency_metrics(views, config=cfg)

    assert out["n_views"] == 3
    assert len(out["pairs"]) == 3  # C(3, 2)
    assert out["mean_inliers"] > 15
    assert out["registered"] is True


def test_pairwise_metrics_unrelated_views_fail_to_register():
    views = [_textured_image(seed=s) for s in (10, 200, 3000)]
    out = pairwise_consistency_metrics(views)
    assert out["n_views"] == 3
    assert out["registered"] is False


def test_evaluator_smoke():
    evaluator = MultiViewConsistencyEvaluator()
    img = _textured_image(seed=7)
    result = evaluator.evaluate([img, _shifted(img, dx=5, dy=0)])
    assert set(result.keys()) >= {
        "n_views",
        "pairs",
        "mean_inliers",
        "mean_dense_support",
        "aggregated_score",
        "registered",
    }


def test_evaluator_external_paths_are_stubs():
    evaluator = MultiViewConsistencyEvaluator()
    img = _textured_image(seed=8)
    with pytest.raises(NotImplementedError):
        evaluator.evaluate_with_colmap([img, img], workdir="/tmp/does-not-exist")
    with pytest.raises(NotImplementedError):
        evaluator.evaluate_with_met3r([img, img])
