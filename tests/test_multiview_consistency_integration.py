"""Tests for vqasynth.multiview_consistency_integration."""

import math

import numpy as np
import pytest

from vqasynth.multiview_consistency_integration import (
    ConsistencySignals,
    MultiviewConsistencyConfig,
    MultiviewConsistencyEvaluator,
    dense_support_ratio,
    evaluate_vqasynth_pointcloud,
    failure_aware_aggregate,
    is_likely_hallucination,
    normalize_match_count,
    pointcloud_validity_ratio,
    signals_for_failed_reconstruction,
)


# ---------------------------------------------------------------------------
# normalize_match_count
# ---------------------------------------------------------------------------


def test_normalize_match_count_zero_and_none():
    assert normalize_match_count(None, 100) == 0.0
    assert normalize_match_count(0, 100) == 0.0
    assert normalize_match_count(-5, 100) == 0.0


def test_normalize_match_count_linear_below_target():
    assert normalize_match_count(50, 100) == pytest.approx(0.5)
    assert normalize_match_count(25, 100) == pytest.approx(0.25)


def test_normalize_match_count_saturates_at_target():
    assert normalize_match_count(100, 100) == 1.0
    assert normalize_match_count(1_000_000, 100) == 1.0


def test_normalize_match_count_rejects_bad_target():
    with pytest.raises(ValueError):
        normalize_match_count(10, 0)
    with pytest.raises(ValueError):
        normalize_match_count(10, -1)


# ---------------------------------------------------------------------------
# dense_support_ratio
# ---------------------------------------------------------------------------


def test_dense_support_ratio_all_supported():
    vis = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=bool)
    # min_views=2 => all 3 points qualify
    assert dense_support_ratio(vis, min_views=2) == 1.0


def test_dense_support_ratio_none_supported():
    vis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    assert dense_support_ratio(vis, min_views=2) == 0.0


def test_dense_support_ratio_mixed():
    vis = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool)
    # min_views=2 => points 0 and 2 qualify => 2/4
    assert dense_support_ratio(vis, min_views=2) == 0.5


def test_dense_support_ratio_empty():
    assert dense_support_ratio(np.zeros((0, 3), dtype=bool)) == 0.0


def test_dense_support_ratio_rejects_bad_min_views():
    vis = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        dense_support_ratio(vis, min_views=0)


def test_dense_support_ratio_rejects_non_2d():
    with pytest.raises(ValueError):
        dense_support_ratio(np.array([1, 0, 1], dtype=bool))


# ---------------------------------------------------------------------------
# pointcloud_validity_ratio
# ---------------------------------------------------------------------------


def test_pointcloud_validity_ratio_all_finite():
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    assert pointcloud_validity_ratio(pts) == 1.0


def test_pointcloud_validity_ratio_with_nan_and_inf():
    pts = np.array(
        [
            [0.0, 0.0, 1.0],
            [np.nan, 0.0, 0.0],
            [0.0, np.inf, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    # 2 valid out of 4
    assert pointcloud_validity_ratio(pts) == pytest.approx(0.5)


def test_pointcloud_validity_ratio_empty():
    assert pointcloud_validity_ratio(np.zeros((0, 3))) == 0.0


def test_pointcloud_validity_ratio_rejects_wrong_shape():
    with pytest.raises(ValueError):
        pointcloud_validity_ratio(np.zeros((4, 2)))


# ---------------------------------------------------------------------------
# failure_aware_aggregate
# ---------------------------------------------------------------------------


def test_aggregate_failed_reconstruction_short_circuits_to_zero():
    cfg = MultiviewConsistencyConfig()
    sig = ConsistencySignals(
        num_matches=999, inlier_ratio=1.0, dense_support_ratio=1.0,
        reconstruction_failed=True,
    )
    assert failure_aware_aggregate(sig, cfg) == 0.0


def test_aggregate_perfect_signals_reaches_one():
    cfg = MultiviewConsistencyConfig()
    sig = ConsistencySignals(
        num_matches=cfg.match_count_target,
        inlier_ratio=1.0,
        dense_support_ratio=1.0,
        reconstruction_failed=False,
    )
    assert failure_aware_aggregate(sig, cfg) == pytest.approx(1.0)


def test_aggregate_zero_in_any_component_zeroes_score():
    # Geometric mean: a single zero kills the score.
    cfg = MultiviewConsistencyConfig()
    sig = ConsistencySignals(
        num_matches=0,
        inlier_ratio=1.0,
        dense_support_ratio=1.0,
        reconstruction_failed=False,
    )
    assert failure_aware_aggregate(sig, cfg) == 0.0


def test_aggregate_missing_signals_are_skipped():
    cfg = MultiviewConsistencyConfig()
    sig = ConsistencySignals(
        num_matches=cfg.match_count_target,
        inlier_ratio=None,
        dense_support_ratio=None,
        reconstruction_failed=False,
    )
    # Only the matches signal contributes; perfect matches -> 1.0
    assert failure_aware_aggregate(sig, cfg) == pytest.approx(1.0)


def test_aggregate_all_missing_returns_zero():
    cfg = MultiviewConsistencyConfig()
    sig = ConsistencySignals(reconstruction_failed=False)
    assert failure_aware_aggregate(sig, cfg) == 0.0


def test_aggregate_partial_signals_are_geometric_mean():
    cfg = MultiviewConsistencyConfig(
        match_count_target=100,
        min_inlier_ratio=0.5,
        min_dense_support_ratio=0.5,
    )
    # matches=50 -> 0.5; inlier_ratio at threshold -> 1.0;
    # dense at threshold -> 1.0; geometric mean = (0.5 * 1 * 1) ** (1/3)
    sig = ConsistencySignals(
        num_matches=50,
        inlier_ratio=0.5,
        dense_support_ratio=0.5,
        reconstruction_failed=False,
    )
    expected = (0.5 * 1.0 * 1.0) ** (1.0 / 3.0)
    assert failure_aware_aggregate(sig, cfg) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# is_likely_hallucination
# ---------------------------------------------------------------------------


def test_is_likely_hallucination_threshold_behavior():
    cfg = MultiviewConsistencyConfig(hallucination_threshold=0.5)
    assert is_likely_hallucination(0.49, cfg) is True
    assert is_likely_hallucination(0.5, cfg) is False
    assert is_likely_hallucination(1.0, cfg) is False
    assert is_likely_hallucination(0.0, cfg) is True


# ---------------------------------------------------------------------------
# signals_for_failed_reconstruction
# ---------------------------------------------------------------------------


def test_signals_for_failed_reconstruction_defaults():
    sig = signals_for_failed_reconstruction(reason="testing")
    assert sig.reconstruction_failed is True
    assert sig.num_matches == 0
    assert sig.inlier_ratio == 0.0
    assert sig.dense_support_ratio == 0.0
    assert sig.notes == {"reason": "testing"}
    assert sig.registration_succeeded is False


# ---------------------------------------------------------------------------
# Evaluator scaffold
# ---------------------------------------------------------------------------


def test_evaluator_default_config():
    ev = MultiviewConsistencyEvaluator()
    assert isinstance(ev.config, MultiviewConsistencyConfig)


def test_evaluator_image_pair_stub_returns_failed_signals():
    ev = MultiviewConsistencyEvaluator()
    sig = ev.evaluate_image_pair(None, None)
    # The stub deliberately reports "I don't know" rather than 1.0 so
    # downstream code can detect the un-wired state.
    assert sig.reconstruction_failed is True
    assert sig.num_matches == 0
    assert "reason" in sig.notes


def test_evaluator_score_scene_no_signals_marks_hallucination():
    ev = MultiviewConsistencyEvaluator()
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    report = ev.score_scene(points_xyz=pts)
    # No multiview signals supplied -> aggregate is 0 -> below threshold.
    assert report["score"] == 0.0
    assert report["likely_hallucination"] is True
    assert report["pointcloud_validity_ratio"] == 1.0
    assert isinstance(report["signals"], ConsistencySignals)


def test_evaluator_score_scene_uses_visibility_matrix():
    cfg = MultiviewConsistencyConfig(
        match_count_target=10,
        min_inlier_ratio=0.5,
        min_dense_support_ratio=0.5,
        min_views_per_point=2,
        hallucination_threshold=0.5,
    )
    ev = MultiviewConsistencyEvaluator(config=cfg)
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    vis = np.array(
        [[1, 1, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=bool
    )  # all 4 points have >=2 views => dense_support = 1.0

    report = ev.score_scene(
        points_xyz=pts,
        view_visibility=vis,
        num_matches=10,
        inlier_ratio=1.0,
    )
    assert report["signals"].dense_support_ratio == pytest.approx(1.0)
    assert report["score"] == pytest.approx(1.0)
    assert report["likely_hallucination"] is False
    assert ev.judge(report) is True


def test_evaluator_score_scene_flags_low_dense_support():
    cfg = MultiviewConsistencyConfig(
        match_count_target=10,
        min_inlier_ratio=0.5,
        min_dense_support_ratio=0.5,
        min_views_per_point=2,
    )
    ev = MultiviewConsistencyEvaluator(config=cfg)
    # Each point seen in exactly one view => dense support 0
    vis = np.eye(4, dtype=bool)
    report = ev.score_scene(
        view_visibility=vis,
        num_matches=10,
        inlier_ratio=1.0,
    )
    assert report["signals"].dense_support_ratio == 0.0
    assert report["score"] == 0.0
    assert report["likely_hallucination"] is True


def test_evaluator_subsamples_large_pointcloud():
    cfg = MultiviewConsistencyConfig(eval_max_points=100)
    ev = MultiviewConsistencyEvaluator(config=cfg)
    pts = np.random.default_rng(42).normal(size=(1_000, 3))
    # Should not raise and should still return a validity in [0, 1].
    report = ev.score_scene(points_xyz=pts)
    assert 0.0 <= report["pointcloud_validity_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# evaluate_vqasynth_pointcloud convenience adapter
# ---------------------------------------------------------------------------


def test_evaluate_vqasynth_pointcloud_returns_report():
    pts = np.array([[0.0, 0.0, 1.0], [np.nan, 0.0, 0.0]])
    report = evaluate_vqasynth_pointcloud(pts)
    assert set(report.keys()) >= {
        "signals", "score", "likely_hallucination", "pointcloud_validity_ratio",
    }
    assert report["pointcloud_validity_ratio"] == pytest.approx(0.5)
    # No multi-view info supplied -> score is 0.
    assert report["score"] == 0.0
