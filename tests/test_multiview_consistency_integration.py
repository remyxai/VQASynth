"""Tests for vqasynth.multiview_consistency_integration."""

import pytest

from vqasynth.multiview_consistency_integration import (
    MultiviewConsistencyConfig,
    MultiviewConsistencyMetric,
    aggregate_pair_scores,
    consistency_from_components,
    dense_support_ratio,
    enumerate_view_pairs,
    match_ratio_score,
    registration_indicator,
)


def test_config_defaults():
    cfg = MultiviewConsistencyConfig()
    assert cfg.pair_strategy == "all"
    assert cfg.min_matches == 25
    assert cfg.aggregation == "mean"
    assert cfg.failure_is_zero is True
    assert cfg.colmap_binary is None


def test_enumerate_view_pairs_all():
    assert enumerate_view_pairs(0) == []
    assert enumerate_view_pairs(1) == []
    assert enumerate_view_pairs(2) == [(0, 1)]
    assert enumerate_view_pairs(3) == [(0, 1), (0, 2), (1, 2)]


def test_enumerate_view_pairs_sequential():
    assert enumerate_view_pairs(3, "sequential") == [(0, 1), (1, 2)]
    assert enumerate_view_pairs(4, "sequential") == [(0, 1), (1, 2), (2, 3)]


def test_enumerate_view_pairs_invalid():
    with pytest.raises(ValueError):
        enumerate_view_pairs(3, "not-a-strategy")


def test_match_ratio_score_bounds():
    assert match_ratio_score(0, 100) == 0.0
    assert match_ratio_score(50, 100) == 0.5
    assert match_ratio_score(100, 100) == 1.0
    # Clipped when matches > keypoints (defensive against bad COLMAP output).
    assert match_ratio_score(200, 100) == 1.0
    # Degenerate input returns 0.0, not a divide-by-zero.
    assert match_ratio_score(10, 0) == 0.0


def test_registration_indicator():
    assert registration_indicator(0, 25) == 0.0
    assert registration_indicator(24, 25) == 0.0
    assert registration_indicator(25, 25) == 1.0
    assert registration_indicator(1000, 25) == 1.0


def test_dense_support_ratio():
    assert dense_support_ratio(0, 1000) == 0.0
    assert dense_support_ratio(500, 1000) == 0.5
    assert dense_support_ratio(2000, 1000) == 1.0
    assert dense_support_ratio(100, 0) == 0.0


def test_aggregate_pair_scores_methods():
    assert aggregate_pair_scores([], "mean") == 0.0
    assert aggregate_pair_scores([0.2, 0.4, 0.6], "mean") == pytest.approx(0.4)
    assert aggregate_pair_scores([0.2, 0.4, 0.6], "min") == pytest.approx(0.2)
    assert aggregate_pair_scores([0.2, 0.4, 0.6], "median") == pytest.approx(0.4)


def test_aggregate_pair_scores_invalid():
    with pytest.raises(ValueError):
        aggregate_pair_scores([0.1, 0.2], "geometric_mean")


def test_consistency_from_components_normal():
    val = consistency_from_components(
        match_score=0.5,
        registration_score=1.0,
        dense_score=0.0,
        reconstruction_failed=False,
    )
    assert val == pytest.approx(0.5)


def test_consistency_from_components_failure_zeroed():
    val = consistency_from_components(1.0, 1.0, 1.0, reconstruction_failed=True)
    assert val == 0.0


def test_consistency_from_components_failure_not_zeroed():
    val = consistency_from_components(
        1.0, 1.0, 1.0, reconstruction_failed=True, failure_is_zero=False
    )
    assert val == pytest.approx(1.0)


def test_metric_score_no_stats_returns_default():
    metric = MultiviewConsistencyMetric()
    # No evidence at all — neither high nor zero confidence.
    assert metric.score(pair_stats=None) == 0.5


def test_metric_score_no_stats_with_failure():
    metric = MultiviewConsistencyMetric()
    assert metric.score(pair_stats=None, reconstruction_failed=True) == 0.0


def test_metric_score_with_pair_stats():
    metric = MultiviewConsistencyMetric()
    pair_stats = [
        {"matches": 100, "keypoints": 200, "dense_points": 5000},
        {"matches": 80, "keypoints": 200, "dense_points": 4000},
    ]
    score = metric.score(pair_stats=pair_stats, num_pixels=10000)
    assert 0.0 <= score <= 1.0
    # All pairs register and produce dense support, so score should be high.
    assert score > 0.5


def test_metric_score_reconstruction_failure_overrides():
    metric = MultiviewConsistencyMetric()
    pair_stats = [{"matches": 100, "keypoints": 200, "dense_points": 5000}]
    score = metric.score(
        pair_stats=pair_stats, num_pixels=10000, reconstruction_failed=True
    )
    assert score == 0.0


def test_run_colmap_is_stub():
    metric = MultiviewConsistencyMetric()
    with pytest.raises(NotImplementedError):
        metric._run_colmap(images=[])


def test_custom_config_threads_through():
    cfg = MultiviewConsistencyConfig(min_matches=200, aggregation="min")
    metric = MultiviewConsistencyMetric(cfg)
    # 100 < min_matches=200, so registration indicator = 0.
    pair_stats = [{"matches": 100, "keypoints": 200, "dense_points": 0}]
    score = metric.score(pair_stats=pair_stats, num_pixels=10000)
    assert score < 0.5
