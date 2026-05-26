"""
Tests for vqasynth.multiview_consistency_integration.

Covers the failure-aware consistency signals, the parametric-family
combinator, the hallucination flag, the config validation, and a smoke
test of the evaluator scaffold.
"""

import pytest

from vqasynth.multiview_consistency_integration import (
    ConsistencyReport,
    MultiviewConsistencyConfig,
    MultiviewConsistencyEvaluator,
    aggregate_failure_aware_score,
    dense_support_score,
    detect_hallucination,
    match_ratio_score,
    parametric_consistency_score,
    reconstruction_failure_score,
    registration_score,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults_are_consistent():
    cfg = MultiviewConsistencyConfig()
    total = (
        cfg.weight_matches + cfg.weight_registration
        + cfg.weight_dense + cfg.weight_recon
    )
    assert total == pytest.approx(1.0)
    assert 0.0 <= cfg.alpha <= 1.0
    assert cfg.gamma > 0
    assert 0.0 <= cfg.hallucination_threshold <= 1.0


def test_config_rejects_negative_weights():
    with pytest.raises(ValueError):
        MultiviewConsistencyConfig(weight_matches=-0.1)


def test_config_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        MultiviewConsistencyConfig(
            weight_matches=0.0,
            weight_registration=0.0,
            weight_dense=0.0,
            weight_recon=0.0,
        )


def test_config_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        MultiviewConsistencyConfig(alpha=1.5)


def test_config_rejects_non_positive_gamma():
    with pytest.raises(ValueError):
        MultiviewConsistencyConfig(gamma=0.0)


# ---------------------------------------------------------------------------
# match_ratio_score
# ---------------------------------------------------------------------------

def test_match_ratio_zero_when_no_matches():
    cfg = MultiviewConsistencyConfig()
    assert match_ratio_score(0, 0, cfg) == 0.0


def test_match_ratio_zero_when_below_inlier_min():
    cfg = MultiviewConsistencyConfig(min_inliers=20)
    assert match_ratio_score(10, 50, cfg) == 0.0


def test_match_ratio_zero_when_below_ratio_min():
    cfg = MultiviewConsistencyConfig(min_inlier_ratio=0.5)
    assert match_ratio_score(20, 100, cfg) == 0.0


def test_match_ratio_high_for_clean_pair():
    cfg = MultiviewConsistencyConfig()
    # 80% inliers, far above defaults => positive non-trivial score
    score = match_ratio_score(80, 100, cfg)
    assert 0.0 < score <= 1.0


def test_match_ratio_maps_perfect_to_one():
    cfg = MultiviewConsistencyConfig()
    assert match_ratio_score(100, 100, cfg) == pytest.approx(1.0)


def test_match_ratio_negative_counts_raise():
    cfg = MultiviewConsistencyConfig()
    with pytest.raises(ValueError):
        match_ratio_score(-1, 10, cfg)


# ---------------------------------------------------------------------------
# registration_score
# ---------------------------------------------------------------------------

def test_registration_zero_when_no_views():
    cfg = MultiviewConsistencyConfig()
    assert registration_score(0, 0, cfg) == 0.0


def test_registration_zero_when_below_threshold():
    cfg = MultiviewConsistencyConfig(min_registered_fraction=0.9)
    assert registration_score(5, 10, cfg) == 0.0


def test_registration_returns_fraction_when_passes_threshold():
    cfg = MultiviewConsistencyConfig(min_registered_fraction=0.5)
    assert registration_score(9, 10, cfg) == pytest.approx(0.9)


def test_registration_one_when_all_registered():
    cfg = MultiviewConsistencyConfig()
    assert registration_score(10, 10, cfg) == pytest.approx(1.0)


def test_registration_rejects_overcount():
    cfg = MultiviewConsistencyConfig()
    with pytest.raises(ValueError):
        registration_score(11, 10, cfg)


# ---------------------------------------------------------------------------
# dense_support_score
# ---------------------------------------------------------------------------

def test_dense_support_empty_views():
    cfg = MultiviewConsistencyConfig()
    assert dense_support_score([], [], cfg) == 0.0


def test_dense_support_length_mismatch_raises():
    cfg = MultiviewConsistencyConfig()
    with pytest.raises(ValueError):
        dense_support_score([10, 20], [100], cfg)


def test_dense_support_zero_total_pixels_view_scores_zero():
    cfg = MultiviewConsistencyConfig()
    assert dense_support_score([0], [0], cfg) == 0.0


def test_dense_support_passes_when_all_views_above_threshold():
    cfg = MultiviewConsistencyConfig(min_dense_support=0.3)
    # 50% and 60% coverage across two views
    score = dense_support_score([50, 60], [100, 100], cfg)
    assert score == pytest.approx(0.55)


def test_dense_support_zeros_views_below_threshold():
    cfg = MultiviewConsistencyConfig(min_dense_support=0.4)
    # First view above (50%), second below (20%) -> mean = (0.5 + 0) / 2
    score = dense_support_score([50, 20], [100, 100], cfg)
    assert score == pytest.approx(0.25)


def test_dense_support_rejects_negative():
    cfg = MultiviewConsistencyConfig()
    with pytest.raises(ValueError):
        dense_support_score([-1], [10], cfg)


def test_dense_support_rejects_overcoverage():
    cfg = MultiviewConsistencyConfig()
    with pytest.raises(ValueError):
        dense_support_score([20], [10], cfg)


# ---------------------------------------------------------------------------
# reconstruction_failure_score
# ---------------------------------------------------------------------------

def test_reconstruction_failure_truthy():
    assert reconstruction_failure_score(True) == 1.0
    assert reconstruction_failure_score(False) == 0.0


# ---------------------------------------------------------------------------
# aggregate_failure_aware_score
# ---------------------------------------------------------------------------

def test_aggregate_failure_aware_with_default_weights():
    cfg = MultiviewConsistencyConfig()
    score = aggregate_failure_aware_score(1.0, 1.0, 1.0, 1.0, cfg)
    assert score == pytest.approx(1.0)
    score = aggregate_failure_aware_score(0.0, 0.0, 0.0, 0.0, cfg)
    assert score == pytest.approx(0.0)


def test_aggregate_failure_aware_weighted_correctly():
    cfg = MultiviewConsistencyConfig(
        weight_matches=1.0,
        weight_registration=0.0,
        weight_dense=0.0,
        weight_recon=0.0,
    )
    # Only matches matter
    assert aggregate_failure_aware_score(0.5, 1.0, 1.0, 1.0, cfg) == pytest.approx(0.5)


def test_aggregate_failure_aware_rejects_out_of_range():
    cfg = MultiviewConsistencyConfig()
    with pytest.raises(ValueError):
        aggregate_failure_aware_score(1.1, 0.5, 0.5, 0.5, cfg)


# ---------------------------------------------------------------------------
# parametric_consistency_score
# ---------------------------------------------------------------------------

def test_parametric_recovers_backbone_when_alpha_one():
    cfg = MultiviewConsistencyConfig(alpha=1.0, gamma=1.0)
    assert parametric_consistency_score(0.7, 0.2, cfg) == pytest.approx(0.7)


def test_parametric_uses_residual_when_alpha_zero():
    cfg = MultiviewConsistencyConfig(alpha=0.0, gamma=1.0)
    assert parametric_consistency_score(0.7, 0.2, cfg) == pytest.approx(0.2)


def test_parametric_gamma_dampens_residual():
    cfg = MultiviewConsistencyConfig(alpha=0.0, gamma=2.0)
    # With gamma=2, 0.5 ** 2 = 0.25
    assert parametric_consistency_score(0.0, 0.5, cfg) == pytest.approx(0.25)


def test_parametric_score_clipped_to_unit_interval():
    cfg = MultiviewConsistencyConfig(alpha=0.5, gamma=1.0)
    score = parametric_consistency_score(1.0, 1.0, cfg)
    assert 0.0 <= score <= 1.0


def test_parametric_rejects_out_of_range_inputs():
    cfg = MultiviewConsistencyConfig()
    with pytest.raises(ValueError):
        parametric_consistency_score(1.1, 0.5, cfg)
    with pytest.raises(ValueError):
        parametric_consistency_score(0.5, -0.1, cfg)


# ---------------------------------------------------------------------------
# detect_hallucination
# ---------------------------------------------------------------------------

def test_detect_hallucination_below_threshold():
    cfg = MultiviewConsistencyConfig(hallucination_threshold=0.5)
    assert detect_hallucination(0.2, cfg) is True
    assert detect_hallucination(0.5, cfg) is False
    assert detect_hallucination(0.9, cfg) is False


def test_detect_hallucination_rejects_out_of_range():
    cfg = MultiviewConsistencyConfig()
    with pytest.raises(ValueError):
        detect_hallucination(1.5, cfg)


# ---------------------------------------------------------------------------
# MultiviewConsistencyEvaluator
# ---------------------------------------------------------------------------

def test_evaluator_clean_scene_passes():
    """A 'clean' multi-view scene should produce a high score and no flag."""
    ev = MultiviewConsistencyEvaluator()
    report = ev.evaluate_from_counts(
        num_inliers=400,
        num_matches=500,
        num_registered=8,
        num_views=8,
        covered_pixels=[80_000] * 8,
        total_pixels=[100_000] * 8,
        reconstruction_succeeded=True,
    )
    assert isinstance(report, ConsistencyReport)
    assert report.combined_score > 0.5
    assert report.hallucination_flag is False


def test_evaluator_hallucination_scene_flagged():
    """
    A scene of mostly unrelated views (few inliers, registration fails,
    reconstruction errors out) should drop the combined score below
    threshold even if the neural backbone reported high confidence.
    """
    ev = MultiviewConsistencyEvaluator()
    report = ev.evaluate_from_counts(
        num_inliers=2,
        num_matches=300,
        num_registered=1,
        num_views=8,
        covered_pixels=[1_000] * 8,
        total_pixels=[100_000] * 8,
        reconstruction_succeeded=False,
    )
    assert report.match_score == 0.0
    assert report.registration_score == 0.0
    assert report.dense_score == 0.0
    assert report.recon_score == 0.0
    assert report.combined_score == 0.0
    assert report.hallucination_flag is True


def test_evaluator_attaches_parametric_when_backbone_given():
    ev = MultiviewConsistencyEvaluator(
        MultiviewConsistencyConfig(alpha=0.5, gamma=1.0)
    )
    report = ev.evaluate_from_counts(
        num_inliers=400,
        num_matches=500,
        num_registered=8,
        num_views=8,
        covered_pixels=[80_000] * 8,
        total_pixels=[100_000] * 8,
        reconstruction_succeeded=True,
        backbone_score=0.9,
    )
    assert report.parametric_score is not None
    assert 0.0 <= report.parametric_score <= 1.0


def test_evaluator_evaluate_is_not_implemented():
    """The end-to-end path requires a COLMAP/matcher backend not wired up yet."""
    ev = MultiviewConsistencyEvaluator()
    with pytest.raises(NotImplementedError):
        ev.evaluate(images=[None, None])


def test_evaluator_diagnostics_round_trip():
    ev = MultiviewConsistencyEvaluator()
    report = ev.evaluate_from_counts(
        num_inliers=50,
        num_matches=200,
        num_registered=6,
        num_views=8,
        covered_pixels=[40_000] * 8,
        total_pixels=[100_000] * 8,
        reconstruction_succeeded=True,
    )
    assert report.diagnostics["num_inliers"] == 50
    assert report.diagnostics["num_matches"] == 200
    assert report.diagnostics["num_registered"] == 6
    assert report.diagnostics["num_views"] == 8
    assert report.diagnostics["reconstruction_succeeded"] is True
