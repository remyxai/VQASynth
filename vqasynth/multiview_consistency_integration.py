"""
Multiview 3D consistency metrics for VQASynth's VGGT reconstruction stage.

Scaffold for the COLMAP-based, failure-aware consistency signals introduced in:

    Can These Views Be One Scene? Evaluating Multiview 3D Consistency when
    3D Foundation Models Hallucinate
    (arXiv:2605.18754v1)

VQASynth currently relies on VGGT (vqasynth/scene_fusion.py) to lift images
into a 3D scene used by the rest of the pipeline. The paper shows VGGT can
hallucinate dense geometry and cross-view support for unrelated views,
repeated images, and random noise. The signals here let the pipeline flag
such hallucinations before downstream QA generation.

Three groups of primitives are exposed:

1. Failure-aware consistency signals (matches, registration, dense support,
   reconstruction-failure) computed from generic feature-matching outputs.
   These are concrete and tested.

2. A parametric family decomposing neural metrics into backbone / residual /
   aggregation components. The closed-form combinator is concrete; the
   backbone signal is taken as input rather than computed here.

3. A class scaffold (MultiviewConsistencyEvaluator) that wires the signals
   into a single score and a hallucination flag. The heavy lifting (running
   COLMAP, extracting SIFT/SuperPoint matches, calling MEt3R) is left as
   documented TODOs so this module doesn't pretend to do work that requires
   external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiviewConsistencyConfig:
    """
    Hyperparameters for the failure-aware consistency metrics.

    Defaults are taken from the paper's reported settings where stated and
    from reasonable choices otherwise. They are deliberately conservative:
    a view-set should score high only when the geometric evidence is strong.
    """

    # --- Match-based signal --------------------------------------------------
    # Minimum inliers (post-RANSAC) for a pair to count as "matched".
    min_inliers: int = 15
    # Minimum inlier/match ratio for a pair to count as geometrically valid.
    min_inlier_ratio: float = 0.10

    # --- Registration signal -------------------------------------------------
    # Minimum fraction of input views that must be successfully registered
    # by COLMAP for the scene to count as "registered".
    min_registered_fraction: float = 0.80

    # --- Dense support signal ------------------------------------------------
    # Minimum fraction of pixels in each view that must be covered by the
    # dense 3D support (i.e., a depth/point estimate consistent with the
    # global model).
    min_dense_support: float = 0.30

    # --- Aggregation weights (failure-aware combined score) -----------------
    # These weight the four failure-aware signals into a single [0, 1] score.
    # The defaults give roughly equal weight, slightly biased towards the
    # match signal since the paper finds geometric verification to be the
    # most reliable individual signal.
    weight_matches: float = 0.35
    weight_registration: float = 0.25
    weight_dense: float = 0.25
    weight_recon: float = 0.15

    # --- Parametric family (backbone / residual / aggregation) --------------
    # The paper introduces a parametric family that recovers MEt3R as a
    # special case and yields variants up to 3x more robust. Here the
    # backbone and residual signals are taken as inputs; alpha controls the
    # mix and gamma is an exponent on the residual term (aggregation).
    alpha: float = 0.5
    gamma: float = 1.0

    # --- Hallucination threshold --------------------------------------------
    # Combined scores below this are flagged as hallucination candidates.
    hallucination_threshold: float = 0.4

    # --- Bookkeeping ---------------------------------------------------------
    # Optional cache directory for COLMAP intermediates. Not used until the
    # COLMAP backend is implemented.
    cache_dir: Optional[str] = None

    def __post_init__(self):
        weights = (
            self.weight_matches,
            self.weight_registration,
            self.weight_dense,
            self.weight_recon,
        )
        if any(w < 0 for w in weights):
            raise ValueError("Aggregation weights must be non-negative.")
        total = sum(weights)
        if total <= 0:
            raise ValueError("At least one aggregation weight must be positive.")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must lie in [0, 1].")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")


# ---------------------------------------------------------------------------
# Failure-aware consistency signals
# ---------------------------------------------------------------------------

def match_ratio_score(num_inliers: int, num_matches: int, config: MultiviewConsistencyConfig) -> float:
    """
    Score a single pair of views by their inlier ratio after geometric
    verification (e.g., RANSAC + fundamental matrix).

    Returns a value in [0, 1]. Pairs that fall below either the absolute
    inlier count or the inlier/match ratio thresholds score 0.

    Args:
        num_inliers: Number of inliers surviving geometric verification.
        num_matches: Total number of putative matches before verification.
        config: Thresholds (min_inliers, min_inlier_ratio).
    """
    if num_inliers < 0 or num_matches < 0:
        raise ValueError("Counts must be non-negative.")
    if num_matches == 0:
        return 0.0
    ratio = num_inliers / num_matches
    if num_inliers < config.min_inliers or ratio < config.min_inlier_ratio:
        return 0.0
    # Map ratio to [0, 1] with a linear ramp above the minimum threshold.
    span = 1.0 - config.min_inlier_ratio
    if span <= 0:
        return 1.0
    return min(1.0, max(0.0, (ratio - config.min_inlier_ratio) / span))


def registration_score(num_registered: int, num_views: int, config: MultiviewConsistencyConfig) -> float:
    """
    Score how completely a COLMAP-style SfM pipeline registers the input
    views. A scene where most views drop out is a strong hallucination
    signal for any neural reconstructor that nonetheless reports high
    consistency.

    Returns a value in [0, 1] (the fraction of registered views), or 0.0
    when that fraction falls below ``config.min_registered_fraction``.
    """
    if num_registered < 0 or num_views < 0:
        raise ValueError("Counts must be non-negative.")
    if num_views == 0:
        return 0.0
    if num_registered > num_views:
        raise ValueError("Registered count cannot exceed total views.")
    frac = num_registered / num_views
    if frac < config.min_registered_fraction:
        return 0.0
    return frac


def dense_support_score(
    covered_pixels: Sequence[int],
    total_pixels: Sequence[int],
    config: MultiviewConsistencyConfig,
) -> float:
    """
    Score the fraction of each view's pixels that have a dense 3D estimate
    consistent with the registered global model. Aggregated as the mean
    across views; views below the per-view minimum score 0 for that view.

    Args:
        covered_pixels: Per-view count of pixels with valid dense support.
        total_pixels:   Per-view total pixel count (must match length).

    Returns a value in [0, 1].
    """
    if len(covered_pixels) != len(total_pixels):
        raise ValueError("covered_pixels and total_pixels must have the same length.")
    if not covered_pixels:
        return 0.0
    per_view = []
    for covered, total in zip(covered_pixels, total_pixels):
        if covered < 0 or total < 0:
            raise ValueError("Counts must be non-negative.")
        if total == 0:
            per_view.append(0.0)
            continue
        if covered > total:
            raise ValueError("Covered pixels cannot exceed total pixels.")
        frac = covered / total
        per_view.append(frac if frac >= config.min_dense_support else 0.0)
    return sum(per_view) / len(per_view)


def reconstruction_failure_score(reconstruction_succeeded: bool) -> float:
    """
    Binary signal: 1.0 if the classical SfM/MVS step finished without
    a hard failure, 0.0 otherwise. The paper treats reconstruction
    failure itself as one of the strongest cues that a view-set is
    inconsistent regardless of what a neural backbone reports.
    """
    return 1.0 if reconstruction_succeeded else 0.0


def aggregate_failure_aware_score(
    match: float,
    registration: float,
    dense: float,
    recon: float,
    config: MultiviewConsistencyConfig,
) -> float:
    """
    Weighted average of the four failure-aware signals using weights from
    ``config``. Each component is clipped into [0, 1] first.
    """
    parts = (match, registration, dense, recon)
    if any((p < 0.0 or p > 1.0) for p in parts):
        raise ValueError("Component scores must lie in [0, 1].")
    weights = (
        config.weight_matches,
        config.weight_registration,
        config.weight_dense,
        config.weight_recon,
    )
    total_weight = sum(weights)
    weighted = sum(p * w for p, w in zip(parts, weights))
    return weighted / total_weight


# ---------------------------------------------------------------------------
# Parametric family (recovers MEt3R as a special case)
# ---------------------------------------------------------------------------

def parametric_consistency_score(
    backbone: float,
    residual: float,
    config: MultiviewConsistencyConfig,
) -> float:
    """
    Closed-form parametric family from the paper, decomposing neural
    consistency metrics into backbone, residual, and aggregation
    components::

        score = alpha * backbone + (1 - alpha) * residual ** gamma

    With ``alpha = 1, gamma = 1`` this reduces to a pure backbone signal
    (recovering MEt3R). Lowering ``alpha`` and raising ``gamma`` makes
    the metric punish residual hallucination more aggressively; the paper
    reports variants up to 3x more robust by tuning this family.

    Both inputs and the result lie in [0, 1].
    """
    if not (0.0 <= backbone <= 1.0) or not (0.0 <= residual <= 1.0):
        raise ValueError("backbone and residual must lie in [0, 1].")
    score = config.alpha * backbone + (1.0 - config.alpha) * (residual ** config.gamma)
    return max(0.0, min(1.0, score))


def detect_hallucination(combined_score: float, config: MultiviewConsistencyConfig) -> bool:
    """True if ``combined_score`` falls below the hallucination threshold."""
    if not (0.0 <= combined_score <= 1.0):
        raise ValueError("combined_score must lie in [0, 1].")
    return combined_score < config.hallucination_threshold


# ---------------------------------------------------------------------------
# Evaluator class scaffold
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyReport:
    """Per-scene output of MultiviewConsistencyEvaluator.evaluate."""

    match_score: float
    registration_score: float
    dense_score: float
    recon_score: float
    combined_score: float
    hallucination_flag: bool
    parametric_score: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)


class MultiviewConsistencyEvaluator:
    """
    Drives the consistency signals against the kind of multi-view input
    VQASynth's scene_fusion stage produces (a set of images plus a VGGT-
    derived point cloud / depth map).

    Heavy lifting is intentionally stubbed:

    - ``_extract_pairwise_matches``: would shell out to COLMAP or run a
      learned matcher (SIFT, SuperPoint+LightGlue, etc.).
    - ``_run_sfm``: would run COLMAP's incremental SfM and return the
      number of registered views and per-view dense support.
    - ``_neural_backbone_score``: would call MEt3R or another learned
      metric on the same view set.

    The class exposes the pure-Python wiring so callers can already pass
    in pre-computed counts (e.g., from an external COLMAP run) and get a
    combined score + hallucination flag back.
    """

    def __init__(self, config: Optional[MultiviewConsistencyConfig] = None):
        self.config = config or MultiviewConsistencyConfig()

    # -- Public API ----------------------------------------------------------

    def evaluate_from_counts(
        self,
        num_inliers: int,
        num_matches: int,
        num_registered: int,
        num_views: int,
        covered_pixels: Sequence[int],
        total_pixels: Sequence[int],
        reconstruction_succeeded: bool,
        backbone_score: Optional[float] = None,
    ) -> ConsistencyReport:
        """
        Compute a full consistency report from pre-computed signal counts.

        This is the path that's fully implemented today. Callers that have
        already run COLMAP (or any equivalent) externally can feed counts
        in directly and get the failure-aware aggregation + hallucination
        flag for free.

        ``backbone_score`` is optional. When provided, the parametric
        family is also evaluated and attached to the report.
        """
        cfg = self.config
        m = match_ratio_score(num_inliers, num_matches, cfg)
        r = registration_score(num_registered, num_views, cfg)
        d = dense_support_score(covered_pixels, total_pixels, cfg)
        f = reconstruction_failure_score(reconstruction_succeeded)
        combined = aggregate_failure_aware_score(m, r, d, f, cfg)
        parametric = None
        if backbone_score is not None:
            parametric = parametric_consistency_score(backbone_score, combined, cfg)
        return ConsistencyReport(
            match_score=m,
            registration_score=r,
            dense_score=d,
            recon_score=f,
            combined_score=combined,
            hallucination_flag=detect_hallucination(combined, cfg),
            parametric_score=parametric,
            diagnostics={
                "num_inliers": num_inliers,
                "num_matches": num_matches,
                "num_registered": num_registered,
                "num_views": num_views,
                "reconstruction_succeeded": reconstruction_succeeded,
            },
        )

    def evaluate(self, images, point_cloud=None) -> ConsistencyReport:
        """
        End-to-end evaluation entry point. Not implemented yet because it
        requires either a COLMAP binary on PATH or a learned matcher with
        its own checkpoints (SuperPoint+LightGlue, etc.). The intended
        wiring is:

            matches = self._extract_pairwise_matches(images)
            sfm = self._run_sfm(images, matches)
            backbone = self._neural_backbone_score(images)
            return self.evaluate_from_counts(
                num_inliers=matches.num_inliers,
                num_matches=matches.num_putative,
                num_registered=sfm.num_registered,
                num_views=len(images),
                covered_pixels=sfm.dense_covered_per_view,
                total_pixels=sfm.dense_total_per_view,
                reconstruction_succeeded=sfm.succeeded,
                backbone_score=backbone,
            )

        See ``evaluate_from_counts`` for the path that works today.
        """
        # TODO(remyx-recommendation): wire COLMAP / learned matcher backend.
        raise NotImplementedError(
            "MultiviewConsistencyEvaluator.evaluate requires a COLMAP or "
            "learned-matcher backend that is not yet integrated. Use "
            "evaluate_from_counts(...) with externally computed counts in "
            "the meantime."
        )

    # -- Stubs for the heavy components -------------------------------------

    def _extract_pairwise_matches(self, images):
        # TODO(remyx-recommendation): SIFT/SuperPoint + RANSAC.
        raise NotImplementedError

    def _run_sfm(self, images, matches):
        # TODO(remyx-recommendation): COLMAP incremental SfM + dense MVS.
        raise NotImplementedError

    def _neural_backbone_score(self, images):
        # TODO(remyx-recommendation): MEt3R or VGGT-derived backbone score.
        raise NotImplementedError
