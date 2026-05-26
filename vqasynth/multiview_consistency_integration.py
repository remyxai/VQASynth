"""
Multiview 3D consistency evaluation scaffolding.

Adapts ideas from:
    "Can These Views Be One Scene? Evaluating Multiview 3D Consistency
     when 3D Foundation Models Hallucinate"
    (https://arxiv.org/abs/2605.18754v1)

The paper shows that neural 3D foundation models (VGGT, MASt3R, DUSt3R,
Fast3R) can hallucinate dense geometry and cross-view support for
unrelated scenes, repeated views, or pure noise. It proposes
COLMAP-grounded "failure-aware" signals — number of feature matches,
registration outcome, dense support, and reconstruction-failure flag —
that correlate better with human judgments than learned reference-free
metrics such as MEt3R.

VQASynth uses VGGT as its depth / point-cloud backbone (see
``vqasynth.scene_fusion``), so the paper's hallucination findings apply
directly to the data this pipeline generates. This module provides the
plumbing for an evaluation pass that flags potentially-hallucinated
reconstructions before they propagate into the synthetic VQA dataset.

What's implemented here:
    * ``MultiviewConsistencyConfig``: paper-default hyperparameters.
    * ``ConsistencySignals``: the four failure-aware signals.
    * Concrete utility functions (``dense_support_ratio``,
      ``failure_aware_aggregate``, ``is_likely_hallucination``,
      ``normalize_match_count``, ``pointcloud_validity_ratio``).
    * ``MultiviewConsistencyEvaluator``: class scaffold whose
      COLMAP-dependent steps are documented TODOs returning sensible
      "no reconstruction" defaults so callers can wire the rest of the
      pipeline before pulling in COLMAP/pycolmap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Config & signals
# ---------------------------------------------------------------------------


@dataclass
class MultiviewConsistencyConfig:
    """Hyperparameters for failure-aware multiview consistency scoring.

    Defaults reflect the paper's reported settings where stated and
    conservative VQASynth-side defaults otherwise.
    """

    # Number of SIFT/LoFTR-style matches at which we consider the pair
    # "well-matched". Match counts above this saturate to 1.0 when
    # normalized. The paper uses match-count thresholds in this range
    # for the COLMAP-matches signal.
    match_count_target: int = 100

    # Inlier ratio threshold for a pair to count as geometrically
    # registered (RANSAC inliers / total matches).
    min_inlier_ratio: float = 0.30

    # Fraction of dense points that must be supported by >= 2 views for
    # the scene to count as densely consistent. Below this, dense
    # support is treated as a failure signal.
    min_dense_support_ratio: float = 0.50

    # Minimum number of views a 3D point must be visible in to count as
    # "multi-view supported". Two views is the geometric minimum.
    min_views_per_point: int = 2

    # Final-score threshold below which the reconstruction is flagged
    # as likely-hallucinated. Tunable per use case.
    hallucination_threshold: float = 0.50

    # Weights for the failure-aware aggregate (matches, registration,
    # dense_support). These are the "aggregation" component of the
    # paper's parametric family (backbone, residual, aggregation).
    # Geometric-mean aggregation is used by default; weights bias which
    # signal dominates.
    weight_matches: float = 1.0
    weight_registration: float = 1.0
    weight_dense_support: float = 1.0

    # Optional cap on point cloud size during evaluation; larger clouds
    # are randomly subsampled to keep evaluation cheap.
    eval_max_points: int = 200_000


@dataclass
class ConsistencySignals:
    """The four failure-aware signals from the paper.

    Any signal that could not be computed should be left as ``None`` so
    the aggregator can decide how to handle it (vs. treating a missing
    signal as 0).
    """

    num_matches: Optional[int] = None
    inlier_ratio: Optional[float] = None
    dense_support_ratio: Optional[float] = None
    reconstruction_failed: bool = False
    # Free-form metadata for debugging / reporting.
    notes: dict = field(default_factory=dict)

    @property
    def registration_succeeded(self) -> bool:
        """Heuristic: registration is OK iff we have an inlier ratio
        above zero. Callers can override by setting ``notes`` and
        passing an explicit min_inlier_ratio to the aggregator."""
        return self.inlier_ratio is not None and self.inlier_ratio > 0.0


def signals_for_failed_reconstruction(reason: str = "not computed") -> ConsistencySignals:
    """Return a ``ConsistencySignals`` value representing a reconstruction
    that did not complete. Useful as a default when COLMAP / matching
    has not been wired up yet."""
    return ConsistencySignals(
        num_matches=0,
        inlier_ratio=0.0,
        dense_support_ratio=0.0,
        reconstruction_failed=True,
        notes={"reason": reason},
    )


# ---------------------------------------------------------------------------
# Concrete utilities
# ---------------------------------------------------------------------------


def normalize_match_count(num_matches: Optional[int], target: int) -> float:
    """Squash a match count to [0, 1] by linearly saturating at ``target``.

    Returns 0.0 for ``None`` or non-positive inputs. Above ``target`` the
    result is clamped at 1.0. ``target`` must be positive.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    if num_matches is None or num_matches <= 0:
        return 0.0
    return float(min(num_matches, target)) / float(target)


def dense_support_ratio(
    view_visibility: np.ndarray, min_views: int = 2
) -> float:
    """Fraction of points visible in at least ``min_views`` views.

    Args:
        view_visibility: ``(N_points, N_views)`` boolean (or 0/1) array
            indicating whether each point is visible in each view.
        min_views: Minimum view count required to count a point as
            multi-view supported.

    Returns 0.0 for empty inputs.
    """
    if min_views < 1:
        raise ValueError("min_views must be >= 1")

    arr = np.asarray(view_visibility)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2:
        raise ValueError(
            f"view_visibility must be 2D (points, views); got shape {arr.shape}"
        )

    per_point = arr.astype(bool).sum(axis=1)
    supported = (per_point >= min_views).sum()
    return float(supported) / float(per_point.shape[0])


def pointcloud_validity_ratio(points_xyz: np.ndarray) -> float:
    """Fraction of points with finite, non-degenerate coordinates.

    A point is invalid if any coordinate is non-finite (NaN / +/-Inf).
    The all-zero point is *not* treated as invalid here — VGGT can emit
    near-origin points for distant content that are still meaningful;
    callers wanting stricter validity should add their own check.

    Returns 0.0 for empty inputs.
    """
    arr = np.asarray(points_xyz)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"points_xyz must have shape (N, 3); got {arr.shape}"
        )

    finite_mask = np.isfinite(arr).all(axis=1)
    return float(finite_mask.sum()) / float(arr.shape[0])


def failure_aware_aggregate(
    signals: ConsistencySignals, config: MultiviewConsistencyConfig
) -> float:
    """Combine the four failure-aware signals into a single [0, 1] score.

    Semantics:
        * If ``reconstruction_failed`` is True, the score is 0 regardless
          of the other signals (the paper's hard-failure signal).
        * Otherwise we compute a weighted geometric mean over:
            - normalized match count
            - registration indicator (1.0 if inlier_ratio >= threshold,
              else inlier_ratio scaled to [0, 1] by that threshold)
            - dense support indicator (1.0 if dense_support_ratio meets
              the configured floor, else dense_support_ratio scaled)
        * Missing signals (``None``) skip that component and renormalize
          the weights so the score still spans [0, 1].

    Weighted geometric mean is used (rather than arithmetic) because the
    paper emphasizes that a single failure signal — e.g. registration
    failing — should pull the metric down even if other signals look
    healthy.
    """
    if signals.reconstruction_failed:
        return 0.0

    components: List[tuple] = []  # (value_in_[0,1], weight)

    if signals.num_matches is not None:
        v = normalize_match_count(signals.num_matches, config.match_count_target)
        components.append((v, config.weight_matches))

    if signals.inlier_ratio is not None:
        thr = max(config.min_inlier_ratio, 1e-6)
        v = min(signals.inlier_ratio / thr, 1.0)
        v = max(v, 0.0)
        components.append((v, config.weight_registration))

    if signals.dense_support_ratio is not None:
        thr = max(config.min_dense_support_ratio, 1e-6)
        v = min(signals.dense_support_ratio / thr, 1.0)
        v = max(v, 0.0)
        components.append((v, config.weight_dense_support))

    if not components:
        return 0.0

    total_weight = sum(w for _, w in components)
    if total_weight <= 0:
        return 0.0

    # Weighted geometric mean. Use log-space and treat exact zeros as a
    # hard zero (the paper's "failure" semantics).
    log_sum = 0.0
    for v, w in components:
        if v <= 0.0:
            return 0.0
        log_sum += w * math.log(v)
    return math.exp(log_sum / total_weight)


def is_likely_hallucination(
    score: float, config: MultiviewConsistencyConfig
) -> bool:
    """Binary judgment: is this reconstruction below the hallucination
    threshold? ``score`` is the value returned by ``failure_aware_aggregate``.
    """
    return score < config.hallucination_threshold


# ---------------------------------------------------------------------------
# Evaluator scaffold
# ---------------------------------------------------------------------------


class MultiviewConsistencyEvaluator:
    """Evaluate failure-aware multiview consistency for a VQASynth scene.

    The intended usage is to wrap ``SpatialSceneConstructor`` output and
    decide whether the reconstructed 3D scene is trustworthy enough to
    use for generating spatial-VQA pairs.

    Current state:
        * ``score_scene`` and ``judge`` are concrete and operate on
          point clouds plus optional per-view visibility matrices.
        * ``evaluate_image_pair`` is a STUB: pulling in COLMAP /
          pycolmap, or a LoFTR / SuperGlue matcher, is left as a TODO
          so that this module imports cleanly without those deps.
    """

    def __init__(self, config: Optional[MultiviewConsistencyConfig] = None):
        self.config = config or MultiviewConsistencyConfig()

    # --- COLMAP-dependent path (stubbed) ----------------------------------

    def evaluate_image_pair(self, image_a, image_b) -> ConsistencySignals:
        """Compute failure-aware signals for a single image pair.

        TODO(remyx-rec): implement using pycolmap or a LoFTR/SuperPoint
        matcher to obtain real match counts and a RANSAC inlier ratio.
        Pseudocode:

            kpts_a, desc_a = sift_or_loftr(image_a)
            kpts_b, desc_b = sift_or_loftr(image_b)
            matches = mutual_nn(desc_a, desc_b)
            F, inlier_mask = cv2.findFundamentalMat(...)
            return ConsistencySignals(
                num_matches=len(matches),
                inlier_ratio=inlier_mask.sum() / max(len(matches), 1),
                ...
            )

        Until that lands, return a "reconstruction failed" sentinel so
        downstream code receives a clear "I do not know" signal instead
        of an over-confident 1.0.
        """
        return signals_for_failed_reconstruction(
            reason="evaluate_image_pair: COLMAP/feature-match backbone not wired up"
        )

    # --- Concrete path ----------------------------------------------------

    def score_scene(
        self,
        points_xyz: Optional[np.ndarray] = None,
        view_visibility: Optional[np.ndarray] = None,
        num_matches: Optional[int] = None,
        inlier_ratio: Optional[float] = None,
    ) -> dict:
        """Score a reconstructed scene using whichever signals the caller
        has been able to compute.

        Args:
            points_xyz: ``(N, 3)`` point coordinates from the VGGT
                reconstruction. Used to compute a validity ratio that
                gets surfaced in the report but not folded into the
                aggregate score (the paper's signals operate on
                COLMAP-style support, not finite-vs-NaN counts).
            view_visibility: Optional ``(N_points, N_views)`` boolean
                matrix. When supplied, ``dense_support_ratio`` is
                computed from it.
            num_matches: Optional pair-level match count to feed in.
            inlier_ratio: Optional pair-level RANSAC inlier ratio.

        Returns a report dict with keys ``signals``, ``score``,
        ``likely_hallucination``, and ``pointcloud_validity_ratio``.
        """
        config = self.config

        dense_ratio = None
        if view_visibility is not None:
            dense_ratio = dense_support_ratio(
                view_visibility, min_views=config.min_views_per_point
            )

        signals = ConsistencySignals(
            num_matches=num_matches,
            inlier_ratio=inlier_ratio,
            dense_support_ratio=dense_ratio,
            reconstruction_failed=False,
        )

        score = failure_aware_aggregate(signals, config)

        validity = None
        if points_xyz is not None:
            arr = np.asarray(points_xyz)
            if arr.size and arr.ndim == 2 and arr.shape[0] > config.eval_max_points:
                idx = np.random.default_rng(0).choice(
                    arr.shape[0], size=config.eval_max_points, replace=False
                )
                arr = arr[idx]
            validity = pointcloud_validity_ratio(arr) if arr.size else 0.0

        return {
            "signals": signals,
            "score": score,
            "likely_hallucination": is_likely_hallucination(score, config),
            "pointcloud_validity_ratio": validity,
        }

    def judge(self, scene_report: dict) -> bool:
        """Return True if the scene is clean to use, False if it should
        be filtered out as likely-hallucinated."""
        return not scene_report.get("likely_hallucination", True)


# ---------------------------------------------------------------------------
# Convenience adapter for VQASynth scene_fusion output
# ---------------------------------------------------------------------------


def evaluate_vqasynth_pointcloud(
    points_xyz: np.ndarray,
    config: Optional[MultiviewConsistencyConfig] = None,
) -> dict:
    """Convenience entry point for the common single-image VQASynth case.

    ``SpatialSceneConstructor`` currently runs VGGT on one image at a
    time, so we have no native multi-view visibility matrix to feed in.
    This helper still returns a report — with the multiview-derived
    signals left as ``None`` — so users can at least see the point
    cloud validity ratio while the multi-view path is being built out.
    """
    evaluator = MultiviewConsistencyEvaluator(config=config)
    return evaluator.score_scene(points_xyz=points_xyz)
