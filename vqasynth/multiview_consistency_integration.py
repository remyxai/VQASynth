"""
Multiview 3D consistency metrics — integration scaffold.

Adapted from "Can These Views Be One Scene? Evaluating Multiview 3D
Consistency when 3D Foundation Models Hallucinate"
(https://arxiv.org/abs/2605.18754v1).

The paper shows that neural multi-view reconstruction backbones (VGGT,
MASt3R, DUSt3R, Fast3R) can hallucinate dense geometry and cross-view
support for unrelated scenes, repeated images, or pure noise — and that
classical COLMAP-based signals (matches, registration, dense support,
reconstruction failure) correlate up to 4x better with human judgments
of multiview consistency than learned metrics like MEt3R.

This module is experimental scaffolding contributed by Remyx
Recommendation (https://github.com/remyxai/mhpd-dpo-training/tree/main/agent).
It provides:
  * a config dataclass holding the paper's reported hyperparameters,
  * concrete utility functions for the parametric metric components
    (match ratio, registration indicator, dense support, aggregation),
  * a class scaffold that combines the components into a single score.

The COLMAP shell-out is intentionally a TODO — the binary is not a
required dependency of vqasynth and this PR does not pretend to invoke
it. Callers can supply precomputed per-pair statistics to `score()`.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MultiviewConsistencyConfig:
    """Hyperparameters for COLMAP-based multi-view consistency scoring."""

    pair_strategy: str = "all"
    min_matches: int = 25
    match_ratio: float = 0.7
    feature_image_size: int = 1024
    aggregation: str = "mean"
    failure_is_zero: bool = True
    colmap_binary: Optional[str] = None


def enumerate_view_pairs(num_views: int, strategy: str = "all") -> List[Tuple[int, int]]:
    """Return (i, j) view-index pairs for a given number of views.

    "all" yields every unordered pair, "sequential" yields consecutive pairs.
    """
    if num_views < 2:
        return []
    if strategy == "all":
        return list(combinations(range(num_views), 2))
    if strategy == "sequential":
        return [(i, i + 1) for i in range(num_views - 1)]
    raise ValueError(f"Unknown pair strategy: {strategy!r}")


def match_ratio_score(num_matches: int, num_keypoints: int) -> float:
    """Fraction of keypoints that matched, clipped to [0, 1]."""
    if num_keypoints <= 0:
        return 0.0
    return float(min(1.0, max(0.0, num_matches / num_keypoints)))


def registration_indicator(num_matches: int, min_matches: int) -> float:
    """1.0 if a pair would register under COLMAP's match threshold, else 0.0."""
    return 1.0 if num_matches >= min_matches else 0.0


def dense_support_ratio(num_dense_points: int, num_pixels: int) -> float:
    """Recovered dense points as a fraction of image pixels, clipped to [0, 1]."""
    if num_pixels <= 0:
        return 0.0
    return float(min(1.0, max(0.0, num_dense_points / num_pixels)))


def aggregate_pair_scores(scores: Sequence[float], method: str = "mean") -> float:
    """Aggregate per-pair scores into a scene-level value.

    Implements the aggregation slot of the paper's backbone/residual/
    aggregation parametric family.
    """
    if len(scores) == 0:
        return 0.0
    arr = np.asarray(list(scores), dtype=float)
    if method == "mean":
        return float(arr.mean())
    if method == "min":
        return float(arr.min())
    if method == "median":
        return float(np.median(arr))
    raise ValueError(f"Unknown aggregation method: {method!r}")


def consistency_from_components(
    match_score: float,
    registration_score: float,
    dense_score: float,
    reconstruction_failed: bool,
    failure_is_zero: bool = True,
) -> float:
    """Combine the four COLMAP-based failure-aware signals into [0, 1]."""
    if reconstruction_failed and failure_is_zero:
        return 0.0
    parts = [match_score, registration_score, dense_score]
    return float(np.clip(np.mean(parts), 0.0, 1.0))


class MultiviewConsistencyMetric:
    """Score multi-view 3D consistency for a small batch of VQASynth views.

    For a single image, this returns a default low-evidence score: one
    image cannot be checked for cross-view consistency, which is itself
    a signal worth surfacing to callers.

    The COLMAP step is intentionally not invoked here. Subclasses or
    callers should either override `_run_colmap` to shell out to the
    binary and parse its sparse database, or pass precomputed per-pair
    statistics directly to `score()`.
    """

    def __init__(self, config: Optional[MultiviewConsistencyConfig] = None):
        self.config = config or MultiviewConsistencyConfig()

    def _run_colmap(self, images):  # pragma: no cover - external dep
        """Shell out to COLMAP and return per-pair statistics.

        TODO: invoke `colmap feature_extractor` / `exhaustive_matcher` /
        `mapper`, then parse the resulting database to return a list of
        dicts with keys "matches", "keypoints", "dense_points" per
        view pair. Requires COLMAP installed on PATH; left out of this
        scaffold to avoid adding an external dependency.
        """
        raise NotImplementedError(
            "COLMAP execution is not implemented in this scaffold. "
            "Override _run_colmap or pass precomputed pair_stats to score()."
        )

    def score(
        self,
        pair_stats: Optional[Sequence[dict]] = None,
        num_pixels: int = 0,
        reconstruction_failed: bool = False,
    ) -> float:
        """Aggregate precomputed per-pair stats into a single consistency score.

        Args:
            pair_stats: list of per-pair dicts with keys "matches",
                "keypoints", "dense_points".
            num_pixels: image pixel count for the dense support ratio.
            reconstruction_failed: True if COLMAP's mapper failed to converge.

        Returns:
            Consistency value in [0, 1] (1.0 = highly consistent).
        """
        cfg = self.config
        if not pair_stats:
            if reconstruction_failed and cfg.failure_is_zero:
                return 0.0
            return 0.5

        match_scores, reg_scores, dense_scores = [], [], []
        for s in pair_stats:
            matches = int(s.get("matches", 0))
            keypoints = int(s.get("keypoints", 0))
            dense_pts = int(s.get("dense_points", 0))
            match_scores.append(match_ratio_score(matches, keypoints))
            reg_scores.append(registration_indicator(matches, cfg.min_matches))
            dense_scores.append(dense_support_ratio(dense_pts, num_pixels or 1))

        return consistency_from_components(
            match_score=aggregate_pair_scores(match_scores, cfg.aggregation),
            registration_score=aggregate_pair_scores(reg_scores, cfg.aggregation),
            dense_score=aggregate_pair_scores(dense_scores, cfg.aggregation),
            reconstruction_failed=reconstruction_failed,
            failure_is_zero=cfg.failure_is_zero,
        )
