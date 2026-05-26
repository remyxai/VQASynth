"""
Multi-view 3D consistency integration for VQASynth.

Adapted from "Can These Views Be One Scene? Evaluating Multiview 3D
Consistency when 3D Foundation Models Hallucinate" (arXiv:2605.18754).

The paper shows that learned 3D backbones (VGGT, MASt3R, DUSt3R, Fast3R)
can hallucinate dense geometry / cross-view support even for unrelated or
noisy inputs, and proposes COLMAP-based failure-aware signals (sparse
matches, registration success, dense-support ratio, reconstruction
failure) that correlate up to 4x better with human judgments than the
learned MEt3R metric.

VQASynth runs VGGT on a single image per scene to fuse depth + masks
into a point cloud. When VGGT hallucinates geometry for an image whose
masks/captions don't actually describe a coherent scene, the downstream
VQA pairs end up grounded in inconsistent 3D. This module gives the
pipeline a lightweight, failure-aware screen that flags such cases
*before* QA generation, and exposes a place to plug in the full
COLMAP-based metrics from the paper when COLMAP is available.

Concretely implemented here:
- ORB feature extraction + Lowe-ratio descriptor matching (OpenCV).
- Geometric verification via fundamental-matrix RANSAC inliers.
- Per-pair scores: inlier count, dense-support ratio.
- Aggregation across all view pairs (mean / min / median).

Stubbed (require external tooling beyond the current dependency set):
- Full COLMAP feature/mapper invocation for registration + reconstruction
  failure signals.
- MEt3R-style neural metric using a learned reconstruction backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MultiViewConsistencyConfig:
    """Hyperparameters for the multi-view consistency screen.

    Defaults track the paper's reported settings for the COLMAP-based
    failure-aware metrics and the parametric MEt3R family.
    """

    # ORB / descriptor matching
    orb_n_features: int = 2048
    lowe_ratio: float = 0.75

    # Geometric verification (fundamental-matrix RANSAC)
    ransac_pixel_threshold: float = 3.0
    ransac_confidence: float = 0.999

    # Failure-aware thresholds (paper Table 3 region for "registration ok")
    min_inlier_matches: int = 15
    min_dense_support: float = 0.05

    # Aggregation across all unordered view pairs
    pair_aggregation: str = "mean"  # one of: mean, min, median

    # Neural-metric components (used when MEt3R-style scoring is wired in)
    backbone: str = "vggt"  # one of: vggt, mast3r, dust3r, fast3r
    residual: str = "rgb_l2"
    aggregation: str = "mean"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_gray_uint8(image) -> np.ndarray:
    """Convert a PIL.Image or numpy array (H, W) / (H, W, C) to gray uint8."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        # Standard luminance weights; matches cv2.COLOR_RGB2GRAY closely
        rgb = arr[..., :3].astype(np.float32)
        gray = (
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        )
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def extract_orb_features(image, n_features: int = 2048):
    """Extract ORB keypoints + descriptors from a single image.

    Returns (keypoints, descriptors). Descriptors is None when no keypoints
    were detected (mirrors OpenCV's behavior so callers can branch on it).
    """
    import cv2

    gray = _to_gray_uint8(image)
    orb = cv2.ORB_create(nfeatures=int(n_features))
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(desc_a, desc_b, ratio: float = 0.75):
    """Lowe-ratio match between two binary descriptor sets.

    Returns a list of cv2.DMatch passing the ratio test. Empty list if
    either descriptor set is missing or too small for knn k=2.
    """
    import cv2

    if desc_a is None or desc_b is None:
        return []
    if len(desc_a) < 2 or len(desc_b) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def count_geometric_inliers(
    kp_a,
    kp_b,
    matches,
    pixel_threshold: float = 3.0,
    confidence: float = 0.999,
) -> int:
    """RANSAC fundamental-matrix inlier count for a set of matches.

    This is the "geometric verification" signal from the paper: matches
    that survive epipolar RANSAC are evidence the two views observe the
    same 3D structure rather than just sharing local texture.
    """
    import cv2

    if len(matches) < 8:
        return 0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

    _, mask = cv2.findFundamentalMat(
        pts_a,
        pts_b,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=float(pixel_threshold),
        confidence=float(confidence),
    )
    if mask is None:
        return 0
    return int(np.count_nonzero(mask))


def dense_support_ratio(inlier_count: int, total_keypoints: int) -> float:
    """Fraction of a view's keypoints with verified cross-view support.

    The paper uses dense-support as one of its failure-aware signals:
    backbones that hallucinate geometry between unrelated views still
    produce dense outputs, but the fraction of keypoints with real
    geometric support remains low.
    """
    if total_keypoints <= 0:
        return 0.0
    return max(0.0, min(1.0, inlier_count / float(total_keypoints)))


def aggregate_pair_scores(
    scores: Sequence[float], aggregation: str = "mean"
) -> float:
    """Aggregate per-pair scores into a single scene-level number."""
    if len(scores) == 0:
        return 0.0
    arr = np.asarray(scores, dtype=np.float64)
    if aggregation == "mean":
        return float(arr.mean())
    if aggregation == "min":
        return float(arr.min())
    if aggregation == "median":
        return float(np.median(arr))
    raise ValueError(
        f"Unknown aggregation '{aggregation}', expected mean|min|median"
    )


def pairwise_consistency_metrics(
    images: Sequence,
    config: Optional[MultiViewConsistencyConfig] = None,
) -> dict:
    """Compute pairwise feature-matching metrics across a set of views.

    Returns a dict with per-pair entries and aggregated scene-level
    numbers. The aggregated values are the lightweight stand-in for the
    paper's COLMAP-based failure-aware signals: when views genuinely
    observe one scene the inlier counts and dense-support ratios are
    high; when the backbone is hallucinating cross-view support they
    collapse toward zero.
    """
    cfg = config or MultiViewConsistencyConfig()

    if len(images) < 2:
        return {
            "n_views": len(images),
            "pairs": [],
            "mean_inliers": 0.0,
            "mean_dense_support": 0.0,
            "aggregated_score": 0.0,
            "registered": False,
        }

    features = [
        extract_orb_features(img, n_features=cfg.orb_n_features) for img in images
    ]

    pair_records: List[dict] = []
    inlier_counts: List[float] = []
    support_ratios: List[float] = []

    for (i, (kp_a, desc_a)), (j, (kp_b, desc_b)) in combinations(
        enumerate(features), 2
    ):
        matches = match_descriptors(desc_a, desc_b, ratio=cfg.lowe_ratio)
        inliers = count_geometric_inliers(
            kp_a,
            kp_b,
            matches,
            pixel_threshold=cfg.ransac_pixel_threshold,
            confidence=cfg.ransac_confidence,
        )
        total_kp = max(len(kp_a), len(kp_b), 1)
        support = dense_support_ratio(inliers, total_kp)

        pair_records.append(
            {
                "view_a": i,
                "view_b": j,
                "matches": len(matches),
                "inliers": inliers,
                "dense_support": support,
            }
        )
        inlier_counts.append(float(inliers))
        support_ratios.append(support)

    aggregated_inliers = aggregate_pair_scores(
        inlier_counts, aggregation=cfg.pair_aggregation
    )
    aggregated_support = aggregate_pair_scores(
        support_ratios, aggregation=cfg.pair_aggregation
    )

    registered = (
        aggregated_inliers >= cfg.min_inlier_matches
        and aggregated_support >= cfg.min_dense_support
    )

    return {
        "n_views": len(images),
        "pairs": pair_records,
        "mean_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
        "mean_dense_support": (
            float(np.mean(support_ratios)) if support_ratios else 0.0
        ),
        "aggregated_score": aggregated_support,
        "registered": bool(registered),
    }


# ---------------------------------------------------------------------------
# Evaluator scaffold
# ---------------------------------------------------------------------------


class MultiViewConsistencyEvaluator:
    """Failure-aware multi-view 3D consistency screen.

    The lightweight `evaluate` path is fully implemented and uses ORB +
    epipolar RANSAC to approximate the paper's COLMAP-based signals. The
    `evaluate_with_colmap` and `evaluate_with_met3r` paths are stubs that
    document where to wire in the full external tooling once COLMAP /
    a MEt3R checkpoint are available.
    """

    def __init__(self, config: Optional[MultiViewConsistencyConfig] = None):
        self.config = config or MultiViewConsistencyConfig()

    def evaluate(self, images: Sequence) -> dict:
        """Run the lightweight, dependency-free consistency screen.

        `images` may be PIL.Image objects or HxW / HxWxC numpy arrays.
        """
        return pairwise_consistency_metrics(images, config=self.config)

    def evaluate_with_colmap(self, images: Sequence, workdir: str) -> dict:
        """COLMAP-based failure-aware metrics (not yet wired up).

        TODO: invoke COLMAP feature_extractor / exhaustive_matcher /
        mapper on the supplied views (writing to `workdir`) and emit:
          - matches: total pairwise inlier-match count
          - registered_image_ratio: registered_images / n_views
          - dense_support: mean per-pixel COLMAP dense support
          - reconstruction_failed: bool, True if mapper produced 0 models
        The paper's COLMAP metrics correlate up to 4x better with human
        consistency judgments than MEt3R on real NVS outputs.
        """
        raise NotImplementedError(
            "COLMAP-backed evaluation requires the COLMAP CLI on PATH; "
            "use evaluate() for the dependency-free screen."
        )

    def evaluate_with_met3r(self, images: Sequence) -> dict:
        """Parametric MEt3R-family neural metric (not yet wired up).

        TODO: load `self.config.backbone` (VGGT / MASt3R / DUSt3R /
        Fast3R), compute the residual specified by `self.config.residual`
        in feature space, then aggregate per `self.config.aggregation`.
        The paper's parametric family recovers MEt3R as one point and
        yields variants up to 3x more robust to hallucinated geometry.
        """
        raise NotImplementedError(
            "MEt3R-style scoring requires a learned reconstruction "
            "backbone checkpoint; use evaluate() for the dependency-free "
            "screen."
        )
