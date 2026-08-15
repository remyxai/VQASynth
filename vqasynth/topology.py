"""Topological and projective pairwise relations between fused object clouds.

Adds the two relation families the prompt-stage vocabulary lacks: containment
("inside") and contact ("touching"). Both are computed directly on the
VGGT-derived per-object point clouds that scene fusion already produces, so
the relations are auditable geometric evidence rather than a caption-model
guess — no extra detector or model run is needed.

Adapted from ByDeWay-V2 (arXiv:2607.27145), which computes pairwise
projective and topological relations between detected objects and injects
them as human-readable predicates into the prompt. Here the relation tests
run on the repo's existing per-object point clouds instead of YOLO-World
boxes, and the outputs feed the template VQA generator rather than a live
MLLM prompt. left/above/etc. projective predicates already exist in
``vqasynth.prompts`` and are reused, not reimplemented.
"""
import random

import numpy as np
from scipy.spatial import cKDTree

from vqasynth.prompt_templates import (
    inside_false_responses,
    inside_predicate_questions,
    inside_true_responses,
    touching_false_responses,
    touching_predicate_questions,
    touching_true_responses,
)


def _cloud_to_points(cloud):
    """Per-object clouds arrive either as open3d geometry or a raw Nx3 array."""
    if isinstance(cloud, np.ndarray):
        return cloud
    return np.asarray(cloud.points)


def _percentile_scale(points, percentile=95):
    """Per-axis extent robust to stray depth outliers.

    Masks come back from SAM2 with a few stray pixels, and VGGT depth is not
    perfectly clean, so a max/min extent can be set by a single bad point. The
    95th-percentile spread keeps thresholds relative to the bulk of the cloud.
    """
    extent = np.percentile(points, percentile, axis=0) - np.percentile(
        points, 100 - percentile, axis=0
    )
    return float(np.median(extent)) if np.any(extent > 0) else 0.0


def is_inside(inner_cloud, outer_cloud, threshold=0.9):
    """Fraction of ``inner`` points inside ``outer``'s axis-aligned hull."""
    inner = _cloud_to_points(inner_cloud)
    outer = _cloud_to_points(outer_cloud)

    if len(inner) == 0 or len(outer) == 0:
        return False

    lower, upper = outer.min(axis=0), outer.max(axis=0)
    contained = np.all((inner >= lower) & (inner <= upper), axis=1)
    fraction = float(contained.mean())

    scale = _percentile_scale(outer)
    if scale > 0 and (inner.max(axis=0) - inner.min(axis=0)).max() > 2.0 * scale:
        return False  # inner spans well beyond outer — a hull, not containment

    return fraction >= threshold


def is_touching(cloud_a, cloud_b, percentile=10):
    """True when the two clouds are within contact range in 3D.

    Contact is judged by the 10th-percentile point-to-point distance between
    the clouds, compared against a scale derived from the objects' own
    extents, so the test does not need metric-calibrated depth.
    """
    points_a = _cloud_to_points(cloud_a)
    points_b = _cloud_to_points(cloud_b)

    if len(points_a) == 0 or len(points_b) == 0:
        return False

    tree_b = cKDTree(points_b)
    distances, _ = tree_b.query(points_a)
    contact_distance = float(np.percentile(distances, percentile))

    # Interquartile spread (p75 - p25) as the object-scale reference.
    scale = max(
        _percentile_scale(points_a, 75), _percentile_scale(points_b, 75), 1e-6
    )
    # 0.25 of object scale: near enough that SAM2 mask bleed or VGGT depth
    # noise at a shared boundary reads as contact, while a visible gap does
    # not (a mug on a table passes; a mug a hand's width above it does not).
    return contact_distance <= 0.25 * scale


class TopologicalRelationGenerator:
    """Renders inside/touching predicates as VQA prompts for an object pair.

    Mirrors the ``*_predicate`` methods on
    :class:`vqasynth.prompts.PromptGenerator` so the two relation families can
    slot into the same prompt-variant pool.
    """

    def inside_predicate(self, A, B):
        A_desc, B_desc = A[0].lower(), B[0].lower()
        is_in = is_inside(A[1], B[1])

        question = random.choice(inside_predicate_questions)
        response = random.choice(
            inside_true_responses if is_in else inside_false_responses
        )

        question = question.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response.replace("[A]", A_desc).replace("[B]", B_desc)
        return question + " Answer: " + answer

    def touching_predicate(self, A, B):
        A_desc, B_desc = A[0].lower(), B[0].lower()
        contact = is_touching(A[1], B[1])

        question = random.choice(touching_predicate_questions)
        response = random.choice(
            touching_true_responses if contact else touching_false_responses
        )

        question = question.replace("[A]", A_desc).replace("[B]", B_desc)
        answer = response.replace("[A]", A_desc).replace("[B]", B_desc)
        return question + " Answer: " + answer
