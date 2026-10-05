"""Privileged spatial context for teacher-conditioned CoT generation.

The reasoning stage's teacher (``vqasynth.r1_reasoning.R1Reasoner``)
formulates chain-of-thought traces from a question/answer pair alone, so
its prompt has to hedge that "some information may be partially
incorrect". The pipeline already holds exact metric facts for every
detected object -- VGGT point clouds fused with SAM2 masks -- in the
``captions`` and ``pointclouds`` columns that survive into the reasoning
stage's source dataset; they were simply never shown to the teacher.

This module turns those columns into a compact textual spatial-guidance
block: object identities with metric centers and sizes, plus the mean
surface distance between the objects a question references. Adapted from
the on-policy self-distillation setup in "Where-OPD: Spatially Guided
On-Policy Self-Distillation of MLLMs with Synthetic Scenes"
(arXiv:2610.02117): the guidance conditions the *teacher* only. The
student-visible training fields (image + question -> reasoning) never
contain it -- the block is emitted in a separate ``spatial_context``
column so downstream consumers can audit it or strip it entirely.
"""

from __future__ import annotations

import os

import numpy as np

# PromptGenerator.human_like_distance scales raw cloud units by this factor
# before rendering them as meters in the Q/A text; the guidance block uses
# the same factor so its measurements agree with the answers the teacher
# must support.
DEFAULT_DISTANCE_SCALE = 10.0

# Mask clouds can hold tens of thousands of points; cap what feeds the
# pairwise distance estimate so the guidance block stays cheap to build.
_MAX_DISTANCE_POINTS = 512


def _load_cloud(value):
    """Pass through in-memory clouds; read path-valued cells lazily.

    The ``pointclouds`` column stores file paths written by the fusion
    stage, so open3d is only imported when a path actually shows up (and
    unreadable cells are skipped gracefully when it is unavailable).
    """
    if isinstance(value, (str, os.PathLike)) and os.path.exists(value):
        try:
            import open3d as o3d

            return o3d.io.read_point_cloud(str(value))
        except Exception as e:  # missing open3d, unreadable file, ...
            print(f"[privileged_context] skipping unreadable point cloud {value}: {e}")
            return None
    return value


def _points_array(cloud):
    """Coerce a cloud (open3d geometry, ndarray, nested list) to (N, 3) floats."""
    if cloud is None:
        return None
    points = getattr(cloud, "points", cloud)
    try:
        arr = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return None
    return np.ascontiguousarray(arr[:, :3])


def _collect_objects(captions, pointclouds, scale):
    """Single pass over the columns: ``(record, points)`` per usable object.

    Path-valued cells are read exactly once; the points ride along so the
    pairwise separation estimate never re-reads the clouds.
    """
    entries = []
    for caption, cloud in zip(captions or [], pointclouds or []):
        pts = _points_array(_load_cloud(cloud))
        description = str(caption).strip() if caption is not None else ""
        if pts is None or not description:
            continue
        entries.append(
            (
                {
                    "description": description,
                    "center": (pts.mean(axis=0) * scale).round(2),
                    "extent": ((pts.max(axis=0) - pts.min(axis=0)) * scale).round(2),
                },
                pts,
            )
        )
    return entries


def object_spatial_records(captions, pointclouds, scale=DEFAULT_DISTANCE_SCALE):
    """Pair each caption with the metric center and size of its point cloud.

    Mirrors the geometry PromptGenerator derives its questions from
    (``get_center`` and the axis-aligned bounding-box extent). Objects
    whose caption is empty or whose cloud is unusable are skipped.
    """
    return [record for record, _ in _collect_objects(captions, pointclouds, scale)]


def match_objects_in_text(text, records):
    """Indices of records whose description appears in ``text`` (case-insensitive).

    Questions are rendered by substituting lowercased captions into
    templates, so substring matching recovers which scene objects a query
    is about. Returned longest-description-first so the most specific
    caption wins when one description contains another.
    """
    haystack = (text or "").lower()
    matched = [
        i
        for i, record in enumerate(records)
        if record["description"] and record["description"].lower() in haystack
    ]
    return sorted(matched, key=lambda i: len(records[i]["description"]), reverse=True)


def _mean_surface_distance(pts_a, pts_b):
    """Mean nearest-neighbor distance between two point sets.

    The same statistic PromptGenerator computes via
    ``compute_point_cloud_distance(...).mean()``, estimated on strided
    subsamples to bound the cost for large mask clouds.
    """
    step_a = max(1, len(pts_a) // _MAX_DISTANCE_POINTS)
    step_b = max(1, len(pts_b) // _MAX_DISTANCE_POINTS)
    a = pts_a[::step_a]
    b = pts_b[::step_b]
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return dists.min(axis=1).mean()


def _format_object(record):
    center = [float(v) for v in record["center"]]
    extent = [float(v) for v in record["extent"]]
    return (
        f'- "{record["description"]}": center '
        f"(x={center[0]}, y={center[1]}, z={center[2]}) meters, size "
        f"{extent[0]} x {extent[1]} x {extent[2]} meters"
    )


def build_spatial_context(
    captions,
    pointclouds,
    question=None,
    scale=DEFAULT_DISTANCE_SCALE,
    max_objects=8,
):
    """Build the teacher-only spatial guidance block for one example.

    Args:
        captions: per-object caption strings for the scene.
        pointclouds: per-object clouds -- in-memory arrays, open3d
            geometries, or file paths into the fusion stage's output.
        question: the current query; objects it mentions are listed first
            with their pairwise separation (the "visual elements relevant
            to this query" of the guidance).
        scale: meters per raw cloud unit (see DEFAULT_DISTANCE_SCALE).
        max_objects: cap on unrelated objects listed; any excess is
            counted explicitly rather than dropped silently.

    Returns:
        The guidance block as a string, or '' when no object yields a
        usable cloud -- callers then fall back to the unguided prompt.
    """
    entries = _collect_objects(captions, pointclouds, scale)
    if not entries:
        return ""
    records = [record for record, _ in entries]
    points_by_index = {i: pts for i, (_, pts) in enumerate(entries)}

    focus = match_objects_in_text(question, records) if question else []

    lines = [
        "Spatial context measured from the scene's 3D reconstruction "
        "(privileged guidance; coordinates in meters at the scene's scale):"
    ]
    if focus:
        lines.append("Objects referenced by this question:")
        for i in focus:
            lines.append(_format_object(records[i]))
        if len(focus) >= 2 and focus[0] in points_by_index and focus[1] in points_by_index:
            separation = (
                _mean_surface_distance(points_by_index[focus[0]], points_by_index[focus[1]])
                * scale
            )
            lines.append(
                f'- mean surface distance between "{records[focus[0]]["description"]}" '
                f'and "{records[focus[1]]["description"]}": {separation:.2f} meters'
            )

    others = [i for i in range(len(records)) if i not in set(focus)]
    if others:
        lines.append("Other detected objects:")
        for i in others[:max_objects]:
            lines.append(_format_object(records[i]))
        hidden = len(others) - min(len(others), max_objects)
        if hidden > 0:
            lines.append(f"- ({hidden} more objects not listed)")

    return "\n".join(lines)
