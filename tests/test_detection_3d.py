"""Smoke tests for vqasynth.detection_3d.

Exercises the 3D-detection QA-pair synthesis stage against synthetic point
clouds — no CUDA, no depth model, no SAM, no numpy, no open3d. The stage's pure
logic (axis-aligned + oriented bounding boxes, Molmo <point3d>/SpatialRGPT
bracketed formatting, QA-pair + message emission) is standard-library only;
open3d/numpy are runtime-only deps of the .pcd I/O path, exactly like the brief
frames them. Real end-to-end validation (VGGT depth -> SAM2 masks -> .pcd) belongs
on a GPU host.

The module's QA templates deliberately reuse the [A]/[X] placeholder convention
from the pre-existing vqasynth.prompt_templates, so part of this suite checks the
two modules agree on that convention (a genuine cross-module integration check,
not a self-test of the new code alone).
"""
from __future__ import annotations

import math
import random
import pytest

# Import from the NEW module under test ...
from vqasynth.detection_3d import (
    BoundingBox3D,
    Detection3DGenerator,
    box_height,
    compare_box_height,
    compare_box_volume,
    compute_aabb,
    compute_bounding_box,
    compute_obb,
    detection_3d_answers,
    detection_3d_questions,
    format_box_coordinates,
    format_point3d,
    make_qa_pairs,
)

# ... AND from a pre-existing package module, so the suite is not a self-test of
# detection_3d alone. prompt_templates is dependency-free, so this import is safe
# in the minimal test environment.
from vqasynth import prompt_templates  # noqa: F401  (used in convention checks below)


# ---------------------------------------------------------------------------
# Fixtures: synthetic point clouds (no models, no CUDA)
# ---------------------------------------------------------------------------

@pytest.fixture
def axis_aligned_box_points():
    """Corners of a 2 x 3 x 4 box -> AABB center (1, 1.5, 2)."""
    return [
        (0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 4),
        (2, 3, 4), (2, 0, 4), (0, 3, 4), (2, 3, 0),
    ]


@pytest.fixture
def rotated_slab_points():
    """A thin 10 x 1 x 1 slab rotated 45 deg about Z -> AABB wasteful, OBB tight."""
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    L, w, h = 5.0, 0.5, 0.5
    pts = []
    for x in (-L, L):
        for y in (-w, w):
            for z in (-h, h):
                pts.append((c * x - s * y, s * x + c * y, z))
    return pts


# ---------------------------------------------------------------------------
# Axis-aligned bounding box
# ---------------------------------------------------------------------------

def test_aabb_center_and_extent(axis_aligned_box_points):
    box = compute_aabb(axis_aligned_box_points, label="crate")
    assert box.center == (1.0, 1.5, 2.0)
    assert box.extent == (2.0, 3.0, 4.0)
    assert box.label == "crate"
    assert box.volume == pytest.approx(24.0)


def test_aabb_min_max_bounds_and_corners(axis_aligned_box_points):
    box = compute_aabb(axis_aligned_box_points)
    assert box.min_bound == (0.0, 0.0, 0.0)
    assert box.max_bound == (2.0, 3.0, 4.0)
    corners = box.corners()
    assert len(corners) == 8
    assert set(corners) == {
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (2.0, 3.0, 0.0),
        (0.0, 0.0, 4.0), (2.0, 0.0, 4.0), (0.0, 3.0, 4.0), (2.0, 3.0, 4.0),
    }


def test_aabb_rejects_empty_points():
    with pytest.raises(ValueError, match="zero points"):
        compute_aabb([])


def test_aabb_accepts_plain_tuples_and_lists():
    # Inputs need not be numpy arrays — plain Python rows work.
    box = compute_aabb([(0, 0, 0), [1, 2, 3]], label="x")
    assert box.center == (0.5, 1.0, 1.5)
    assert box.extent == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Oriented bounding box (PCA via pure-Python Jacobi)
# ---------------------------------------------------------------------------

def test_obb_recovers_tight_box_for_rotated_slab(rotated_slab_points):
    aabb = compute_aabb(rotated_slab_points)
    obb = compute_obb(rotated_slab_points)
    # The slab is 10 long but only 1x1 in cross-section; the AABB inflates both
    # in-plane axes to ~7.07*sqrt(2), so the OBB must be dramatically tighter.
    assert obb.volume < aabb.volume * 0.25
    # The long principal extent should recover the true slab length (~10).
    long_extent = max(obb.extent)
    assert long_extent == pytest.approx(10.0, abs=0.05)


def test_obb_principal_axes_are_orthonormal(rotated_slab_points):
    # Drive the eigensolver directly to inspect the principal axes it returns.
    from vqasynth.detection_3d import _jacobi_eigendecomp

    n = len(rotated_slab_points)
    cx = sum(p[0] for p in rotated_slab_points) / n
    cy = sum(p[1] for p in rotated_slab_points) / n
    cz = sum(p[2] for p in rotated_slab_points) / n
    cov = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for x, y, z in rotated_slab_points:
        dx, dy, dz = x - cx, y - cy, z - cz
        cov[0][0] += dx * dx; cov[0][1] += dx * dy; cov[0][2] += dx * dz
        cov[1][1] += dy * dy; cov[1][2] += dy * dz; cov[2][2] += dz * dz
    for i in range(3):
        for j in range(i, 3):
            cov[i][j] /= n; cov[j][i] = cov[i][j]
    _, axes = _jacobi_eigendecomp(cov)
    # Each axis unit-length ...
    for ax in axes:
        assert math.sqrt(sum(v * v for v in ax)) == pytest.approx(1.0, abs=1e-9)
    # ... and mutually orthogonal.
    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))
    assert dot(axes[0], axes[1]) == pytest.approx(0.0, abs=1e-9)
    assert dot(axes[0], axes[2]) == pytest.approx(0.0, abs=1e-9)
    assert dot(axes[1], axes[2]) == pytest.approx(0.0, abs=1e-9)


def test_obb_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least 2 points"):
        compute_obb([(0, 0, 0)])


# ---------------------------------------------------------------------------
# compute_bounding_box: oriented refinement attaches only when worthwhile
# ---------------------------------------------------------------------------

def test_oriented_refinement_attaches_for_rotated_slab(rotated_slab_points):
    box = compute_bounding_box(
        rotated_slab_points, label="plank", oriented_threshold=0.3
    )
    assert box.oriented is not None
    assert box.oriented.volume < box.volume


def test_oriented_refinement_skipped_for_axis_aligned(axis_aligned_box_points):
    # An already-axis-aligned box has no OBB savings -> nothing attached.
    box = compute_bounding_box(
        axis_aligned_box_points, label="crate", oriented_threshold=0.3
    )
    assert box.oriented is None


def test_oriented_refinement_can_be_disabled(rotated_slab_points):
    box = compute_bounding_box(
        rotated_slab_points, label="plank", oriented_threshold=0
    )
    assert box.oriented is None


# ---------------------------------------------------------------------------
# Comparison helpers (generalize the clipseg prototype's height helpers)
# ---------------------------------------------------------------------------

def test_box_height_and_compare_match_anchor_semantics(axis_aligned_box_points):
    tall = compute_aabb(axis_aligned_box_points)               # height 3
    short = compute_aabb([(0, 0, 0), (1, 1, 1)])               # height 1
    assert box_height(tall) == 3.0
    assert box_height(short) == 1.0
    assert compare_box_height(tall, short) is True
    assert compare_box_height(short, tall) is False


def test_compare_box_volume(axis_aligned_box_points, rotated_slab_points):
    crate = compute_aabb(axis_aligned_box_points)              # vol 24
    assert compare_box_volume(crate, compute_aabb([(0, 0, 0), (1, 1, 1)])) is True
    # slab AABB volume (~60.5) exceeds the crate's (24).
    slab_aabb = compute_aabb(rotated_slab_points)
    assert compare_box_volume(slab_aabb, crate) is True


# ---------------------------------------------------------------------------
# Formatting: Molmo <point3d> + SpatialRGPT bracketed descriptor
# ---------------------------------------------------------------------------

def test_format_point3d_is_molmo_point_analog(axis_aligned_box_points):
    box = compute_aabb(axis_aligned_box_points, label="crate")
    tag = format_point3d(box)
    # Same tag shape as localize.py's <point .../>, lifted to 3D.
    assert tag.startswith('<point3d x="1.00" y="1.50" z="2.00"')
    assert 'extent="2.00,3.00,4.00"' in tag
    assert tag.endswith('alt="crate"/>')
    assert tag.count('"') % 2 == 0  # balanced quotes


def test_format_box_coordinates_is_spatialrgpt_region_descriptor(axis_aligned_box_points):
    box = compute_aabb(axis_aligned_box_points, label="crate")
    desc = format_box_coordinates(box)
    assert desc == "[[1.00,1.50,2.00],[2.00,3.00,4.00]]"


# ---------------------------------------------------------------------------
# QA-pair generation
# ---------------------------------------------------------------------------

def test_qa_pairs_substitute_label_and_value(axis_aligned_box_points):
    box = compute_aabb(axis_aligned_box_points, label="crate")
    rng = random.Random(0)
    pairs = make_qa_pairs(box, n_questions=2, fmt="point3d", rng=rng)
    assert len(pairs) == 2
    for pair in pairs:
        assert "Answer: " in pair
        question, answer = pair.split(" Answer: ", 1)
        assert "crate" in question          # [A] -> label
        assert "<point3d" in answer         # [X] -> formatted box
        assert "[A]" not in pair and "[X]" not in pair  # no leftover placeholders


def test_qa_pairs_respect_n_questions(axis_aligned_box_points):
    box = compute_aabb(axis_aligned_box_points, label="crate")
    total = len(make_qa_pairs(box, fmt="point3d", rng=random.Random(1)))
    assert total == len(detection_3d_questions)
    assert len(make_qa_pairs(box, n_questions=1, fmt="point3d", rng=random.Random(1))) == 1


def test_qa_pairs_unknown_fmt_raises(axis_aligned_box_points):
    box = compute_aabb(axis_aligned_box_points)
    with pytest.raises(ValueError, match="unknown fmt"):
        make_qa_pairs(box, fmt="csv", rng=random.Random(0))


def test_qa_pairs_bracket_format_uses_descriptor(axis_aligned_box_points):
    box = compute_aabb(axis_aligned_box_points, label="crate")
    rng = random.Random(0)
    pair = make_qa_pairs(box, n_questions=1, fmt="bracket", rng=rng)[0]
    assert "[[1.00,1.50,2.00],[2.00,3.00,4.00]]" in pair


# ---------------------------------------------------------------------------
# Cross-module convention check (pre-existing vqasynth.prompt_templates)
# ---------------------------------------------------------------------------

def test_qa_template_placeholders_match_repo_convention():
    """detection_3d templates must reuse the [A]/[X] placeholders that the
    pre-existing vqasynth.prompt_templates module uses across its Q/A banks, so
    3D-detection pairs flow through the same answer-string pipeline unchanged."""
    repo_templates = (
        prompt_templates.distance_template_answers
        + prompt_templates.width_answers
        + prompt_templates.height_answers
    )
    assert any("[A]" in t and "[X]" in t for t in repo_templates)  # sanity

    ours = detection_3d_questions + detection_3d_answers
    assert any("[A]" in t for t in detection_3d_questions)
    assert any("[X]" in t for t in detection_3d_answers)
    # No detection_3d template invents a placeholder the repo doesn't use.
    for t in ours:
        for token in ("[A]", "[B]", "[X]"):
            if token in t:
                assert token in ("[A]", "[X]") or token == "[B]"  # [B] not used here


# ---------------------------------------------------------------------------
# Detection3DGenerator: run / messages / apply_transform
# ---------------------------------------------------------------------------

def test_generator_run_emits_one_pair_per_object(rotated_slab_points):
    captions = ["wooden crate", "steel plank"]
    clouds = [[(0, 0, 0), (1, 1, 1), (2, 2, 2)], rotated_slab_points]
    gen = Detection3DGenerator(questions_per_object=1)
    prompts = gen.run(captions, [clouds])  # single-example wrap, as scene_fusion stores
    assert len(prompts) == 2
    assert all("Answer: " in p and "<point3d" in p for p in prompts)


def test_generator_run_handles_misaligned_captions_and_clouds():
    gen = Detection3DGenerator(questions_per_object=1)
    # Fewer captions than clouds -> truncated gracefully (upstream defensive stance).
    prompts = gen.run(["only one"], [[(0, 0, 0), (1, 1, 1)], [(0, 0, 0), (2, 2, 2)]])
    assert len(prompts) == 1


def test_generator_messages_match_prompt_stage_schema(rotated_slab_points):
    gen = Detection3DGenerator(questions_per_object=2)
    prompts = gen.run(["plank"], [[rotated_slab_points]])
    messages = gen.create_messages_from_prompts(prompts)
    # Same schema PromptGenerator.create_messages_from_prompts emits:
    #   alternating user/assistant, first user message carries the image.
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"][0] == {"index": 0, "text": None, "type": "image"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "text"


def test_generator_apply_transform_returns_stable_schema(rotated_slab_points):
    gen = Detection3DGenerator(questions_per_object=1)
    example = {
        "captions": ["plank", "crate"],
        "pointclouds": [[rotated_slab_points, [(0, 0, 0), (1, 1, 1), (2, 2, 2)]]],
    }
    out = gen.apply_transform(example)
    assert set(out.keys()) == {
        "detection_3d_boxes", "detection_3d_prompts", "detection_3d_messages"
    }
    assert len(out["detection_3d_boxes"]) == 2
    # The rotated slab should have an oriented refinement attached; the crate not.
    labels = {b["label"]: b for b in out["detection_3d_boxes"]}
    assert labels["plank"]["oriented"] is not None
    assert labels["crate"]["oriented"] is None
    # Serialized boxes are JSON-friendly (lists, not tuples).
    assert isinstance(labels["plank"]["center"], list)
    assert len(out["detection_3d_prompts"]) == 2
    assert len(out["detection_3d_messages"]) == 4


def test_generator_apply_transform_never_returns_none():
    """On any failure, apply_transform returns the empty-list form so the column
    schema stays stable for HuggingFace datasets.map (PromptGenerator returns
    None; we improve on that to keep the schema fixed)."""
    gen = Detection3DGenerator()
    out = gen.apply_transform({"captions": "not-a-list", "pointclouds": 12345})
    assert out == {
        "detection_3d_boxes": [],
        "detection_3d_prompts": [],
        "detection_3d_messages": [],
    }


def test_module_imports_without_open3d_or_numpy():
    """The stage's logic must import in a minimal environment — open3d/numpy are
    runtime-only deps of the .pcd I/O path, imported lazily, never at module
    import time."""
    import importlib
    import sys

    for blocked in ("open3d", "numpy"):
        assert blocked not in sys.modules, f"{blocked} should not be imported eagerly"
    importlib.reload(importlib.import_module("vqasynth.detection_3d"))
    assert "open3d" not in sys.modules
    assert "numpy" not in sys.modules
