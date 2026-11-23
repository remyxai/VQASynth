"""Smoke tests for vqasynth.camera_relations.

Exercises the camera-object relation QA stage (Ultimate3D taxonomy) against
synthetic point clouds — no CUDA, no depth model, no SAM, no numpy, no
open3d for the core suite. The binning math, geometry helpers, template
banks, and QA emission are standard-library only; open3d is a runtime-only
dep of the ``.pcd`` I/O path.

The module's QA templates deliberately reuse the [A]/[X] placeholder
convention from the pre-existing vqasynth.prompt_templates, so part of this
suite checks the two modules agree on that convention (a genuine
cross-module integration check, not a self-test of the new code alone).

The last section exercises the real call site — the wiring added to
``vqasynth.prompts.PromptGenerator.run``/``apply_transform``. vqasynth.promps
imports the VGGT model package at import time (via scene_fusion); when that
package is absent it is stubbed with inert modules so the wiring itself is
still testable without a GPU host (mirroring the guarded-integration
philosophy of tests/test_correspondence.py), and the end-to-end test writes
real .pcd files when open3d is importable.
"""
from __future__ import annotations

import math
import random
import sys
import types

import pytest

# Import from the NEW module under test ...
from vqasynth.camera_relations import (
    CLOSE_UP_MAX_RATIO,
    LONG_SHOT_MIN_RATIO,
    MIN_ORIENTATION_CONFIDENCE,
    ORIENTATION_CLASSES,
    SHOT_CLASSES,
    VIEWPOINT_CLASSES,
    CameraRelationGenerator,
    camera_distance,
    camera_elevation,
    camera_orientation_answers,
    camera_orientation_questions,
    camera_relation_prompts,
    camera_shot_answers,
    camera_shot_questions,
    camera_viewpoint_answers,
    camera_viewpoint_questions,
    classify_orientation,
    classify_shot,
    classify_viewpoint,
    make_camera_relation_qa,
    object_size,
)

# ... AND from pre-existing package modules, so the suite is not a self-test
# of camera_relations alone. prompt_templates is dependency-free, so this
# import is safe in the minimal test environment.
from vqasynth import prompt_templates  # noqa: F401  (convention checks below)


# ---------------------------------------------------------------------------
# Fixtures / helpers: synthetic point clouds (no models, no CUDA)
# ---------------------------------------------------------------------------

def cube_points(cx, cy, cz, side=1.0):
    """Corners + face centers + centroid of an axis-aligned cube."""
    r = side / 2.0
    pts = [
        (cx + dx * r, cy + dy * r, cz + dz * r)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    return pts


# A 1m cube centered at (0, 1, 2) in camera space (camera at the origin, +Y
# up, +Z into the scene): centroid elevation atan2(1, 2) ~= 26.57 deg
# (low-angle), distance sqrt(5) ~= 2.236 -> framing ratio 2.236 (medium shot).
LOW_ANGLE_MEDIUM_SHOT = cube_points(0, 1, 2)
# Centroid (0, -1, 20): elevation atan2(-1, 20) ~= -2.86 deg (eye-level),
# ratio sqrt(401) ~= 20.02 (long shot).
EYE_LEVEL_LONG_SHOT = cube_points(0, -1, 20)
# Centroid (0, -3, 1): elevation atan2(-3, 1) ~= -71.57 deg (high-angle),
# ratio sqrt(10) / 2 ~= 1.58 (medium shot).
HIGH_ANGLE_MEDIUM_SHOT = cube_points(0, -3, 1, side=2.0)
# Centroid (0, 0, 0.75), side 0.5: elevation 0 (eye-level), ratio 0.75 /
# 0.5 = 1.5 — sits exactly on the close-up/medium boundary.
EYE_LEVEL_BOUNDARY = cube_points(0, 0, 0.75, side=0.5)


# ---------------------------------------------------------------------------
# Taxonomy binning
# ---------------------------------------------------------------------------

def test_orientation_bins_cover_circle_and_boundaries():
    assert ORIENTATION_CLASSES == (
        "front",
        "front-right",
        "right",
        "back-right",
        "back",
        "back-left",
        "left",
        "front-left",
    )
    # Cardinal directions: azimuth 0 faces the camera, sectors run
    # counterclockwise (see module docstring).
    for azimuth, expected in [
        (0, "front"),
        (45, "front-right"),
        (90, "right"),
        (135, "back-right"),
        (180, "back"),
        (225, "back-left"),
        (270, "left"),
        (315, "front-left"),
    ]:
        assert classify_orientation(azimuth) == expected
    # Sector boundaries sit at 22.5 + k*45 degrees, values wrap modulo 360.
    assert classify_orientation(22.4) == "front"
    assert classify_orientation(22.6) == "front-right"
    assert classify_orientation(359.9) == "front"
    assert classify_orientation(-90) == "left"  # -90 % 360 == 270
    assert classify_orientation(720 + 90) == "right"


def test_viewpoint_bins():
    assert classify_viewpoint(30.0) == "low-angle"  # object above camera
    assert classify_viewpoint(-30.0) == "high-angle"  # camera looks down
    assert classify_viewpoint(0.0) == "eye-level"
    assert classify_viewpoint(15.0) == "eye-level"  # band is inclusive
    assert classify_viewpoint(15.1) == "low-angle"
    assert classify_viewpoint(-15.1) == "high-angle"


def test_shot_bins():
    assert classify_shot(1.0, 1.0) == "close-up"
    assert classify_shot(CLOSE_UP_MAX_RATIO, 1.0) == "medium shot"
    assert classify_shot(3.9, 1.0) == "medium shot"
    assert classify_shot(LONG_SHOT_MIN_RATIO, 1.0) == "long shot"
    assert classify_shot(10.0, 1.0) == "long shot"
    # Scale-free: same ratio in centimeters bins identically.
    assert (
        classify_shot(223.6, 100.0) == classify_shot(2.236, 1.0)
    )
    with pytest.raises(ValueError):
        classify_shot(1.0, 0.0)


# ---------------------------------------------------------------------------
# Camera-space geometry helpers
# ---------------------------------------------------------------------------

def test_geometry_from_synthetic_clouds():
    assert camera_elevation(LOW_ANGLE_MEDIUM_SHOT) == pytest.approx(
        math.degrees(math.atan2(1, 2))
    )
    assert camera_distance(LOW_ANGLE_MEDIUM_SHOT) == pytest.approx(
        math.sqrt(5.0)
    )
    assert object_size(HIGH_ANGLE_MEDIUM_SHOT) == pytest.approx(2.0)
    for pts, viewpoint, shot in [
        (LOW_ANGLE_MEDIUM_SHOT, "low-angle", "medium shot"),
        (EYE_LEVEL_LONG_SHOT, "eye-level", "long shot"),
        (HIGH_ANGLE_MEDIUM_SHOT, "high-angle", "medium shot"),
        (EYE_LEVEL_BOUNDARY, "eye-level", "medium shot"),
    ]:
        assert classify_viewpoint(camera_elevation(pts)) == viewpoint
        assert classify_shot(camera_distance(pts), object_size(pts)) == shot
    assert set(VIEWPOINT_CLASSES) == {"low-angle", "eye-level", "high-angle"}
    assert set(SHOT_CLASSES) == {"close-up", "medium shot", "long shot"}


def test_geometry_rejects_empty_clouds():
    for fn in (camera_elevation, camera_distance, object_size):
        with pytest.raises(ValueError):
            fn([])


# ---------------------------------------------------------------------------
# QA emission
# ---------------------------------------------------------------------------

def test_make_qa_format_labels_and_gating():
    orientation = {"azimuth": 90.0, "polar": 0.0, "rotation": 0.0,
                   "confidence": 0.9}
    rng = random.Random(0)
    pairs = make_camera_relation_qa(
        "Red Chair", LOW_ANGLE_MEDIUM_SHOT, orientation, rng=rng
    )
    # viewpoint + shot + orientation, in the repo's question/answer shape
    assert len(pairs) == 3
    assert all(" Answer: " in p for p in pairs)
    assert all("red chair" in p for p in pairs)  # caption lowercased
    joined = "\n".join(pairs)
    assert "low-angle" in joined
    assert "medium shot" in joined
    assert "right" in joined  # azimuth 90 -> object's right side to camera

    # Without an orientation estimate only viewpoint + shot are emitted.
    assert len(make_camera_relation_qa("Red Chair", LOW_ANGLE_MEDIUM_SHOT)) == 2

    # Low Orient-Anything confidence gates the orientation pair out.
    weak = dict(orientation, confidence=MIN_ORIENTATION_CONFIDENCE - 0.1)
    assert len(make_camera_relation_qa("Red Chair", LOW_ANGLE_MEDIUM_SHOT, weak)) == 2

    # Deterministic under a seeded rng.
    again = make_camera_relation_qa(
        "Red Chair", LOW_ANGLE_MEDIUM_SHOT, orientation, rng=random.Random(0)
    )
    assert again == pairs


def test_camera_relation_prompts_end_to_end_synthetic():
    captions = ["Red Mug", "Wooden Chair"]
    clouds = [LOW_ANGLE_MEDIUM_SHOT, EYE_LEVEL_LONG_SHOT]
    orientations = [None, {"azimuth": 225.0, "confidence": 0.8}]
    prompts = camera_relation_prompts(captions, clouds, orientations)
    # 2 pairs for the mug (no orientation), 3 for the chair.
    assert len(prompts) == 5
    assert all(" Answer: " in p for p in prompts)
    joined = "\n".join(prompts)
    for class_word in ("low-angle", "eye-level", "medium shot", "long shot",
                       "back-left"):
        assert class_word in joined
    # The single-example list wrap (as scene_fusion stores it) unwraps.
    wrapped = camera_relation_prompts([captions[0]], [clouds], None)
    assert len(wrapped) == 2


def test_camera_relation_prompts_never_raises_on_bad_objects(capsys):
    # The empty cloud raises inside object_centroid; the healthy object's
    # pairs must survive, and the error must be reported, not propagated.
    prompts = camera_relation_prompts(
        ["Red Mug", "Broken Cloud"],
        [LOW_ANGLE_MEDIUM_SHOT, []],
        None,
    )
    assert len(prompts) == 2
    assert "skipping" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Cross-module convention check (pre-existing vqasynth.prompt_templates)
# ---------------------------------------------------------------------------

def test_qa_template_placeholders_match_repo_convention():
    """camera_relations templates must reuse the [A]/[X] placeholders that the
    pre-existing vqasynth.prompt_templates module uses across its Q/A banks,
    so camera-relation pairs flow through the same answer-string pipeline
    (PromptGenerator.create_messages_from_prompts) unchanged."""
    repo_templates = (
        prompt_templates.distance_template_answers
        + prompt_templates.width_answers
        + prompt_templates.height_answers
    )
    assert any("[A]" in t and "[X]" in t for t in repo_templates)  # sanity

    ours = (
        camera_orientation_questions + camera_orientation_answers
        + camera_viewpoint_questions + camera_viewpoint_answers
        + camera_shot_questions + camera_shot_answers
    )
    assert any("[A]" in t for t in
               camera_orientation_questions + camera_viewpoint_questions
               + camera_shot_questions)
    assert any("[X]" in t for t in
               camera_orientation_answers + camera_viewpoint_answers
               + camera_shot_answers)
    # No camera_relations template invents a placeholder the repo doesn't
    # use: substituting [A]/[X] must leave no bracketed token behind.
    for t in ours:
        filled = t.replace("[A]", "mug").replace("[X]", "close-up")
        assert "[" not in filled and "]" not in filled, t


# ---------------------------------------------------------------------------
# CameraRelationGenerator: run / messages / apply_transform
# ---------------------------------------------------------------------------

def test_generator_run_and_messages():
    gen = CameraRelationGenerator()
    prompts = gen.run(
        ["Red Mug"], [LOW_ANGLE_MEDIUM_SHOT], [{"azimuth": 0.0,
                                                "confidence": 0.9}]
    )
    assert len(prompts) == 3
    assert "front" in "\n".join(prompts)
    messages = gen.create_messages_from_prompts(prompts)
    assert [m["role"] for m in messages] == ["user", "assistant"] * 3
    assert messages[0]["content"][0] == {
        "index": 0, "text": None, "type": "image"
    }
    # The orientation class word lands in one assistant answer whichever
    # answer template the (unseeded) choice picks.
    assistant_texts = [
        m["content"][0]["text"] for m in messages if m["role"] == "assistant"
    ]
    assert any("front" in t for t in assistant_texts)


def test_generator_apply_transform_adds_columns():
    gen = CameraRelationGenerator()
    example = {
        "captions": ["Red Mug"],
        "pointclouds": [[LOW_ANGLE_MEDIUM_SHOT]],
        "orientation": [None],
    }
    out = gen.apply_transform(example)
    assert set(out) == {"camera_relation_prompts", "camera_relation_messages"}
    assert len(out["camera_relation_prompts"]) == 2
    assert len(out["camera_relation_messages"]) == 4


def test_generator_apply_transform_degrades_to_empty():
    gen = CameraRelationGenerator()
    out = gen.apply_transform({"captions": [], "pointclouds": []})
    assert out == {"camera_relation_prompts": [],
                   "camera_relation_messages": []}


# ---------------------------------------------------------------------------
# The call site: vqasynth.prompts.PromptGenerator wiring
# ---------------------------------------------------------------------------

# What scene_fusion imports at module time from the VGGT package (absent on
# CPU-only test hosts). Inert stubs let the wiring be exercised anyway.
_VGGT_STUB_ATTRS = {
    "vggt": {},
    "vggt.models": {},
    "vggt.models.vggt": {"VGGT": object},
    "vggt.utils": {},
    "vggt.utils.pose_enc": {"pose_encoding_to_extri_intri": lambda *a, **k: None},
    "vggt.utils.geometry": {
        "unproject_depth_map_to_point_map": lambda *a, **k: None
    },
}


def _import_vqasynth_prompts():
    """Import vqasynth.prompts, stubbing only the VGGT package if missing.

    setdefault keeps real packages winning when present, so this is a no-op
    on a GPU host with VGGT installed.
    """
    try:
        import vqasynth.prompts as prompts_mod
        return prompts_mod
    except ModuleNotFoundError:
        for name, attrs in _VGGT_STUB_ATTRS.items():
            stub = types.ModuleType(name)
            for attr, value in attrs.items():
                setattr(stub, attr, value)
            sys.modules.setdefault(name, stub)
        try:
            import vqasynth.prompts as prompts_mod
            return prompts_mod
        except ModuleNotFoundError:
            return None


@pytest.fixture
def prompt_generator():
    before = set(sys.modules)
    prompts_mod = _import_vqasynth_prompts()
    if prompts_mod is None:
        pytest.skip("vqasynth.prompts not importable in this environment")
    yield prompts_mod.PromptGenerator()
    # Restore sys.modules so later test files see the environment they would
    # have seen without this suite (test_describe_anything legitimately
    # skips when vqasynth.prompts is unavailable; detection_3d's import-
    # hygiene test asserts on what sys.modules contains).
    for name in list(sys.modules):
        if name not in before and (
            name in _VGGT_STUB_ATTRS
            or name in ("vqasynth.prompts", "vqasynth.scene_fusion", "open3d")
        ):
            del sys.modules[name]


def _write_pcd(tmp_path, pts, name):
    o3d = pytest.importorskip("open3d")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    path = str(tmp_path / name)
    assert o3d.io.write_point_cloud(path, pcd), f"failed to write {path}"
    return path


def test_prompt_generator_run_appends_camera_relations(
    tmp_path, prompt_generator
):
    """The wiring edit: PromptGenerator.run must append camera-object
    relation QA to its spatial-predicate output, passing orientations
    through, with categorical answers kept out of the numeric filter."""
    path = _write_pcd(tmp_path, LOW_ANGLE_MEDIUM_SHOT, "mug.pcd")
    orientations = [{"azimuth": 90.0, "polar": 0.0, "rotation": 0.0,
                     "confidence": 0.9}]
    prompts = prompt_generator.run(
        ["a red chair"], [path], False, orientations=orientations
    )
    # One object -> no predicate pairs are possible, so the camera-relation
    # pairs are the entire output: viewpoint + shot + orientation.
    assert len(prompts) == 3
    joined = "\n".join(prompts)
    assert "low-angle" in joined
    assert "medium shot" in joined
    assert "right" in joined
    assert all(" Answer: " in p for p in prompts)
    messages = prompt_generator.create_messages_from_prompts(prompts)
    assert messages[0]["content"][0] == {
        "index": 0, "text": None, "type": "image"
    }


def test_prompt_generator_run_without_orientation_column(
    tmp_path, prompt_generator
):
    """Backward-compatible 3-arg call: no orientation column -> still
    viewpoint + shot QA from the point clouds alone."""
    path = _write_pcd(tmp_path, EYE_LEVEL_LONG_SHOT, "chair.pcd")
    prompts = prompt_generator.run(["a wooden chair"], [path], False)
    assert len(prompts) == 2
    joined = "\n".join(prompts)
    assert "eye-level" in joined and "long shot" in joined
    assert "facing the camera" not in joined  # no orientation QA


def test_prompt_generator_run_mixes_with_predicate_output(
    tmp_path, prompt_generator
):
    """Two objects: predicate pairs AND camera-relation pairs coexist."""
    paths = [
        _write_pcd(tmp_path, LOW_ANGLE_MEDIUM_SHOT, "mug.pcd"),
        _write_pcd(tmp_path, EYE_LEVEL_LONG_SHOT, "chair.pcd"),
    ]
    prompts = prompt_generator.run(
        ["a red mug", "a wooden chair"], paths, True, orientations=None
    )
    joined = "\n".join(prompts)
    assert len(prompts) > 4
    for class_word in ("low-angle", "eye-level", "medium shot", "long shot"):
        assert class_word in joined


def test_prompt_generator_apply_transform_passes_orientation_column(
    tmp_path, prompt_generator
):
    """apply_transform must forward the dataset's optional orientation
    column into run(); low-confidence entries only lose the orientation
    pair, not the viewpoint / shot pairs."""
    path = _write_pcd(tmp_path, LOW_ANGLE_MEDIUM_SHOT, "mug.pcd")
    example = {
        "captions": ["a red chair"],
        "pointclouds": [path],
        "is_canonicalized": False,
        "orientation": [
            {"azimuth": 90.0, "polar": 0.0, "rotation": 0.0,
             "confidence": MIN_ORIENTATION_CONFIDENCE - 0.1}
        ],
    }
    out = prompt_generator.apply_transform(example)
    assert len(out["prompts"]) == 2  # gated orientation, kept viewpoint/shot
    assert "low-angle" in "\n".join(out["prompts"])
    assert out["truncated_prompts"] <= out["prompts"]
    assert out["messages"] and out["messages"][0]["content"][0]["type"] == (
        "image"
    )
