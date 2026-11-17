"""CPU tests for the perspective-taking / cross-view QA families in
vqasynth.prompts.

Mirrors the philosophy of ``tests/test_orientation.py``: verify the stage's
logic against synthetic inputs with KNOWN camera extrinsics — no CUDA, no
VGGT weights, no model download. The perspective transform is pure numpy and
is checked against hand-computed ground truth (an object that is right in the
world frame must be reported left from a camera yawed 180 degrees, and the
allocentric vs egocentric answers must differ).

``vqasynth.prompts`` imports ``vqasynth.scene_fusion`` (for
``restore_pointclouds``), which imports the Docker-only ``vggt`` package at
module import time. That one package is stubbed below so the *real*
PromptGenerator runs unmodified; open3d / numpy / torch are the real installs.
Real end-to-end validation with VGGT-fused multi-view scenes belongs on a GPU
host (GPU-validation note: run the prompt stage over a >=2-frame sample and
confirm "camera N" prompts appear and flip correctly across viewpoints).
"""
from __future__ import annotations

import importlib.util
import random
import re
import sys
import types

import numpy as np
import pytest
import open3d as o3d


# ---------------------------------------------------------------------------
# vggt stub — the only dependency on vqasynth.prompts' import path that is
# absent outside the Docker image. Attributes are never touched by these
# tests; they only need to exist for the module-level import to succeed.
# ---------------------------------------------------------------------------
def _install_vggt_stub():
    if importlib.util.find_spec("vggt") is not None:
        return  # real install present, nothing to stub

    vggt = types.ModuleType("vggt")
    vggt.__path__ = []
    models = types.ModuleType("vggt.models")
    models.__path__ = []
    models_vggt = types.ModuleType("vggt.models.vggt")
    models_vggt.VGGT = object
    utils = types.ModuleType("vggt.utils")
    utils.__path__ = []
    pose_enc = types.ModuleType("vggt.utils.pose_enc")
    pose_enc.pose_encoding_to_extri_intri = None
    geometry = types.ModuleType("vggt.utils.geometry")
    geometry.unproject_depth_map_to_point_map = None

    sys.modules.update(
        {
            "vggt": vggt,
            "vggt.models": models,
            "vggt.models.vggt": models_vggt,
            "vggt.utils": utils,
            "vggt.utils.pose_enc": pose_enc,
            "vggt.utils.geometry": geometry,
        }
    )


_install_vggt_stub()

from vqasynth.prompts import PromptGenerator
# Pre-existing module in the package — exercised for real below (slot-bank
# conventions + bank-membership checks), so this suite is not a self-test of
# the new code alone.
from vqasynth import prompt_templates


# ---------------------------------------------------------------------------
# Fixtures: known world-frame geometry + known camera poses.
# OpenCV convention throughout (x right, y down, z forward), matching VGGT's
# pose_encoding_to_extri_intri output and parse_extrinsics' contract.
# ---------------------------------------------------------------------------
IDENTITY_POSE = np.hstack([np.eye(3), np.zeros((3, 1))])  # at origin, looking +z
YAW_180_POSE = np.hstack(
    [np.diag([-1.0, 1.0, -1.0]), np.zeros((3, 1))]  # 180° yaw about y at origin
)
TWO_VIEW_EXTRINSICS = np.stack([IDENTITY_POSE, YAW_180_POSE])  # (2, 3, 4)


def _cloud_at(center):
    """Single-point cloud whose get_center() is exactly ``center``."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(center, dtype=np.float64)[None, :]
    )
    return pcd


def _pair_at(a_center, b_center, a_desc="red mug", b_desc="blue cup"):
    return (a_desc, _cloud_at(a_center)), (b_desc, _cloud_at(b_center))


def _filled(bank, a="red mug", b="blue cup", v=None, w=None, x=None):
    """Instantiate a template bank the way PromptGenerator substitutes slots."""
    out = []
    for t in bank:
        t = t.replace("[A]", a).replace("[B]", b)
        if v is not None:
            t = t.replace("[V]", v)
        if w is not None:
            t = t.replace("[W]", w)
        if x is not None:
            t = t.replace("[X]", x)
        out.append(t)
    return out


def _names_viewpoint(prompt):
    """True for prompts emitted by the viewpoint families.

    Every such prompt names a concrete camera ("camera 2" / "the camera 1
    view"). Plain substring "camera" is NOT a safe discriminator — the
    pre-existing front/behind question banks say "further to camera compared
    to" — but "camera <digit>" only occurs in substituted viewpoint labels.
    """
    return re.search(r"camera \d", prompt) is not None


# ---------------------------------------------------------------------------
# parse_extrinsics — input normalization / gating
# ---------------------------------------------------------------------------
def test_parse_extrinsics_accepts_supported_shapes():
    gen = PromptGenerator()

    poses = gen.parse_extrinsics(TWO_VIEW_EXTRINSICS)
    assert isinstance(poses, np.ndarray)
    assert poses.shape == (2, 3, 4)

    # nested lists (how a HF-datasets column arrives) and (S, 4, 4) both work
    assert gen.parse_extrinsics(TWO_VIEW_EXTRINSICS.tolist()).shape == (2, 3, 4)
    poses_44 = np.stack(
        [np.vstack([p, [0.0, 0.0, 0.0, 1.0]]) for p in TWO_VIEW_EXTRINSICS]
    )
    assert np.allclose(gen.parse_extrinsics(poses_44), TWO_VIEW_EXTRINSICS)


def test_parse_extrinsics_rejects_unusable_inputs():
    gen = PromptGenerator()

    assert gen.parse_extrinsics(None) is None
    assert gen.parse_extrinsics(IDENTITY_POSE) is None  # single pose < 2 frames
    assert gen.parse_extrinsics([IDENTITY_POSE]) is None
    assert gen.parse_extrinsics(np.full((2, 3, 4), np.nan)) is None
    assert gen.parse_extrinsics(np.eye(3)[None, ...]) is None  # rotation-only
    assert gen.parse_extrinsics([{"rotation": np.eye(3)}]) is None


# ---------------------------------------------------------------------------
# transform_points_to_camera — the perspective transform vs hand-computed GT
# ---------------------------------------------------------------------------
def test_transform_points_to_camera_matches_ground_truth():
    gen = PromptGenerator()

    a_world = np.array([-1.0, 0.0, 5.0])
    b_world = np.array([1.0, 0.0, 5.0])

    # identity camera: camera frame == world frame
    assert np.allclose(gen.transform_points_to_camera(a_world, IDENTITY_POSE), a_world)

    # 180° yaw about y mirrors x and z: left becomes right, near becomes far
    assert np.allclose(
        gen.transform_points_to_camera(a_world, YAW_180_POSE), [1.0, 0.0, -5.0]
    )
    assert np.allclose(
        gen.transform_points_to_camera(b_world, YAW_180_POSE), [-1.0, 0.0, -5.0]
    )

    # batched (N, 3) form and 4x4 pose form
    batch = gen.transform_points_to_camera(
        np.stack([a_world, b_world]),
        np.vstack([YAW_180_POSE, [0.0, 0.0, 0.0, 1.0]]),
    )
    assert np.allclose(batch, [[1.0, 0.0, -5.0], [-1.0, 0.0, -5.0]])


# ---------------------------------------------------------------------------
# Perspective-taking predicates — the brief's acceptance bar
# ---------------------------------------------------------------------------
def test_perspective_left_matches_ground_truth_and_flips_across_views():
    gen = PromptGenerator()
    # A is LEFT of B in the world frame (x: -1 < 1); both ~5m in front of camera 1.
    A, B = _pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])

    # camera 1 (identity pose): egocentric order == allocentric order
    question, answer = gen.perspective_left_predicate(
        A, B, 0, TWO_VIEW_EXTRINSICS
    ).split(" Answer: ")
    assert question in _filled(
        prompt_templates.perspective_left_predicate_questions, v="camera 1"
    )
    assert answer in _filled(
        prompt_templates.perspective_left_true_responses, v="camera 1"
    )

    # camera 2 (180° yaw): the world-left object is NOT left from here
    question, answer = gen.perspective_left_predicate(
        A, B, 1, TWO_VIEW_EXTRINSICS
    ).split(" Answer: ")
    assert question in _filled(
        prompt_templates.perspective_left_predicate_questions, v="camera 2"
    )
    assert answer in _filled(
        prompt_templates.perspective_left_false_responses, v="camera 2"
    )


def test_world_right_object_reported_left_from_rotated_camera():
    """The brief's literal example: right in the world -> left from a camera
    rotated 180 degrees."""
    gen = PromptGenerator()
    # B is RIGHT of A in the world frame.
    A, B = _pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])

    # allocentric (world) predicate: B is not left of A
    _, world_answer = gen.left_predicate(B, A).split(" Answer: ")
    assert world_answer in _filled(
        prompt_templates.left_false_responses, a="blue cup", b="red mug"
    )

    # egocentric (camera 2, 180° yaw): B IS left of A
    _, view_answer = gen.perspective_left_predicate(
        B, A, 1, TWO_VIEW_EXTRINSICS
    ).split(" Answer: ")
    assert view_answer in _filled(
        prompt_templates.perspective_left_true_responses,
        a="blue cup",
        b="red mug",
        v="camera 2",
    )
    # ... and the allocentric vs egocentric answers differ as expected
    assert world_answer.split()[0].rstrip(",") != view_answer.split()[0].rstrip(",")


def test_perspective_right_predicate_flips_symmetrically():
    gen = PromptGenerator()
    A, B = _pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])

    # A is not right of B in the world / from camera 1 ...
    _, answer = gen.perspective_right_predicate(A, B, 0, TWO_VIEW_EXTRINSICS).split(
        " Answer: "
    )
    assert answer in _filled(
        prompt_templates.perspective_right_false_responses, v="camera 1"
    )
    # ... but IS right of B from the 180° camera.
    _, answer = gen.perspective_right_predicate(A, B, 1, TWO_VIEW_EXTRINSICS).split(
        " Answer: "
    )
    assert answer in _filled(
        prompt_templates.perspective_right_true_responses, v="camera 2"
    )


def test_perspective_front_behind_follow_camera_depth_axis():
    gen = PromptGenerator()
    # A camera standing 10m back from two objects at z = 3 and z = 7, looking +z:
    # p_cam = p_world + (0, 0, 10) -> both in front, A nearer.
    back_pose = np.hstack([np.eye(3), [[0.0], [0.0], [10.0]]])
    # Same camera translated to z = 10 and yawed 180° (now looking -z):
    # z_cam = -z_world + 10 -> both in front, order flipped.
    back_yaw_pose = np.hstack(
        [np.diag([-1.0, 1.0, -1.0]), [[0.0], [0.0], [10.0]]]
    )
    extr = np.stack([back_pose, back_yaw_pose])

    A, B = _pair_at([0.0, 0.0, 3.0], [0.0, 0.0, 7.0])

    _, answer = gen.perspective_front_predicate(A, B, 0, extr).split(" Answer: ")
    assert answer in _filled(
        prompt_templates.perspective_front_true_responses, v="camera 1"
    )
    _, answer = gen.perspective_behind_predicate(B, A, 0, extr).split(" Answer: ")
    assert answer in _filled(
        prompt_templates.perspective_behind_true_responses,
        a="blue cup",
        b="red mug",
        v="camera 1",
    )
    # from the flipped camera the depth order reverses: A is no longer in front
    _, answer = gen.perspective_front_predicate(A, B, 1, extr).split(" Answer: ")
    assert answer in _filled(
        prompt_templates.perspective_front_false_responses, v="camera 2"
    )


def test_perspective_left_choice_selects_view_correct_object():
    gen = PromptGenerator()
    A, B = _pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])

    # world / camera 1: the red mug is more to the left
    _, answer = gen.perspective_left_choice(A, B, 0, TWO_VIEW_EXTRINSICS).split(
        " Answer: "
    )
    assert answer in _filled(
        prompt_templates.perspective_left_choice_responses, v="camera 1", x="red mug"
    )
    assert "red mug" in answer and "blue cup" not in answer

    # camera 2 (180° yaw): the blue cup is more to the left
    _, answer = gen.perspective_left_choice(A, B, 1, TWO_VIEW_EXTRINSICS).split(
        " Answer: "
    )
    assert answer in _filled(
        prompt_templates.perspective_left_choice_responses, v="camera 2", x="blue cup"
    )
    assert "blue cup" in answer and "red mug" not in answer


# ---------------------------------------------------------------------------
# Cross-view / frame-of-reference
# ---------------------------------------------------------------------------
def test_cross_view_order_allocentric_vs_egocentric():
    gen = PromptGenerator()
    A, B = _pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])
    world = "scene's overall layout"

    # camera 1 shares the world axes: orders agree
    question, answer = gen.cross_view_order_predicate(
        A, B, 0, None, TWO_VIEW_EXTRINSICS
    ).split(" Answer: ")
    assert question in _filled(
        prompt_templates.cross_view_order_questions, v="camera 1 view", w=world
    )
    assert answer in _filled(
        prompt_templates.cross_view_order_true_responses, v="camera 1 view", w=world
    )

    # camera 2 (180° yaw) vs the world layout: the order flips
    question, answer = gen.cross_view_order_predicate(
        A, B, 1, None, TWO_VIEW_EXTRINSICS
    ).split(" Answer: ")
    assert question in _filled(
        prompt_templates.cross_view_order_questions, v="camera 2 view", w=world
    )
    assert answer in _filled(
        prompt_templates.cross_view_order_false_responses, v="camera 2 view", w=world
    )


def test_cross_view_order_between_two_egocentric_views():
    gen = PromptGenerator()
    A, B = _pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])

    # camera 2 vs camera 1 disagree; camera 1 vs itself trivially agrees
    _, answer = gen.cross_view_order_predicate(
        A, B, 1, 0, TWO_VIEW_EXTRINSICS
    ).split(" Answer: ")
    assert answer in _filled(
        prompt_templates.cross_view_order_false_responses,
        v="camera 2 view",
        w="camera 1 view",
    )
    _, answer = gen.cross_view_order_predicate(
        A, B, 0, 0, TWO_VIEW_EXTRINSICS
    ).split(" Answer: ")
    assert answer in _filled(
        prompt_templates.cross_view_order_true_responses,
        v="camera 1 view",
        w="camera 1 view",
    )


# ---------------------------------------------------------------------------
# Registration + multi-frame gating (single-frame back-compat)
# ---------------------------------------------------------------------------
def test_build_viewpoint_variants_gates_on_frame_count():
    gen = PromptGenerator()
    A, B = _pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])

    # no extrinsics / a single pose -> no viewpoint families at all
    assert gen.build_viewpoint_variants(None) == []
    assert gen.build_viewpoint_variants(IDENTITY_POSE) == []

    variants = gen.build_viewpoint_variants(TWO_VIEW_EXTRINSICS)
    assert len(variants) == 6

    results = [fn(A, B) for fn in variants]
    assert all(" Answer: " in s for s in results)
    # every emitted prompt names the concrete viewpoint it was judged from
    assert all(("camera 1" in s) or ("camera 2" in s) for s in results)


def test_evaluate_predicates_on_pairs_adds_viewpoint_prompts_only_multi_frame():
    gen = PromptGenerator()
    pairs = [_pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])]

    def union_over_seeds(extr):
        seen = set()
        for seed in range(12):
            random.seed(seed)
            seen.update(gen.evaluate_predicates_on_pairs(pairs, True, extrinsics=extr))
        return seen

    with_views = union_over_seeds(TWO_VIEW_EXTRINSICS)
    assert with_views
    assert any(_names_viewpoint(s) for s in with_views)

    without_views = union_over_seeds(None)
    assert without_views
    assert not any(_names_viewpoint(s) for s in without_views)


def test_apply_transform_threads_optional_extrinsics_column(tmp_path):
    gen = PromptGenerator()

    a_path = str(tmp_path / "object_a.pcd")
    b_path = str(tmp_path / "object_b.pcd")
    o3d.io.write_point_cloud(a_path, _cloud_at([-1.0, 0.0, 5.0]))
    o3d.io.write_point_cloud(b_path, _cloud_at([1.0, 0.0, 5.0]))

    base_example = {
        "captions": ["red mug", "blue cup"],
        "pointclouds": [[a_path, b_path]],  # single-example wrap, as scene_fusion stores
        "is_canonicalized": [True],
    }

    def union_over_seeds(example):
        prompts = set()
        for seed in range(12):
            random.seed(seed)
            out = gen.apply_transform(dict(example))
            assert out is not None
            prompts.update(out["prompts"])
        return prompts

    # multi-frame sample: the optional column reaches the viewpoint families
    with_views = union_over_seeds(
        {**base_example, "extrinsics": TWO_VIEW_EXTRINSICS.tolist()}
    )
    assert any(_names_viewpoint(p) for p in with_views)

    # single-frame sample (column absent entirely): unchanged behavior
    without_views = union_over_seeds(dict(base_example))
    assert without_views
    assert not any(_names_viewpoint(p) for p in without_views)


# ---------------------------------------------------------------------------
# Template-bank conventions (cross-module check against the pre-existing
# prompt_templates, in the spirit of tests/test_detection_3d.py)
# ---------------------------------------------------------------------------
def test_new_banks_follow_slot_conventions():
    predicate_question_banks = [
        prompt_templates.perspective_left_predicate_questions,
        prompt_templates.perspective_right_predicate_questions,
        prompt_templates.perspective_front_predicate_questions,
        prompt_templates.perspective_behind_predicate_questions,
    ]
    for bank in predicate_question_banks:
        for template in bank:
            assert "[A]" in template and "[B]" in template and "[V]" in template

    assert all(
        "[X]" in t and "[V]" in t
        for t in prompt_templates.perspective_left_choice_responses
    )

    for template in prompt_templates.cross_view_order_questions:
        assert "[A]" in template and "[B]" in template
        assert "[V]" in template and "[W]" in template

    # true/false banks stay symmetric so both answers are always expressible
    assert len(prompt_templates.perspective_left_true_responses) == len(
        prompt_templates.perspective_left_false_responses
    )
    assert len(prompt_templates.cross_view_order_true_responses) == len(
        prompt_templates.cross_view_order_false_responses
    )


def test_perspective_predicates_reject_missing_pose():
    gen = PromptGenerator()
    A, B = _pair_at([-1.0, 0.0, 5.0], [1.0, 0.0, 5.0])

    with pytest.raises(ValueError):
        gen.perspective_left_predicate(A, B, 0, None)
    with pytest.raises(ValueError):
        gen.perspective_left_predicate(A, B, 5, TWO_VIEW_EXTRINSICS)  # bad frame idx
    with pytest.raises(ValueError):
        gen.cross_view_order_predicate(A, B, 0, None, None)
