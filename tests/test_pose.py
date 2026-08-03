"""Smoke tests for vqasynth.pose.

Verifies the pose-parsing, name-mapping and PoseText-style QA generation against
synthetic Molmo ``<point>`` output — no CUDA, no real Molmo install, no model
download. Mirrors the philosophy of tests/test_vggt_speedups.py: exercise the
pure-Python mechanics here; real end-to-end pose inference belongs on a GPU host.

The relative-keypoint QA path is validated against the existing spatial-predicate
corpus in vqasynth.prompt_templates (a pre-existing module), so this file does
not merely self-test the new module.
"""
from __future__ import annotations

import re

import pytest

from vqasynth import prompt_templates
from vqasynth.pose import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
    KEYPOINT_INDEX,
    build_pose_prompt,
    build_pose_qa_pairs,
    normalize_keypoint_name,
    parse_pose,
    _disp,
    _extract_points,
    _spatial_relation,
)


# Synthetic Molmo output for a 200x400 image. Mixes canonical names, synonyms
# (hand -> wrist), an unknown object (backpack, must be filtered), and a
# duplicate joint (second nose, first detection must win).
MO = (
    '<point x="50" y="10" alt="nose">'
    '<point x="30" y="25" alt="left shoulder">'
    '<point x="70" y="25" alt="Right Shoulder">'
    '<point x="20" y="50" alt="left hand">'
    '<point x="75" y="90" alt="right ankle">'
    '<point x="10" y="10" alt="a backpack">'
    '<point x="40" y="40" alt="nose">'
)
W, H = 200, 400


def _pose_from_xy(xy_by_name, w=100, h=100):
    """Build a pose dict directly from {canonical_name: [x, y]} for QA tests."""
    keypoints = []
    for idx, name in enumerate(COCO_KEYPOINTS):
        xy = xy_by_name.get(name)
        keypoints.append({"index": idx, "name": name, "xy": xy, "visible": xy is not None})
    return {
        "keypoints": keypoints,
        "image_size": (w, h),
        "num_detected": sum(1 for k in keypoints if k["visible"]),
    }


def test_skeleton_invariants():
    assert len(COCO_KEYPOINTS) == 17
    assert len(set(COCO_KEYPOINTS)) == 17
    assert KEYPOINT_INDEX == {n: i for i, n in enumerate(COCO_KEYPOINTS)}
    for a, b in COCO_SKELETON:
        assert 0 <= a < 17 and 0 <= b < 17 and a != b


@pytest.mark.parametrize("raw, expected", [
    ("nose", "nose"),
    ("Left Shoulder", "left_shoulder"),
    ("person's left knee", "left_knee"),
    ("r eye", "right_eye"),
    ("the nose", "nose"),
    ("right hand", "right_wrist"),       # hand -> wrist
    ("left foot", "left_ankle"),         # foot -> ankle
    ("left elbow", "left_elbow"),
    ("RIGHT_HIP", "right_hip"),
    ("a backpack", None),                # not a body joint
    ("", None),
    (None, None),
])
def test_normalize_keypoint_name(raw, expected):
    assert normalize_keypoint_name(raw) == expected


def test_extract_points_matches_molmo_output():
    """The shared/local parser returns every <point> token Molmo emitted."""
    pts = _extract_points(MO, W, H)
    # 7 <point> tokens in MO (incl. backpack + duplicate nose); all <= 100.
    assert len(pts) == 7
    nose = pts[0]
    assert nose["points"] == [100.0, 40.0]   # 50/100*200, 10/100*400
    assert nose["caption"] == "nose"


def test_parse_pose_maps_named_joints():
    pose = parse_pose(MO, W, H)
    assert pose["num_detected"] == 5
    assert pose["image_size"] == (W, H)

    by_name = {k["name"]: k for k in pose["keypoints"]}
    assert by_name["nose"]["xy"] == [100.0, 40.0]              # first detection wins
    assert by_name["left_shoulder"]["xy"] == [60.0, 100.0]
    assert by_name["right_shoulder"]["xy"] == [140.0, 100.0]
    assert by_name["left_wrist"]["xy"] == [40.0, 200.0]        # "left hand" mapped
    assert by_name["right_ankle"]["xy"] == [150.0, 360.0]

    # Ordering + missing joints reported as not visible (COCO "not labeled").
    assert [k["name"] for k in pose["keypoints"]] == COCO_KEYPOINTS
    assert by_name["left_eye"]["visible"] is False
    assert by_name["left_eye"]["xy"] is None


def test_parse_pose_agrees_with_shared_localize_parser():
    """When sam2/transformers are installed, parse_pose must reuse the real
    vqasynth.localize.extract_points_and_descriptions and agree with it."""
    pytest.importorskip("sam2")
    from vqasynth.localize import extract_points_and_descriptions
    assert _extract_points(MO, W, H) == extract_points_and_descriptions(MO, W, H)


def test_build_pose_prompt_lists_all_joints():
    prompt = build_pose_prompt()
    for name in COCO_KEYPOINTS:
        assert _disp(name) in prompt
    assert "0 to 100" in prompt
    assert 'alt="joint name"' in prompt


def test_build_pose_qa_point_localization_is_normalized():
    pose = parse_pose(MO, W, H)
    qa = build_pose_qa_pairs(pose, seed=0)
    assert qa, "expected at least one QA pair"

    point_answers = [a for q in qa for a in [q["answer"]] if "<point" in a]
    assert point_answers, "expected at least one <point>-token answer"
    for ans in point_answers:
        for x, y in re.findall(r'<point x="([0-9.]+)" y="([0-9.]+)"', ans):
            assert 0.0 <= float(x) <= 100.0
            assert 0.0 <= float(y) <= 100.0


def test_build_pose_qa_whole_pose_covers_visible_joints():
    pose = parse_pose(MO, W, H)
    qa = build_pose_qa_pairs(pose, max_questions=50, seed=1)
    whole = max(qa, key=lambda q: q["answer"].count("<point"))
    for name in ["nose", "left_shoulder", "right_shoulder", "left_wrist", "right_ankle"]:
        assert f'alt="{_disp(name)}"' in whole["answer"]


def test_build_pose_qa_relative_uses_prompt_templates():
    """A left/right/above/below pair must be phrased with the pre-existing
    vqasynth.prompt_templates predicate corpus (not ad-hoc strings)."""
    # left wrist (x=20) is to the LEFT of right wrist (x=80).
    pose = _pose_from_xy({"left_wrist": [20, 50], "right_wrist": [80, 50]}, w=100, h=100)
    qa = build_pose_qa_pairs(pose, max_questions=50, seed=0)

    found = False
    for q in qa:
        de = (
            q["answer"]
            .replace("left wrist", "[A]")
            .replace("right wrist", "[B]")
        )
        if de in prompt_templates.left_true_responses:
            found = True
            break
    assert found, (
        "expected a left-relation answer drawn from prompt_templates."
        "left_true_responses; got: " + repr([q["answer"] for q in qa])
    )


@pytest.mark.parametrize("a, b, rel", [
    ([20, 50], [80, 50], "left"),
    ([80, 50], [20, 50], "right"),
    ([50, 20], [50, 80], "above"),
    ([50, 80], [50, 20], "below"),
    ([50, 50], [50, 50], "same"),
])
def test_spatial_relation(a, b, rel):
    assert _spatial_relation(a, b) == rel


def test_build_pose_qa_is_deterministic_with_seed():
    pose = parse_pose(MO, W, H)
    assert build_pose_qa_pairs(pose, seed=42) == build_pose_qa_pairs(pose, seed=42)
    # different seed -> at least plausibly different ordering/content possible
    assert isinstance(build_pose_qa_pairs(pose, seed=1)[0]["question"], str)
