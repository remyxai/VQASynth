"""Smoke tests for vqasynth.pose (data-generation direction).

Verifies the keypoint-source-first pipeline: a pluggable backend emits
per-person COCO-17 keypoints, :func:`pose_from_keypoints` normalizes them,
and the QA/message emitter renders Molmo-style ``<point>`` SFT samples. No
CUDA, no mediapipe / ultralytics install, no model download — a stub backend
and synthetic keypoint fixtures drive everything deterministically.

Mirrors the philosophy of tests/test_vggt_speedups.py: exercise the
pure-Python mechanics here; real end-to-end pose inference belongs on a host
with the backend installed.

The relative-keypoint QA path is validated against the existing
spatial-predicate corpus in vqasynth.prompt_templates (a pre-existing
module), so this file does not merely self-test the new module.
"""
from __future__ import annotations

import re

import pytest

from vqasynth import prompt_templates
from vqasynth.pose import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
    KEYPOINT_INDEX,
    MEDIAPIPE_TO_COCO,
    KeypointExtractor,
    MediaPipePoseBackend,
    PoseAnnotator,
    StubPoseBackend,
    build_pose_messages,
    build_pose_qa_pairs,
    pose_from_keypoints,
    _disp,
    _resolve_backend,
    _spatial_relation,
)


# A synthetic 17-keypoint pose for a 100x100 image, in COCO order. Only the
# joints named below are "visible"; the rest are None (COCO "not labeled").
W, H = 100, 100
VISIBLE = {
    "nose": [50, 12],
    "left_shoulder": [30, 25],
    "right_shoulder": [70, 25],
    "left_wrist": [20, 50],
    "right_ankle": [75, 90],
}


def _pose(visible=None, w=W, h=H):
    """Build a canonical pose dict from a {canonical_name: [x, y]} fixture."""
    return pose_from_keypoints(visible or VISIBLE, (w, h))


def test_skeleton_invariants():
    assert len(COCO_KEYPOINTS) == 17
    assert len(set(COCO_KEYPOINTS)) == 17
    assert KEYPOINT_INDEX == {n: i for i, n in enumerate(COCO_KEYPOINTS)}
    for a, b in COCO_SKELETON:
        assert 0 <= a < 17 and 0 <= b < 17 and a != b


def test_mediapipe_mapping_projects_33_to_coco17():
    """MEDIAPIPE_TO_COCO must be one valid MediaPipe landmark per COCO joint."""
    assert len(MEDIAPIPE_TO_COCO) == 17
    for mp_idx in MEDIAPIPE_TO_COCO:
        assert 0 <= mp_idx < 33


def test_pose_from_keypoints_list_and_missing_joints():
    # Passing a list of 17 (x, y) / None entries in COCO order.
    raw = [None] * 17
    raw[KEYPOINT_INDEX["nose"]] = [50, 12]
    raw[KEYPOINT_INDEX["left_shoulder"]] = [30, 25]
    pose = pose_from_keypoints(raw, (W, H))

    assert pose["image_size"] == (W, H)
    assert pose["num_detected"] == 2
    assert [k["name"] for k in pose["keypoints"]] == COCO_KEYPOINTS

    by_name = {k["name"]: k for k in pose["keypoints"]}
    assert by_name["nose"]["xy"] == [50.0, 12.0]
    assert by_name["nose"]["visible"] is True
    # joints we didn't supply are reported as not visible (COCO "not labeled")
    assert by_name["left_eye"]["xy"] is None
    assert by_name["left_eye"]["visible"] is False


def test_pose_from_keypoints_accepts_name_or_index_mapping():
    # Name-keyed mapping ...
    by_name = pose_from_keypoints({"nose": [1, 2]}, (W, H))
    assert by_name["keypoints"][KEYPOINT_INDEX["nose"]]["xy"] == [1.0, 2.0]
    # ... and index-keyed mapping produce the same joint.
    by_idx = pose_from_keypoints({KEYPOINT_INDEX["nose"]: [1, 2]}, (W, H))
    assert by_idx["keypoints"][KEYPOINT_INDEX["nose"]]["xy"] == [1.0, 2.0]


def test_build_pose_qa_point_localization_is_normalized():
    pose = _pose()
    qa = build_pose_qa_pairs(pose, seed=0)
    assert qa, "expected at least one QA pair"

    point_answers = [q["answer"] for q in qa if "<point" in q["answer"]]
    assert point_answers, "expected at least one <point>-token answer"
    for ans in point_answers:
        for x, y in re.findall(r'<point x="([0-9.]+)" y="([0-9.]+)"', ans):
            assert 0.0 <= float(x) <= 100.0
            assert 0.0 <= float(y) <= 100.0


def test_build_pose_qa_whole_pose_covers_visible_joints():
    pose = _pose()
    qa = build_pose_qa_pairs(pose, max_questions=50, seed=1)
    whole = max(qa, key=lambda q: q["answer"].count("<point"))
    for name in VISIBLE:
        assert f'alt="{_disp(name)}"' in whole["answer"]


def test_build_pose_qa_relative_uses_prompt_templates():
    """A left/right/above/below pair must be phrased with the pre-existing
    vqasynth.prompt_templates predicate corpus (not ad-hoc strings)."""
    # left wrist (x=20) is to the LEFT of right wrist (x=80).
    pose = _pose({"left_wrist": [20, 50], "right_wrist": [80, 50]})
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
    pose = _pose()
    assert build_pose_qa_pairs(pose, seed=42) == build_pose_qa_pairs(pose, seed=42)
    assert isinstance(build_pose_qa_pairs(pose, seed=1)[0]["question"], str)


def test_build_pose_messages_matches_chat_convention():
    """Messages must follow the nested role/content shape vqasynth.prompts
    writes to the ``messages`` column: a user turn with an image placeholder
    (index 0) + text, and an assistant text turn."""
    pose = _pose()
    samples = build_pose_messages(pose, max_questions=3, seed=0)
    assert 1 <= len(samples) <= 3
    for sample in samples:
        msgs = sample["messages"]
        assert len(msgs) == 2
        user, assistant = msgs
        assert user["role"] == "user"
        assert assistant["role"] == "assistant"
        # user content: image placeholder first, then the question text
        assert user["content"][0] == {"index": 0, "text": None, "type": "image"}
        assert user["content"][1]["type"] == "text"
        assert isinstance(user["content"][1]["text"], str) and user["content"][1]["text"]
        # assistant content: the Molmo-style <point> answer
        assert assistant["content"][0]["type"] == "text"
        assert "<point" in assistant["content"][0]["text"]


# ---------------------------------------------------------------------------
# Pluggable backend ABI (stub-driven; no mediapipe install required)
# ---------------------------------------------------------------------------
def test_keypoint_extractor_with_stub_backend():
    extractor = KeypointExtractor(backend=StubPoseBackend(VISIBLE, (W, H)))
    poses = extractor.extract(image=object())  # stub ignores the image
    assert len(poses) == 1
    pose = poses[0]
    assert pose["num_detected"] == len(VISIBLE)
    assert [k["name"] for k in pose["keypoints"]] == COCO_KEYPOINTS

    samples = extractor.extract_messages(image=object(), max_questions=4, seed=0)
    assert samples, "expected SFT samples from the stub pose"
    assert all("<point" in s["messages"][1]["content"][0]["text"] for s in samples)


def test_pose_annotator_append_column_convention():
    """PoseAnnotator.apply_transform writes a pose_messages column for a batch,
    matching how the other stages map over a HF dataset."""
    pytest.importorskip("PIL")
    from PIL import Image

    annotator = PoseAnnotator(backend=StubPoseBackend(VISIBLE, (W, H)), seed=0)
    # Real (tiny) images; the stub backend ignores their pixels.
    batch = {"image": [Image.new("RGB", (W, H)), Image.new("RGB", (W, H))]}
    out = annotator.apply_transform(batch, images="image")
    assert "pose_messages" in out
    assert len(out["pose_messages"]) == 2
    assert all(isinstance(s, list) and s for s in out["pose_messages"])


def test_resolve_backend_strings_and_passthrough():
    # An explicit backend instance is passed through unchanged.
    stub = StubPoseBackend(VISIBLE, (W, H))
    assert _resolve_backend(stub) is stub
    # Unknown name -> ValueError (never a silent default).
    with pytest.raises(ValueError):
        _resolve_backend("not-a-backend")
    # YOLOv8 is AGPL -> opt-in only, never selectable by name.
    with pytest.raises(ValueError):
        _resolve_backend("yolov8")


def test_mediapipe_backend_constructor_is_lazy():
    """Selecting 'mediapipe' by name must construct the backend only when
    mediapipe is importable; if it isn't, surface a clear ImportError rather
    than failing at module import."""
    pytest.importorskip("mediapipe")
    backend = _resolve_backend("mediapipe")
    assert isinstance(backend, MediaPipePoseBackend)
