"""Tests for vqasynth.privileged_context and its R1Reasoner wiring.

Mirrors the philosophy of ``tests/test_correspondence.py``: exercise the
privileged-context pathway without open3d, a GPU, or network access. Point
clouds are plain nested lists (the duck-typed in-memory path), and the
teacher call is captured by a stubbed OpenAI client so the tests can
assert exactly what the teacher saw versus what the student-visible
fields contain — the on-policy self-distillation property this wiring
exists to enforce.
"""
from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from vqasynth.privileged_context import (
    build_spatial_context,
    match_objects_in_text,
    object_spatial_records,
)
from vqasynth.r1_reasoning import R1Reasoner

CHAIR_POINTS = [[0.0, 0.0, 1.0], [0.2, 0.4, 1.2], [0.1, 0.2, 1.4]]
TABLE_POINTS = [[2.0, 0.0, 2.0], [2.4, 0.3, 2.2], [2.2, 0.1, 2.4]]
LAMP_POINTS = [[-1.0, 1.5, 0.5], [-1.1, 1.6, 0.6]]
CAPTIONS = ["wooden chair", "table"]


def _messages(question="How far is the wooden chair from the table?", answer="about 2 meters"):
    return [
        {
            "role": "user",
            "content": [
                {"index": 0, "text": None, "type": "image"},
                {"index": None, "text": question, "type": "text"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"index": None, "text": answer, "type": "text"}],
        },
    ]


class _StubCompletions:
    def __init__(self, captured):
        self._captured = captured

    def create(self, model, messages):
        self._captured.append(messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="stub reasoning"))]
        )


class _StubClient:
    """Records every chat.completions.create payload the teacher sends."""

    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=_StubCompletions(self.prompts))


def _reasoner():
    reasoner = R1Reasoner(
        api_key="test-key",
        model="stub-model",
        image_column="image",
        text_column="messages",
    )
    reasoner.client = _StubClient()
    return reasoner


def _sent_text(reasoner, call=0):
    """The text part of the user message from the given teacher call."""
    return reasoner.client.prompts[call][0]["content"][0]["text"]


class TestSpatialContext:
    def test_records_pair_captions_with_metric_geometry(self):
        records = object_spatial_records(CAPTIONS, [CHAIR_POINTS, TABLE_POINTS])
        assert [r["description"] for r in records] == CAPTIONS
        chair = records[0]
        # centers/extents are means and max-min of the raw points, scaled to
        # the same meters-per-unit factor the Q/A answers use (x10 here).
        assert float(chair["center"][0]) == 0.1 * 10
        assert float(chair["center"][2]) == 1.2 * 10
        assert float(chair["extent"][1]) == 0.4 * 10
        assert float(chair["extent"][2]) == 0.4 * 10

    def test_records_skip_unusable_entries(self):
        records = object_spatial_records(["ok", None, "no cloud"], [CHAIR_POINTS, TABLE_POINTS, None])
        assert [r["description"] for r in records] == ["ok"]

    def test_match_prefers_longest_description(self):
        records = object_spatial_records(
            ["box", "cardboard box on a pallet"], [CHAIR_POINTS, TABLE_POINTS]
        )
        # "box" is a substring of the longer caption, so both match — but the
        # more specific description must be ranked first.
        assert match_objects_in_text("where is the cardboard box on a pallet?", records) == [1, 0]
        assert match_objects_in_text("nothing mentioned", records) == []

    def test_build_context_focuses_question_objects_with_separation(self):
        question = "How far is the wooden chair from the table?"
        context = build_spatial_context(CAPTIONS, [CHAIR_POINTS, TABLE_POINTS], question)
        assert context.startswith("Spatial context measured from the scene's 3D reconstruction")
        assert "Objects referenced by this question:" in context
        assert '"wooden chair"' in context and '"table"' in context
        assert "mean surface distance" in context
        # Both objects are referenced, so none land in the leftovers section.
        assert "Other detected objects:" not in context

    def test_build_context_lists_unreferenced_objects_separately(self):
        question = "How far is the wooden chair from the table?"
        context = build_spatial_context(
            CAPTIONS + ["a lamp"], [CHAIR_POINTS, TABLE_POINTS, LAMP_POINTS], question
        )
        focus, separator, others = context.partition("Other detected objects:")
        assert separator
        assert '"a lamp"' in others
        assert '"a lamp"' not in focus

    def test_build_context_empty_without_usable_scene(self):
        assert build_spatial_context(None, None, "any question") == ""
        assert build_spatial_context([], [], "any question") == ""
        assert build_spatial_context(["a chair"], [None], "any question") == ""


class TestR1ReasonerWiring:
    def test_run_conditions_teacher_on_privileged_context(self):
        reasoner = _reasoner()
        context = build_spatial_context(CAPTIONS, [CHAIR_POINTS, TABLE_POINTS], "q")
        reasoner.run("question?", "answer", Image.new("RGB", (2, 2)), context)
        sent = _sent_text(reasoner)
        assert context in sent
        # Measured facts replace the legacy hedge about incorrect information.
        assert "Some information may be partially incorrect" not in sent
        assert "reliable ground truth" in sent

    def test_run_keeps_legacy_prompt_without_context(self):
        reasoner = _reasoner()
        reasoner.run("question?", "answer", Image.new("RGB", (2, 2)))
        sent = _sent_text(reasoner)
        assert "Some information may be partially incorrect" in sent
        assert "Spatial context" not in sent

    def test_apply_transform_emits_distillation_pair(self):
        reasoner = _reasoner()
        example = {
            "image": [Image.new("RGB", (2, 2))],
            "messages": [_messages()],
            "captions": [CAPTIONS],
            "pointclouds": [[CHAIR_POINTS, TABLE_POINTS]],
        }
        out = reasoner.apply_transform(example, images="image", text="messages")

        # Student side: the input is the question alone — no metric guidance.
        assert "How far is the wooden chair" in out["input"][0]
        assert "mean surface distance" not in out["input"][0]
        assert out["output"][0] == "stub reasoning"
        assert out["reasoning"][0] == "on"

        # Teacher side: the call was conditioned on the measured facts, which
        # are preserved in their own audit column rather than leaking into
        # the student-visible fields.
        assert '"wooden chair"' in out["spatial_context"][0]
        sent = _sent_text(reasoner)
        assert "mean surface distance" in sent
        assert '"wooden chair"' in sent

    def test_apply_transform_without_scene_columns_matches_legacy(self):
        reasoner = _reasoner()
        example = {
            "image": [Image.new("RGB", (2, 2))],
            "messages": [_messages()],
        }
        out = reasoner.apply_transform(example, images="image", text="messages")
        assert out["spatial_context"][0] == ""
        assert out["output"][0] == "stub reasoning"
        assert "Some information may be partially incorrect" in _sent_text(reasoner)

    def test_apply_transform_single_example(self):
        reasoner = _reasoner()
        example = {
            "image": Image.new("RGB", (2, 2)),
            "messages": _messages(),
            "captions": CAPTIONS,
            "pointclouds": [CHAIR_POINTS, TABLE_POINTS],
        }
        out = reasoner.apply_transform(example, images="image", text="messages")
        assert "How far is the wooden chair" in out["input"]
        assert '"wooden chair"' in out["spatial_context"]
        assert out["reasoning"] == "on"
