"""CPU tests for the Phase-3 grounded + soft-trace teacher in vqasynth.r1_reasoning.

Mirrors the CPU-mock philosophy of the other stage tests (test_scene_fusion.py /
test_correspondence.py): the OpenAI client is replaced by a recorder so NO
network call is made — what is under test is ``SoftTraceReasoner``'s OWN logic:
that the grounded scene state (calibrated metric positions + surfaced depth_conf
+ cross-view inlier_ratio) is injected into the teacher prompt, that confidence
is PROPAGATED (low-confidence measurements rendered as rough/hedged, high-
confidence not), that malformed/empty state degrades to the base reasoner
(back-compat), and that ``apply_transform`` threads the grounded-state column.

The real teacher call (trace quality) needs a live model and is validated
downstream, not here.
"""
from __future__ import annotations

from PIL import Image

from vqasynth.r1_reasoning import R1Reasoner, SoftTraceReasoner


class _RecorderCompletions:
    """Stands in for client.chat.completions; records the last call's messages."""

    def __init__(self):
        self.last_messages = None

    def create(self, model, messages):
        self.last_messages = messages

        class _Msg:
            content = "<think>mock</think><answer>mock</answer>"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _RecorderClient:
    def __init__(self):
        self.chat = type("chat", (), {"completions": _RecorderCompletions()})()


def _make_reasoner(cls=SoftTraceReasoner):
    r = cls(api_key="dummy", model="mock-model", image_column="image", text_column="messages")
    r.client = _RecorderClient()  # no network
    return r


def _prompt_text(reasoner):
    """Extract the text block from the recorded user message."""
    msgs = reasoner.client.chat.completions.last_messages
    assert msgs is not None, "client was never called"
    for part in msgs[0]["content"]:
        if part.get("type") == "text":
            return part["text"]
    raise AssertionError("no text part in recorded message")


IMG = Image.new("RGB", (8, 8), (127, 127, 127))

GROUNDED = {
    "frames": 2,
    "objects": [
        {"label": "chair", "center_m": [0.1, 0.0, 2.10], "depth_conf": 0.90, "frame_ids": [0, 1]},
        {"label": "mug", "center_m": [-0.3, 0.1, 0.70], "depth_conf": 0.30, "frame_ids": [0]},
    ],
    "pairs": [
        {"a": "chair", "b": "mug", "distance_m": 1.42, "inlier_ratio": 0.82},
    ],
}


def test_grounded_state_injected_into_prompt():
    r = _make_reasoner()
    out = r.run("How far is the chair?", "About 2.1 meters.", IMG, grounded_state=GROUNDED)
    assert out  # recorder returns a mock trace
    text = _prompt_text(r)
    # grounded objects + calibrated metric distances are cited
    assert "chair" in text and "mug" in text
    assert "2.10 m" in text  # chair depth (z) in meters
    assert "1.42 m" in text  # calibrated pairwise distance
    # multi-view provenance surfaced
    assert "2 views" in text
    # the soft-composition system prompt is in play
    assert "PROPAGATE YOUR CONFIDENCE" in text


def test_confidence_is_propagated_not_flattened():
    r = _make_reasoner()
    r.run("How far is the chair?", "About 2.1 meters.", IMG, grounded_state=GROUNDED)
    text = _prompt_text(r)
    # high-confidence object -> confident phrasing; low-confidence -> hedged
    assert "fairly confident" in text          # chair, depth_conf 0.90
    assert "low confidence, a rough estimate" in text  # mug, depth_conf 0.30
    # reliable cross-view match flagged as solid
    assert "solid" in text and "0.82" in text


def test_confidence_phrase_thresholds():
    f = SoftTraceReasoner._confidence_phrase
    assert f(0.95) == "fairly confident"
    assert f(0.80) == "fairly confident"
    assert f(0.65) == "moderately sure"
    assert f(0.49) == "low confidence, a rough estimate"
    assert f(None) is None
    assert f("not-a-number") is None


def test_shaky_crossview_match_flagged():
    r = _make_reasoner()
    state = {"pairs": [{"a": "box", "b": "lamp", "distance_m": 0.9, "inlier_ratio": 0.35}]}
    r.run("q", "a", IMG, grounded_state=state)
    text = _prompt_text(r)
    assert "shaky" in text and "0.35" in text


def test_backcompat_no_grounded_state_falls_back_to_base():
    r = _make_reasoner()
    r.run("How far is the chair?", "About 2.1 meters.", IMG, grounded_state=None)
    text = _prompt_text(r)
    # base R1Reasoner system prompt, NOT the grounded block
    assert "formulate a reasoning trace" in text
    assert "Things I can make out in the scene" not in text
    assert "PROPAGATE YOUR CONFIDENCE" not in text


def test_malformed_state_degrades_gracefully():
    r = _make_reasoner()
    # empty dict, bad JSON string, and non-dict all -> base fallback
    for bad in ({}, "{not json", "[]", 42):
        assert r._format_grounded_block(bad) == ""
    # a JSON *string* of a valid record is accepted
    import json

    block = r._format_grounded_block(json.dumps(GROUNDED))
    assert "chair" in block and "2.10 m" in block


def test_apply_transform_threads_grounded_state_column():
    r = _make_reasoner()
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "How far is the chair?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "About 2.1 meters."}]},
    ]
    examples = {"image": IMG, "messages": conversation, "grounded_state": GROUNDED}
    out = r.apply_transform(examples)
    assert out["reasoning"] == "on"
    assert out["output"]  # a trace was produced
    # the grounded block reached the teacher prompt
    text = _prompt_text(r)
    assert "chair" in text and "fairly confident" in text


def test_apply_transform_missing_column_is_backcompat():
    r = _make_reasoner()
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "How far is the chair?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "About 2.1 meters."}]},
    ]
    examples = {"image": IMG, "messages": conversation}  # no grounded_state column
    out = r.apply_transform(examples)
    assert out["reasoning"] == "on"
    text = _prompt_text(r)
    assert "Things I can make out in the scene" not in text  # base path
