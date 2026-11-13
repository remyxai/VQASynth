"""CPU tests for GroundedBBoxTraceReasoner (SpatialThinker-style teacher).

Client is mocked (no API/GPU); what is under test is the four-tag scene-graph
prompt construction: the <observe>/<scene>/<think>/<answer> instruction, the JSON
subgraph schema with pixel bboxes, the image-size bound, the inherited grounded
hints + confidence propagation, and image-size inference from a PIL image.
"""
from __future__ import annotations

from PIL import Image

from vqasynth.r1_reasoning import GroundedBBoxTraceReasoner


class _RecorderCompletions:
    def __init__(self):
        self.last_messages = None

    def create(self, model, messages):
        self.last_messages = messages

        class _Msg:
            content = "<observe>x</observe><scene>{}</scene><think>y</think><answer>z</answer>"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _RecorderClient:
    def __init__(self):
        self.chat = type("chat", (), {"completions": _RecorderCompletions()})()


def _reasoner():
    r = GroundedBBoxTraceReasoner(api_key="x", model="m", image_column="image", text_column="messages")
    r.client = _RecorderClient()
    return r


def _text(r):
    for part in r.client.chat.completions.last_messages[0]["content"]:
        if part.get("type") == "text":
            return part["text"]
    raise AssertionError("no text part")


IMG = Image.new("RGB", (640, 480), (120, 120, 120))

GROUNDED = {
    "frames": 1,
    "objects": [
        {"label": "chair", "center_m": [0.1, 0.0, 2.1], "depth_conf": 0.9},
        {"label": "mug", "center_m": [-0.3, 0.1, 0.7], "depth_conf": 0.3},
    ],
    "pairs": [{"a": "chair", "b": "mug", "distance_m": 1.42, "inlier_ratio": 0.82}],
}


def test_four_tag_scene_graph_instruction():
    r = _reasoner()
    out = r.run("Where is the chair?", "(A) left", IMG, grounded_state=GROUNDED)
    assert out
    text = _text(r)
    for tag in ("<observe>", "<scene>", "<think>", "<answer>"):
        assert tag in text
    # scene JSON schema cues + pixel-bbox requirement
    assert '"objects"' in text and '"bbox"' in text and '"relations"' in text
    assert "ABSOLUTE PIXEL" in text
    assert "PROPAGATE YOUR CONFIDENCE" in text


def test_image_size_inferred_from_pil():
    r = _reasoner()
    r.run("q", "a", IMG, grounded_state=GROUNDED)
    text = _text(r)
    assert "Image size: 640 x 480" in text


def test_grounded_hints_injected_with_confidence():
    r = _reasoner()
    r.run("q", "a", IMG, grounded_state=GROUNDED)
    text = _text(r)
    assert "chair" in text and "mug" in text
    assert "fairly confident" in text                   # chair depth_conf 0.9
    assert "low confidence, a rough estimate" in text   # mug depth_conf 0.3


def test_works_without_grounded_state():
    # still emits the four-tag format (teacher grounds visually); no hint block
    r = _reasoner()
    out = r.run("q", "a", IMG, grounded_state=None)
    assert out
    text = _text(r)
    assert "<scene>" in text and "Image size: 640 x 480" in text
    assert "Things I can make out in the scene" not in text


def test_explicit_image_size_override():
    r = _reasoner()
    r.run("q", "a", IMG, grounded_state=None, image_size=(1024, 768))
    assert "Image size: 1024 x 768" in _text(r)


def test_perspective_addendum_in_bbox_reasoner():
    r = _reasoner()
    r.run("From the viewpoint of camera 1, where is the chair?", "(A) left", IMG, grounded_state=GROUNDED)
    assert "re-anchor to that viewpoint" in _text(r)
