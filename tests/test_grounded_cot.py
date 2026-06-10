"""Tests for the grounded four-stage CoT scaffold and its wiring into
``R1Reasoner`` (the existing reasoning call site)."""

from vqasynth.grounded_cot import (
    SCAFFOLD_STAGES,
    build_scaffold_system_prompt,
    parse_scaffold,
    scaffold_grounding_score,
    is_well_grounded,
)
# Import from the existing (non-new) call-site module under test.
from vqasynth.r1_reasoning import R1Reasoner


GROUNDED_TRACE = (
    "<entities>a red mug, a wooden table</entities>"
    "<relations>the mug rests on the table</relations>"
    "<think>The mug sits on the table, so the mug height adds to the table "
    "height when I estimate the distance.</think>"
    "<answer>about 90 centimeters</answer>"
)

UNGROUNDED_TRACE = (
    "<think>I feel like the gadget floats somewhere up high near a vague "
    "structure, roughly speaking.</think>"
    "<answer>maybe far</answer>"
)


def test_build_prompt_lists_four_ordered_stages():
    prompt = build_scaffold_system_prompt("DOMAIN FRAMING")
    assert "DOMAIN FRAMING" in prompt
    # Each stage tag must be requested, in causal order.
    positions = [prompt.index(f"<{tag}>") for _, tag in SCAFFOLD_STAGES]
    assert positions == sorted(positions)


def test_parse_scaffold_extracts_each_stage():
    parsed = parse_scaffold(GROUNDED_TRACE)
    assert parsed["entity_grounding"] == "a red mug, a wooden table"
    assert parsed["relation_modeling"] == "the mug rests on the table"
    assert parsed["answer"] == "about 90 centimeters"


def test_grounding_score_rewards_anchored_reasoning():
    grounded = scaffold_grounding_score(GROUNDED_TRACE)
    ungrounded = scaffold_grounding_score(UNGROUNDED_TRACE)
    assert grounded > ungrounded
    assert is_well_grounded(GROUNDED_TRACE)
    assert not is_well_grounded(UNGROUNDED_TRACE)


class _RecordingClient:
    """Stands in for the OpenAI client; captures the messages it is called
    with and returns a canned grounded trace."""

    def __init__(self):
        self.captured = None
        self.chat = self  # so .chat.completions.create resolves
        self.completions = self

    def create(self, model, messages):
        self.captured = messages

        class _Msg:
            content = GROUNDED_TRACE

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def test_r1reasoner_run_uses_grounded_scaffold(monkeypatch):
    # Avoid constructing a real OpenAI client (no network / no key).
    monkeypatch.setattr(
        "vqasynth.r1_reasoning.OpenAI", lambda *a, **k: object()
    )
    reasoner = R1Reasoner(
        api_key="x", model="m", image_column="image", text_column="messages"
    )
    fake = _RecordingClient()
    reasoner.client = fake

    out = reasoner.run("How far is the mug?", "about 90 cm", image=None)

    # The wiring edit must send the four-stage scaffold in the system prompt.
    sent_text = fake.captured[0]["content"][0]["text"]
    for _, tag in SCAFFOLD_STAGES:
        assert f"<{tag}>" in sent_text
    # And the returned trace is still a plain reasoning string into 'output'.
    assert isinstance(out, str)
    assert is_well_grounded(out)
