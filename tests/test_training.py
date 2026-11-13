"""CPU tests for the Phase-5 training glue (vqasynth.training).

Data formatting (gated rows -> chat SFT examples) and the GRPO reward adapter
(wraps the dense reward; accepts string or chat-message completions). Pure CPU.
"""
from __future__ import annotations

import pytest

from vqasynth import training as T

MESSAGES = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Where is the chair?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "(A) left"}]},
]
TRACE = "<think>roughly 2 m, not sure</think><answer>(A) left</answer>"


def test_extract_question():
    assert T.extract_question(MESSAGES) == "Where is the chair?"
    assert T.extract_question([]) is None
    assert T.extract_question("nope") is None


def test_to_sft_example():
    ex = T.to_sft_example("Where is the chair?", TRACE)
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["user", "assistant"]
    # image placeholder + question in the user turn
    types = [c["type"] for c in ex["messages"][0]["content"]]
    assert "image" in types and "text" in types
    assert ex["messages"][1]["content"][0]["text"] == TRACE


def test_to_sft_example_with_system_and_empties():
    ex = T.to_sft_example("q", "t", system="be grounded")
    assert ex["messages"][0]["role"] == "system"
    assert T.to_sft_example("", TRACE) is None
    assert T.to_sft_example("q", "") is None


def test_format_sft_dataset_drops_empty():
    rows = [
        {"messages": MESSAGES, "output": TRACE},
        {"messages": MESSAGES, "output": ""},        # dropped (empty trace)
        {"messages": [], "output": TRACE},           # dropped (no question)
    ]
    examples, dropped = T.format_sft_dataset(
        rows, question_of=lambda r: T.extract_question(r["messages"]), trace_of=lambda r: r["output"]
    )
    assert len(examples) == 1 and dropped == 2


# --- GRPO reward adapter ---------------------------------------------------

GS = {
    "objects": [
        {"label": "red cube", "bbox": [10, 10, 50, 50]},
        {"label": "green sphere", "bbox": [100, 100, 140, 140]},
    ],
    "pairs": [{"a": "red cube", "b": "green sphere"}],
}
GOOD = (
    "<observe>a cube and a sphere</observe>"
    '<scene>{"objects":[{"id":0,"label":"red cube","bbox":[10,10,50,50]},'
    '{"id":1,"label":"green sphere","bbox":[100,100,140,140]}],'
    '"relations":[{"subject":0,"predicate":"related","object":1}]}</scene>'
    "<think>cube is left</think><answer>(A)</answer>"
)


def test_grpo_reward_batch_string_completions():
    rewards = T.grpo_reward([GOOD], grounded_state=[GS], answer=["(A)"])
    assert rewards[0] == pytest.approx(1.0, abs=1e-6)
    # wrong answer -> accuracy 0, spatial gated off -> 0.3
    wrong = T.grpo_reward([GOOD.replace("(A)", "(B)")], grounded_state=[GS], answer=["(A)"])
    assert wrong[0] == pytest.approx(0.3, abs=1e-6)


def test_grpo_reward_accepts_chat_message_completions():
    chat = [[{"role": "assistant", "content": [{"type": "text", "text": GOOD}]}]]
    rewards = T.grpo_reward(chat, grounded_state=[GS], answer=["(A)"])
    assert rewards[0] == pytest.approx(1.0, abs=1e-6)


def test_grpo_reward_breakdown():
    br = T.grpo_reward([GOOD], grounded_state=[GS], answer=["(A)"], return_breakdown=True)
    assert set(br[0]) == {"total", "format", "count", "accuracy", "spatial"}
    assert br[0]["spatial"] == pytest.approx(1.0, abs=1e-6)


def test_make_grpo_reward_is_trl_shaped():
    fn = T.make_grpo_reward()
    assert fn.__name__ == "vqasynth_pp_dense_reward"
    # trainer-style call: completions + forwarded dataset columns
    out = fn(completions=[GOOD], grounded_state=[GS], answer=["(A)"])
    assert out[0] == pytest.approx(1.0, abs=1e-6)
