"""Contract tests for SpatialAnnotatorVlmClient.

Verifies the adapter satisfies lerobot's ``VlmClient`` protocol without
depending on lerobot itself. Uses a MockAnnotator that returns a fixed
response so tests are deterministic and don't need real LLM access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from experiments.nooa_agent.lerobot_adapter import (
    SpatialAnnotatorVlmClient,
    _extract_image_and_prompt,
    _run_sync,
    _strip_to_json,
)


@dataclass
class _MockAnswer:
    answer: str
    confidence: str = "high"
    supporting_evidence: list = field(default_factory=list)
    tool_calls_used: int = 3


class _MockAnnotator:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[tuple[Any, str]] = []

    async def annotate(self, image, question):
        self.calls.append((image, question))
        return _MockAnswer(answer=self.response_text)


# ── _extract_image_and_prompt ─────────────────────────────────────────

def test_extract_openai_multimodal_shape():
    sentinel = object()
    messages = [
        {"role": "system", "content": "You are..."},
        {"role": "user", "content": [
            {"type": "image", "image": sentinel},
            {"type": "text", "text": "How far apart?"},
        ]},
    ]
    image, prompt = _extract_image_and_prompt(messages)
    assert image is sentinel
    assert prompt == "How far apart?"

def test_extract_openai_image_url_variant():
    """Some callers use type=image_url with a nested {url: ...} dict."""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,X"}},
        {"type": "text", "text": "Describe."},
    ]}]
    image, prompt = _extract_image_and_prompt(messages)
    assert image == "data:image/jpeg;base64,X"
    assert prompt == "Describe."

def test_extract_uses_last_user_turn_only():
    """Earlier user turns are context; the last one is the actual query."""
    img_a, img_b = object(), object()
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img_a},
            {"type": "text", "text": "Old"},
        ]},
        {"role": "assistant", "content": "Old reply"},
        {"role": "user", "content": [
            {"type": "image", "image": img_b},
            {"type": "text", "text": "New"},
        ]},
    ]
    image, prompt = _extract_image_and_prompt(messages)
    assert image is img_b
    assert prompt == "New"

def test_extract_text_only_content_returns_no_image():
    messages = [{"role": "user", "content": "just text"}]
    image, prompt = _extract_image_and_prompt(messages)
    assert image is None
    assert prompt == "just text"


# ── _strip_to_json ────────────────────────────────────────────────────

def test_strip_to_json_plain():
    assert _strip_to_json('{"a": 1}') == {"a": 1}

def test_strip_to_json_json_fence():
    assert _strip_to_json('```json\n{"a": 1, "b": "x"}\n```') == {"a": 1, "b": "x"}

def test_strip_to_json_qwen3_thinking_block():
    text = '<think>let me think...</think>\n{"answer": 42}'
    assert _strip_to_json(text) == {"answer": 42}

def test_strip_to_json_embedded_in_prose():
    text = 'Here is the answer: {"result": "ok"} — that\'s it.'
    assert _strip_to_json(text) == {"result": "ok"}

def test_strip_to_json_nested_object():
    text = '{"outer": {"inner": {"deep": [1, 2, 3]}}}'
    result = _strip_to_json(text)
    assert result["outer"]["inner"]["deep"] == [1, 2, 3]

def test_strip_to_json_raises_on_no_json():
    with pytest.raises(ValueError, match="No JSON object"):
        _strip_to_json("just prose, no braces at all")

def test_strip_to_json_raises_on_unbalanced():
    with pytest.raises(ValueError):
        _strip_to_json("{'a': 1, missing closing brace")


# ── _run_sync ──────────────────────────────────────────────────────────

def test_run_sync_from_sync_context():
    """The async→sync bridge must not deadlock or lose the result."""
    async def sample():
        return 42
    assert _run_sync(sample()) == 42


# ── SpatialAnnotatorVlmClient contract ────────────────────────────────

def test_vlm_client_returns_list_of_same_length_as_batch():
    agent = _MockAnnotator('{"summary": "ok"}')
    vlm = SpatialAnnotatorVlmClient(annotator=agent)
    batch = [
        [{"role": "user", "content": [
            {"type": "image", "image": object()},
            {"type": "text", "text": "Q1"}]}],
        [{"role": "user", "content": [
            {"type": "image", "image": object()},
            {"type": "text", "text": "Q2"}]}],
    ]
    results = vlm.generate_json(batch)
    assert len(results) == 2
    assert results[0] == {"summary": "ok"}
    assert len(agent.calls) == 2

def test_vlm_client_dispatches_image_and_prompt_to_annotator():
    agent = _MockAnnotator('{"x": 1}')
    vlm = SpatialAnnotatorVlmClient(annotator=agent)
    sentinel = object()
    vlm.generate_json([[{
        "role": "user", "content": [
            {"type": "image", "image": sentinel},
            {"type": "text", "text": "the prompt"},
        ]}]])
    call_img, call_prompt = agent.calls[0]
    assert call_img is sentinel
    assert call_prompt == "the prompt"

def test_vlm_client_accepts_protocol_kwargs():
    """max_new_tokens + temperature accepted (protocol requirement), ignored
    downstream — LLM config lives on the SpatialAnnotator construction path."""
    agent = _MockAnnotator('{"a": 1}')
    vlm = SpatialAnnotatorVlmClient(annotator=agent)
    result = vlm.generate_json(
        [[{"role": "user", "content": [
            {"type": "image", "image": object()},
            {"type": "text", "text": "q"},
        ]}]],
        max_new_tokens=512,
        temperature=0.7,
    )
    assert result == [{"a": 1}]

def test_vlm_client_errors_clearly_when_image_missing():
    agent = _MockAnnotator('{}')
    vlm = SpatialAnnotatorVlmClient(annotator=agent)
    with pytest.raises(ValueError, match="image block"):
        vlm.generate_json([[{"role": "user", "content": [
            {"type": "text", "text": "text only"},
        ]}]])

def test_vlm_client_parses_json_from_qwen3_thinking_response():
    """End-to-end contract check: annotator emits thinking-tagged JSON,
    adapter parses it out."""
    agent = _MockAnnotator(
        '<think>Let me consider the workers.</think>\n'
        '{"scene_perception": "warehouse", "count": 2}'
    )
    vlm = SpatialAnnotatorVlmClient(annotator=agent)
    result = vlm.generate_json([[{"role": "user", "content": [
        {"type": "image", "image": object()},
        {"type": "text", "text": "describe"},
    ]}]])
    assert result == [{"scene_perception": "warehouse", "count": 2}]
