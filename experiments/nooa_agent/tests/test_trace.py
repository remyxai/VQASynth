"""Tests for trace capture + serialization.

No NOOA install required — uses local pydantic mocks that mimic NOOA's event
shape (event_type + model_dump). Verifies the capture/serialize/write pipeline
end-to-end without needing the real runtime.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from experiments.nooa_agent.trace import (
    AnnotateTrace,
    TraceWriter,
    capture_trace,
    qwen_vl_serialize,
)


# ── Mock NOOA event shapes ─────────────────────────────────────────────
# Match NOOA's pydantic BaseModel + event_type attribute pattern.

class _MockTask(BaseModel):
    event_type: str = "task"
    prompt: str
    images: list[dict] = []


class _MockToolCall(BaseModel):
    event_type: str = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


class _MockToolResult(BaseModel):
    event_type: str = "tool_result"
    tool_call_id: str
    value: Any = None
    error: str | None = None


class _MockMessage(BaseModel):
    event_type: str = "message"
    content: str


class _MockEventManager:
    def __init__(self, events):
        self._events = events
    def values(self):
        return self._events


class _MockAgent:
    def __init__(self, events):
        self.event_manager = _MockEventManager(events)


@dataclass
class _MockAnswer:
    answer: str = "test answer"
    confidence: str = "high"
    supporting_evidence: list = field(default_factory=lambda: ["evidence 1"])
    tool_calls_used: int = 3


# ── capture_trace ──────────────────────────────────────────────────────

def test_capture_trace_grabs_only_delta_since_index():
    """NOOA's event_manager accumulates events across calls; capture should
    only include events at/after since_event_idx."""
    events = [
        _MockTask(prompt="old task"),
        _MockToolCall(tool_call_id="a1", name="old_tool", arguments={}),
        _MockTask(prompt="current task"),                          # index 2
        _MockToolCall(tool_call_id="b1", name="detect_objects", arguments={"phrase": "worker"}),
        _MockMessage(content="Done."),
    ]
    agent = _MockAgent(events)
    trace = capture_trace(
        agent=agent,
        image_ref="/data/img.jpg",
        question="How far?",
        system_prompt="You are an agent.",
        answer=_MockAnswer(),
        elapsed_s=1.5,
        since_event_idx=2,  # skip the first two "old" events
    )
    assert len(trace.events) == 3
    assert trace.events[0]["event_type"] == "task"
    assert trace.events[0]["prompt"] == "current task"
    assert trace.events[1]["event_type"] == "tool_call"
    assert trace.events[1]["arguments"] == {"phrase": "worker"}
    assert trace.events[2]["content"] == "Done."


def test_capture_trace_serializes_answer_fields():
    agent = _MockAgent([])
    trace = capture_trace(
        agent=agent, image_ref="x", question="q", system_prompt="s",
        answer=_MockAnswer(answer="A", confidence="low", supporting_evidence=["e1", "e2"],
                           tool_calls_used=7),
        elapsed_s=0.1,
    )
    assert trace.final_answer == {
        "answer": "A", "confidence": "low",
        "supporting_evidence": ["e1", "e2"], "tool_calls_used": 7,
    }
    assert trace.wall_clock_s == 0.1


# ── qwen_vl_serialize ──────────────────────────────────────────────────

def test_qwen_serialize_emits_openai_messages_shape():
    trace = AnnotateTrace(
        image_ref="/data/warehouse.jpg",
        question="How far apart are the two workers?",
        system_prompt="You are a spatial annotation agent.",
        events=[
            {"event_type": "task", "prompt": "How far apart are the two workers?",
             "images": [{"placeholder": "yes"}]},
            {"event_type": "tool_call", "tool_call_id": "c1",
             "name": "detect_objects", "arguments": {"phrase": "worker"}},
            {"event_type": "tool_result", "tool_call_id": "c1",
             "value": "[Box(x1=100, ...), Box(x1=400, ...)]"},
            {"event_type": "tool_call", "tool_call_id": "c2",
             "name": "distance_3d", "arguments": {"box_a": "...", "box_b": "..."}},
            {"event_type": "tool_result", "tool_call_id": "c2",
             "value": "{'distance_m': 2.04}"},
            {"event_type": "message",
             "content": "The two workers are 2.04 meters apart."},
        ],
        final_answer={"answer": "...", "confidence": "medium",
                      "supporting_evidence": [], "tool_calls_used": 2},
        wall_clock_s=1.23,
    )
    out = qwen_vl_serialize(trace)
    msgs = out["messages"]

    # System first, then user, then alternating assistant/tool, ending assistant
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # User content should be a list with image + text blocks
    user_content = msgs[1]["content"]
    assert any(b.get("type") == "image" and b.get("image") == "/data/warehouse.jpg"
               for b in user_content)
    assert any(b.get("type") == "text" and "workers" in b.get("text", "")
               for b in user_content)

    # Two tool_call/tool_result cycles + final assistant
    tool_call_msgs = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(tool_call_msgs) == 2
    # Arguments should be JSON-string, per OpenAI format
    args_str = tool_call_msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args_str) == {"phrase": "worker"}

    tool_result_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_result_msgs) == 2
    assert tool_result_msgs[0]["tool_call_id"] == "c1"

    # Meta block carries the summary — useful for filtering the dataset later
    assert out["meta"]["confidence"] == "medium"
    assert out["meta"]["image_ref"] == "/data/warehouse.jpg"


def test_qwen_serialize_tool_result_prefers_error_over_value():
    """If a tool errored, the error message should surface to the LLM
    (that's how NOOA feeds it back for retry), not the empty/None value."""
    trace = AnnotateTrace(
        image_ref="x", question="q", system_prompt="",
        events=[
            {"event_type": "tool_result", "tool_call_id": "c1",
             "value": None, "error": "ImportError: numpy 2.0 incompat"},
        ],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 0},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    tool_msg = [m for m in out["messages"] if m["role"] == "tool"][0]
    assert "numpy" in tool_msg["content"]


def test_qwen_serialize_synthesizes_image_block_when_task_has_no_images():
    """Even if the task event didn't carry an image block, the trace's
    image_ref means there WAS an image — surface it in the user turn so
    the training row is complete."""
    trace = AnnotateTrace(
        image_ref="/data/warehouse.jpg", question="q", system_prompt="",
        events=[{"event_type": "task", "prompt": "q", "images": []}],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 0},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    user_content = out["messages"][0]["content"]  # no system → user is first
    assert any(b.get("type") == "image" for b in user_content)


# ── TraceWriter ────────────────────────────────────────────────────────

def test_trace_writer_appends_one_line_per_trace():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "traces.jsonl")
        with TraceWriter(path) as writer:
            for i in range(3):
                trace = AnnotateTrace(
                    image_ref=f"/data/img{i}.jpg",
                    question=f"Question {i}",
                    system_prompt="",
                    events=[{"event_type": "message", "content": f"answer {i}"}],
                    final_answer={"answer": f"a{i}", "confidence": "high",
                                  "supporting_evidence": [], "tool_calls_used": 1},
                    wall_clock_s=0.1,
                )
                writer.write(trace)

        lines = Path(path).read_text().splitlines()
        assert len(lines) == 3
        for i, line in enumerate(lines):
            row = json.loads(line)
            assert row["meta"]["image_ref"] == f"/data/img{i}.jpg"
            assert row["meta"]["question"] == f"Question {i}"


def test_trace_writer_uses_pluggable_serializer():
    """Custom serializer → we can target any downstream format."""
    def minimal_serializer(trace):
        return {"q": trace.question, "a": trace.final_answer["answer"]}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "minimal.jsonl")
        with TraceWriter(path, serializer=minimal_serializer) as writer:
            trace = AnnotateTrace(
                image_ref="x", question="How tall?", system_prompt="",
                events=[], wall_clock_s=0.0,
                final_answer={"answer": "1.8m", "confidence": "high",
                              "supporting_evidence": [], "tool_calls_used": 1},
            )
            writer.write(trace)
        row = json.loads(Path(path).read_text().strip())
        assert row == {"q": "How tall?", "a": "1.8m"}
