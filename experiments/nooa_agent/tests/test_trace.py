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
# Match NOOA's actual event schemas verified against nooa/context_blocks/
# events.py + nooa/events.py:
# - event_type is the CLASS NAME (auto-derived, PascalCase)
# - ToolCallEvent holds its result INLINE via a nested `result` field
#   (ToolResult with tool_call_id + content + result_status)
# - PythonOutput is a SEPARATE event (only for CodeAct execute_python)

class _MockTask(BaseModel):
    event_type: str = "Task"
    prompt: str
    images: list[dict] = []


class _MockToolCallEvent(BaseModel):
    """Mirrors nooa.context_blocks.events.ToolCallEvent — result is nested."""
    event_type: str = "ToolCallEvent"
    tool_call_id: str
    name: str
    arguments: dict
    result: dict | None = None    # ToolResult dict: {tool_call_id, content, result_status}


class _MockPythonOutput(BaseModel):
    """CodeAct execute_python output — only relevant if CodeAct is enabled."""
    event_type: str = "PythonOutput"
    tool_call_id: str
    execution_status: str = "complete"   # ResultStatus enum's str form
    stdout: str = ""
    stderr: str = ""


class _MockMessage(BaseModel):
    event_type: str = "Message"
    content: str


class _MockLLMOutput(BaseModel):
    event_type: str = "LLMOutput"
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
        _MockToolCallEvent(tool_call_id="a1", name="old_tool", arguments={}),
        _MockTask(prompt="current task"),                          # index 2
        _MockToolCallEvent(tool_call_id="b1", name="detect_objects",
                           arguments={"phrase": "worker"},
                           result={"tool_call_id": "b1", "content": "[Box(...)]",
                                   "result_status": "complete"}),
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
    assert trace.events[0]["event_type"] == "Task"
    assert trace.events[0]["prompt"] == "current task"
    assert trace.events[1]["event_type"] == "ToolCallEvent"
    assert trace.events[1]["arguments"] == {"phrase": "worker"}
    # Nested result carries through the dump
    assert trace.events[1]["result"]["content"] == "[Box(...)]"
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
    """ToolCallEvent has result NESTED — one event, two output messages."""
    trace = AnnotateTrace(
        image_ref="/data/warehouse.jpg",
        question="How far apart are the two workers?",
        system_prompt="You are a spatial annotation agent.",
        events=[
            {"event_type": "Task", "prompt": "How far apart are the two workers?",
             "images": [{"placeholder": "yes"}]},
            {"event_type": "ToolCallEvent", "tool_call_id": "c1",
             "name": "detect_objects", "arguments": {"phrase": "worker"},
             "result": {"tool_call_id": "c1",
                        "content": "[Box(x1=100, ...), Box(x1=400, ...)]",
                        "result_status": "complete"}},
            {"event_type": "ToolCallEvent", "tool_call_id": "c2",
             "name": "distance_3d",
             "arguments": {"box_a": "...", "box_b": "..."},
             "result": {"tool_call_id": "c2",
                        "content": "{'distance_m': 2.04}",
                        "result_status": "complete"}},
            {"event_type": "LLMOutput",
             "content": "The two workers are 2.04 meters apart."},
        ],
        final_answer={"answer": "...", "confidence": "medium",
                      "supporting_evidence": [], "tool_calls_used": 2},
        wall_clock_s=1.23,
    )
    out = qwen_vl_serialize(trace)
    msgs = out["messages"]

    # System first, then user, then two assistant+tool pairs, then final assistant
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    user_content = msgs[1]["content"]
    assert any(b.get("type") == "image" and b.get("image") == "/data/warehouse.jpg"
               for b in user_content)
    assert any(b.get("type") == "text" and "workers" in b.get("text", "")
               for b in user_content)

    # Two ToolCallEvents → 2 assistant + 2 tool messages
    tool_call_msgs = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(tool_call_msgs) == 2
    args_str = tool_call_msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args_str) == {"phrase": "worker"}

    tool_result_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_result_msgs) == 2
    # ids are shortened; verify each tool_result matches its preceding call
    assert tool_result_msgs[0]["tool_call_id"] == tool_call_msgs[0]["tool_calls"][0]["id"]
    assert tool_result_msgs[1]["tool_call_id"] == tool_call_msgs[1]["tool_calls"][0]["id"]
    assert "Box" in tool_result_msgs[0]["content"]

    # LLMOutput becomes the final assistant text (no tool_calls key)
    final = [m for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")][-1]
    assert "2.04 meters" in final["content"]

    # Meta block carries the summary — useful for filtering the dataset later
    assert out["meta"]["confidence"] == "medium"
    assert out["meta"]["image_ref"] == "/data/warehouse.jpg"


def test_qwen_serialize_python_output_from_codeact_path():
    """PythonOutput is only emitted for CodeAct execute_python. Verify we
    still handle it correctly so a CodeAct-enabled variant would just work."""
    trace = AnnotateTrace(
        image_ref="x", question="q", system_prompt="",
        events=[
            {"event_type": "PythonOutput", "tool_call_id": "c1",
             "execution_status": "complete",
             "stdout": "median distance: 3.42m", "stderr": ""},
        ],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 1},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    tool_msg = [m for m in out["messages"] if m["role"] == "tool"][0]
    assert "3.42m" in tool_msg["content"]


def test_qwen_serialize_python_output_prefers_stderr_on_error_status():
    """When execute_python errored, stderr carries the useful diagnostic."""
    trace = AnnotateTrace(
        image_ref="x", question="q", system_prompt="",
        events=[
            {"event_type": "PythonOutput", "tool_call_id": "c1",
             "execution_status": "error",
             "stdout": "", "stderr": "NameError: name 'foo' is not defined"},
        ],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 1},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    tool_msg = [m for m in out["messages"] if m["role"] == "tool"][0]
    assert "NameError" in tool_msg["content"]


def test_qwen_serialize_always_emits_image_block_from_image_ref():
    """We ignore NOOA's opaque Task.images and reference by image_ref."""
    trace = AnnotateTrace(
        image_ref="/data/warehouse.jpg", question="q", system_prompt="",
        events=[{"event_type": "Task", "prompt": "q", "images": []}],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 0},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    user_content = out["messages"][0]["content"]  # no system → user is first
    image_blocks = [b for b in user_content if b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image"] == "/data/warehouse.jpg"


def test_qwen_serialize_skips_prefill_events():
    """CodeAct emits input-inspection code as a 'prefill_*' ToolCallEvent +
    matching PythonOutput at the top of every generation. It's noise, not
    substantive reasoning — must not appear in training rows."""
    trace = AnnotateTrace(
        image_ref="/data/img.jpg", question="q", system_prompt="",
        events=[
            {"event_type": "Task", "prompt": "system-formatted task", "images": []},
            {"event_type": "ToolCallEvent",
             "tool_call_id": "prefill_abc123",
             "name": "execute_python",
             "arguments": {"code": "pprint(image)"},
             "result": {"content": "status: complete", "result_status": "complete"}},
            {"event_type": "PythonOutput",
             "tool_call_id": "prefill_abc123",
             "execution_status": "complete", "stdout": "<PIL...>", "stderr": ""},
            {"event_type": "ToolCallEvent",
             "tool_call_id": "call_real_thing",
             "name": "execute_python",
             "arguments": {"code": "workers = self.detect_objects(image, phrase='worker')"},
             "result": {"content": "status: complete", "result_status": "complete"}},
            {"event_type": "PythonOutput",
             "tool_call_id": "call_real_thing",
             "execution_status": "complete", "stdout": "[Box(...)]", "stderr": ""},
        ],
        final_answer={"answer": "", "confidence": "high",
                      "supporting_evidence": [], "tool_calls_used": 1},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    # No prefill code, no prefill output should surface
    joined = json.dumps(out)
    assert "pprint(image)" not in joined
    assert "<PIL..." not in joined
    assert "detect_objects" in joined  # the real call did make it


def test_qwen_serialize_execute_python_pairs_with_python_output_only():
    """execute_python's nested ToolResult is just 'status: complete' — the
    real output arrives in the subsequent PythonOutput event. Emitting both
    would double-count."""
    trace = AnnotateTrace(
        image_ref="x", question="q", system_prompt="",
        events=[
            {"event_type": "ToolCallEvent",
             "tool_call_id": "c1", "name": "execute_python",
             "arguments": {"code": "print('hi')"},
             "result": {"tool_call_id": "c1", "content": "status: complete",
                        "result_status": "complete"}},
            {"event_type": "PythonOutput", "tool_call_id": "c1",
             "execution_status": "complete", "stdout": "hi\n", "stderr": ""},
        ],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 1},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
    # Exactly ONE tool result — from PythonOutput, not the nested "status: complete"
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"].strip() == "hi"
    assert "status: complete" not in tool_msgs[0]["content"]


def test_qwen_serialize_return_result_has_no_paired_tool_result():
    """return_result is the completion signal; its nested 'Result accepted'
    is protocol chatter, not training-useful. The final assistant tool_call
    should stand alone."""
    trace = AnnotateTrace(
        image_ref="x", question="q", system_prompt="",
        events=[
            {"event_type": "ToolCallEvent",
             "tool_call_id": "final", "name": "return_result",
             "arguments": {"result": "SpatialAnswer(answer='...', ...)"},
             "result": {"tool_call_id": "final",
                        "content": "Result accepted (inline).",
                        "result_status": "complete"}},
        ],
        final_answer={"answer": "...", "confidence": "high",
                      "supporting_evidence": [], "tool_calls_used": 1},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    roles = [m["role"] for m in out["messages"]]
    assert "assistant" in roles     # return_result did emit
    assert "tool" not in roles       # no paired tool result


def test_qwen_serialize_short_ids_replace_thought_token_ids():
    """NOOA embeds Gemini's thinking-model reasoning tokens in tool_call_ids.
    They're hundreds of chars of opaque base64. Remap to sequential call_1,
    call_2, ... within one trace so training rows are readable + consistent."""
    long_id_1 = "call_5ed555abb9__thought__" + "x" * 200
    long_id_2 = "call_ae0de849__thought__" + "y" * 200
    trace = AnnotateTrace(
        image_ref="x", question="q", system_prompt="",
        events=[
            {"event_type": "ToolCallEvent", "tool_call_id": long_id_1,
             "name": "execute_python", "arguments": {"code": "a=1"}},
            {"event_type": "PythonOutput", "tool_call_id": long_id_1,
             "execution_status": "complete", "stdout": "", "stderr": ""},
            {"event_type": "ToolCallEvent", "tool_call_id": long_id_2,
             "name": "execute_python", "arguments": {"code": "b=2"}},
            {"event_type": "PythonOutput", "tool_call_id": long_id_2,
             "execution_status": "complete", "stdout": "", "stderr": ""},
        ],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 2},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    # No long id survives anywhere in the output
    joined = json.dumps(out)
    assert long_id_1 not in joined
    assert long_id_2 not in joined
    # Sequential short ids appear
    assert "call_1" in joined
    assert "call_2" in joined
    # Pairs match: the first assistant tool_call's id equals the first tool msg's id
    asst_msgs = [m for m in out["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
    tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
    assert asst_msgs[0]["tool_calls"][0]["id"] == tool_msgs[0]["tool_call_id"]
    assert asst_msgs[1]["tool_calls"][0]["id"] == tool_msgs[1]["tool_call_id"]


def test_qwen_serialize_uses_trace_question_not_task_prompt():
    """NOOA's Task.prompt is a runtime-formatted template (starts with
    '## Task: <method_name>' and dumps the docstring). The actual user
    question is passed as a Python variable and never lives in Task.prompt.
    For clean Qwen training input, use trace.question."""
    trace = AnnotateTrace(
        image_ref="/data/img.jpg",
        question="How far apart are the two workers?",
        system_prompt="",
        events=[
            {"event_type": "Task",
             "prompt": "## Task: annotate\n\n[whole docstring template here]",
             "images": []},
        ],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 0},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    text_blocks = [b for m in out["messages"] if m["role"] == "user"
                   for b in m["content"] if b.get("type") == "text"]
    assert text_blocks[0]["text"] == "How far apart are the two workers?"
    # The NOOA template text must NOT appear
    joined = json.dumps(out)
    assert "## Task: annotate" not in joined


def test_qwen_serialize_tool_call_without_nested_result_still_emits_call():
    """ToolCallEvent.result is Optional — an in-flight call (result not yet
    filled) should still emit the assistant tool_call message, just no tool
    tool_result. Rare but possible if annotate() is captured mid-stream."""
    trace = AnnotateTrace(
        image_ref="x", question="q", system_prompt="",
        events=[
            {"event_type": "ToolCallEvent", "tool_call_id": "c1",
             "name": "detect_objects", "arguments": {"phrase": "worker"},
             "result": None},
        ],
        final_answer={"answer": "", "confidence": "low",
                      "supporting_evidence": [], "tool_calls_used": 1},
        wall_clock_s=0.0,
    )
    out = qwen_vl_serialize(trace)
    roles = [m["role"] for m in out["messages"]]
    assert "assistant" in roles
    assert "tool" not in roles


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
