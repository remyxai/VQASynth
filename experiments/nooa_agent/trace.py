"""Trace capture + serialization for VQASynth spatial-annotation traces.

Captures the full tool-call sequence from an ``annotate()`` call and serializes
to formats consumable by downstream fine-tuning pipelines. Qwen2.5-VL /
Qwen3-VL is the primary target; the serializer is a plain callable so other
targets (Anthropic tool_use, Llama tool-calling variants) can be added as
separate functions when needed.

Design:
- ``AnnotateTrace`` — neutral capture. Holds NOOA's pydantic-dumped events
  + our summary. Format-agnostic; a single trace can be re-serialized to any
  downstream target.
- ``capture_trace(agent, ...)`` — snapshots events since ``since_event_idx``
  so multi-annotate agents don't accumulate stale events into each trace.
- ``qwen_vl_serialize`` — emits OpenAI-messages format that Qwen's HuggingFace
  chat template consumes natively via ``tokenizer.apply_chat_template()``.
- ``TraceWriter`` — append-only JSONL with the serializer plugged in.

See also: NOOA event-log investigation notes in the branch history.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AnnotateTrace:
    """Neutral, format-agnostic trace of one annotate() call."""
    image_ref: str                       # path or hash — never inline bytes
    question: str
    system_prompt: str
    events: list[dict]                   # pydantic-dumped NOOA events, in order
    final_answer: dict                   # SpatialAnswer as dict
    wall_clock_s: float
    tool_schema: list[dict] = field(default_factory=list)


def capture_trace(
    agent: Any,
    *,
    image_ref: str,
    question: str,
    system_prompt: str,
    answer: Any,
    elapsed_s: float,
    since_event_idx: int = 0,
    tool_schema: list[dict] | None = None,
) -> AnnotateTrace:
    """Snapshot NOOA events since ``since_event_idx`` and wrap into a trace.

    NOOA's ``event_manager`` accumulates events across ``annotate()`` calls on
    the same agent instance. ``since_event_idx`` is the length of the event
    log BEFORE this annotate() call — everything after is what this call
    produced. Capture the length before calling ``annotate()``:

        before = len(list(agent.event_manager.values()))
        answer = await agent.annotate(image, question)
        trace = capture_trace(agent, ..., since_event_idx=before)
    """
    all_events = list(agent.event_manager.values())
    delta = all_events[since_event_idx:]
    return AnnotateTrace(
        image_ref=image_ref,
        question=question,
        system_prompt=system_prompt,
        events=[_dump_event(e) for e in delta],
        final_answer=_dump_answer(answer),
        wall_clock_s=elapsed_s,
        tool_schema=tool_schema or [],
    )


def _dump_event(event: Any) -> dict:
    """NOOA events are pydantic BaseModels — ``model_dump(mode='json')``
    gives us a JSON-safe dict. Fallback for anything else keeps repr()."""
    if hasattr(event, "model_dump"):
        try:
            return event.model_dump(mode="json")
        except Exception as e:
            return {"event_type": getattr(event, "event_type", "unknown"),
                    "_dump_error": str(e), "_repr": repr(event)[:200]}
    return {"event_type": getattr(event, "event_type", "unknown"),
            "_repr": repr(event)[:200]}


def _dump_answer(answer: Any) -> dict:
    return {
        "answer": getattr(answer, "answer", ""),
        "confidence": getattr(answer, "confidence", "unknown"),
        "supporting_evidence": list(getattr(answer, "supporting_evidence", [])),
        "tool_calls_used": getattr(answer, "tool_calls_used", 0),
    }


# ── Qwen2.5-VL / Qwen3-VL serializer ─────────────────────────────────
#
# Emits OpenAI-messages format that Qwen's HF chat template consumes via
# ``tokenizer.apply_chat_template(msgs, tokenize=False)``. Same shape both
# Qwen2.5-VL and Qwen3-VL expect. Image content blocks reference the file
# by path (``image_ref``); the training data loader resolves them.


# NOOA event_type strings are the CLASS NAME (auto-derived via
# EventBase.model_post_init from type(self).__name__). Verified against
# nooa/context_blocks/events.py + nooa/events.py — no explicit overrides.
# The codeact_event_sequence.py example uses lowercase strings that would
# never match; treat that example as broken and match source-of-truth.

_TASK = "Task"
_TOOL_CALL = "ToolCallEvent"
_PYTHON_OUTPUT = "PythonOutput"
_ASSISTANT_TEXT_EVENTS = frozenset({"Message", "LLMOutput", "TextOnlyReply", "AssistantEvent"})
# Skipped: DebugTrace (METADATA), Error (retry signal shown to LLM for retry
# but not canonical dialog), Feedback (execution-feedback nudge), Reasoning
# (legacy — no longer emitted), lifecycle hooks (BeforeTurn/AfterTurn/
# BeforeAgentCall/AfterAgentCall/LLMCallStart/LLMCallEnd), Notification,
# Summary (compaction artifact), TuiSession*, Metadata subclasses.


def qwen_vl_serialize(trace: AnnotateTrace) -> dict:
    """Serialize AnnotateTrace → OpenAI-messages dict for Qwen VL fine-tuning.

    NOOA specifics that shape this transform:
    - ``ToolCallEvent`` holds its ``result: ToolResult | None`` INLINE, not
      as a separate event. We emit both the assistant tool_call and the tool
      tool_result from the one ToolCallEvent.
    - ``PythonOutput`` is a separate event, ONLY emitted for the CodeAct
      ``execute_python`` path. For our SpatialAnnotator (typed tool methods,
      no CodeAct), we'll see ToolCallEvent-with-inline-result, not PythonOutput.
      Handled both ways so a CodeAct-enabled variant would just work.
    """
    messages: list[dict] = []

    if trace.system_prompt or trace.tool_schema:
        sys_content = trace.system_prompt or ""
        if trace.tool_schema:
            sys_content += "\n\n# Tools\n" + json.dumps(trace.tool_schema, indent=2)
        messages.append({"role": "system", "content": sys_content})

    for ev in trace.events:
        etype = ev.get("event_type", "")

        if etype == _TASK:
            # Initial user turn: image + question. NOOA's Task.images is
            # list[dict] of multimodal content blocks (opaque to us — could
            # be data URLs, file refs, or PIL wrappers). We ignore the inline
            # blocks and reference the image by path via trace.image_ref.
            content: list[dict] = [{"type": "image", "image": trace.image_ref}]
            content.append({"type": "text", "text": ev.get("prompt", trace.question)})
            messages.append({"role": "user", "content": content})

        elif etype == _TOOL_CALL:
            # Assistant emits the tool_call
            call_id = ev.get("tool_call_id", "")
            args = ev.get("arguments", {})
            args_str = json.dumps(args) if isinstance(args, (dict, list)) else str(args)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": ev.get("name", "unknown"),
                        "arguments": args_str,
                    },
                }],
            })
            # Then emit the tool result from the NESTED result field
            nested = ev.get("result")
            if isinstance(nested, dict):
                # ToolResult carries `content: str` + `result_status`
                messages.append({
                    "role": "tool",
                    "tool_call_id": nested.get("tool_call_id", call_id),
                    "content": nested.get("content", ""),
                })

        elif etype == _PYTHON_OUTPUT:
            # CodeAct execute_python output — separate event, ResultStatus
            # enum + stdout/stderr fields. Prefer stderr if execution errored.
            call_id = ev.get("tool_call_id", "")
            status = ev.get("execution_status", "")
            stdout = ev.get("stdout", "") or ""
            stderr = ev.get("stderr", "") or ""
            content = stderr if str(status).lower().endswith("error") and stderr else stdout
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(content),
            })

        elif etype in _ASSISTANT_TEXT_EVENTS:
            text = ev.get("content", "")
            if text:
                messages.append({"role": "assistant", "content": text})

    return {
        "messages": messages,
        "meta": {
            "image_ref": trace.image_ref,
            "question": trace.question,
            "confidence": trace.final_answer.get("confidence"),
            "tool_calls_used": trace.final_answer.get("tool_calls_used"),
            "wall_clock_s": trace.wall_clock_s,
        },
    }


# ── Writer: streaming JSONL ──────────────────────────────────────────


class TraceWriter:
    """Append-only JSONL writer with a pluggable serializer.

    Streams one line per trace so batch runs are resumable and greppable.
    Use as a context manager or call ``.close()`` when done.
    """

    def __init__(
        self,
        path: str,
        serializer: Callable[[AnnotateTrace], dict] = qwen_vl_serialize,
    ):
        self._path = path
        self._serializer = serializer
        self._f = None

    def open(self):
        if self._f is None:
            self._f = open(self._path, "a")
        return self

    def write(self, trace: AnnotateTrace) -> None:
        if self._f is None:
            self.open()
        line = json.dumps(self._serializer(trace), default=str)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()


__all__ = ["AnnotateTrace", "capture_trace", "qwen_vl_serialize", "TraceWriter"]
