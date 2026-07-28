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

# CodeAct-specific conventions verified against a real event dump:
# - Prefill events (tool_call_id starts with "prefill_") are input-inspection
#   noise emitted at the top of every generation — not substantive reasoning.
# - execute_python's nested ToolResult carries only "status: complete"; the
#   real code output arrives in a subsequent PythonOutput event. Emitting
#   both would produce doubled tool-result messages.
# - return_result signals completion inline; its nested "Result accepted"
#   ToolResult is protocol chatter, not training-useful.
_PREFILL_ID_PREFIX = "prefill_"
_TOOL_NAMES_WITH_SEPARATE_OUTPUT_EVENT = frozenset({"execute_python"})
_TOOL_NAMES_SKIP_NESTED_RESULT = (
    frozenset({"return_result"}) | _TOOL_NAMES_WITH_SEPARATE_OUTPUT_EVENT
)


def _make_id_shortener():
    """Remap NOOA's long, thought-token-embedded tool_call_ids to short
    monotonic ids stable within one trace. The originals contain the LLM's
    reasoning tokens as opaque base64 and are hundreds of chars long."""
    remap: dict[str, str] = {}

    def short(original: str) -> str:
        if original not in remap:
            remap[original] = f"call_{len(remap) + 1}"
        return remap[original]

    return short


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
    shorten = _make_id_shortener()

    if trace.system_prompt or trace.tool_schema:
        sys_content = trace.system_prompt or ""
        if trace.tool_schema:
            sys_content += "\n\n# Tools\n" + json.dumps(trace.tool_schema, indent=2)
        messages.append({"role": "system", "content": sys_content})

    # Emit the user turn once (image + question). NOOA's Task event carries
    # a NOOA-formatted prompt template (not the user's question) and empty
    # Task.images; instead we synthesize the training-shape input from
    # trace.image_ref + trace.question. Skip further Task events below.
    saw_task = False

    for ev in trace.events:
        etype = ev.get("event_type", "")
        tid = ev.get("tool_call_id", "") or ""

        # Skip CodeAct's input-inspection prefill — always emitted, never
        # substantive. Applies to both the ToolCallEvent and the paired
        # PythonOutput; both share the "prefill_*" id.
        if tid.startswith(_PREFILL_ID_PREFIX):
            continue

        if etype == _TASK:
            if not saw_task:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": trace.image_ref},
                        {"type": "text", "text": trace.question},
                    ],
                })
                saw_task = True
            continue

        if etype == _TOOL_CALL:
            name = ev.get("name", "unknown")
            call_id = shorten(tid) if tid else f"call_{len(messages)}"
            args = ev.get("arguments", {})
            args_str = json.dumps(args) if isinstance(args, (dict, list)) else str(args)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }],
            })
            # Only emit nested result for direct-tool-call events. For
            # execute_python + return_result the nested result is protocol
            # chatter — the real output arrives via PythonOutput (or in the
            # case of return_result, the trace simply ends).
            if name not in _TOOL_NAMES_SKIP_NESTED_RESULT:
                nested = ev.get("result")
                if isinstance(nested, dict):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": nested.get("content", ""),
                    })
            continue

        if etype == _PYTHON_OUTPUT:
            call_id = shorten(tid) if tid else f"call_{len(messages)}"
            status = ev.get("execution_status", "")
            stdout = ev.get("stdout", "") or ""
            stderr = ev.get("stderr", "") or ""
            content = stderr if str(status).lower().endswith("error") and stderr else stdout
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(content),
            })
            continue

        if etype in _ASSISTANT_TEXT_EVENTS:
            text = ev.get("content", "")
            if text:
                messages.append({"role": "assistant", "content": text})
            continue

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
