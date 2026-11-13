"""SFT data formatting for the VQASynth++ Phase-5 cold-start.

Turns gated pipeline rows (a grounded trace + its question) into chat examples for
Stage-1 SFT: ``user(image + question) -> assistant(trace target)``. The trace is
the r1_reasoning output (``SoftTraceReasoner`` or ``GroundedBBoxTraceReasoner``);
the image stays in the row's image column and is referenced by a placeholder in
the user turn, matching the pipeline's ``messages`` convention.
"""
from __future__ import annotations


def extract_question(messages):
    """Return the first user text from a pipeline ``messages`` list, or None."""
    if not isinstance(messages, (list, tuple)):
        return None
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            for c in m.get("content", []) or []:
                if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                    return c["text"]
    return None


def to_sft_example(question, trace, system=None):
    """Build a chat SFT example ``{"messages": [...]}`` with an image placeholder in
    the user turn and the trace as the assistant target. Returns None if the
    question or trace is empty (so the row is dropped)."""
    if not question or not trace:
        return None
    messages = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append(
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    )
    messages.append({"role": "assistant", "content": [{"type": "text", "text": trace}]})
    return {"messages": messages}


def format_sft_dataset(rows, question_of, trace_of, system=None):
    """Map rows -> SFT examples, dropping rows with no question or empty trace.

    ``question_of`` / ``trace_of`` extract those fields from a row (e.g. the gated
    output of ``trace_validation.filter_dataset``). Returns ``(examples, n_dropped)``.
    """
    examples, dropped = [], 0
    for row in rows:
        ex = to_sft_example(question_of(row), trace_of(row), system=system)
        if ex is None:
            dropped += 1
            continue
        examples.append(ex)
    return examples, dropped
