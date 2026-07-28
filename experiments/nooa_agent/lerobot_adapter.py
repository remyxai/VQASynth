"""VlmClient adapter — plug SpatialAnnotator into lerobot's steerable pipeline.

lerobot's annotation pipeline (huggingface/lerobot#4036 and the earlier
merged ``Vqa``/``Plan`` modules) uses a shared ``VlmClient`` protocol whose
entire surface is one method::

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]

That protocol is durable — it's shared across every annotation module in
the pipeline, so any restructuring would break the whole surface at once.
The specifics of any individual module (ECoT's 4-field schema, PR #4036's
config keys, the contact-sheet-per-anchor pattern) are PR-review-fragile
and deliberately NOT reflected in this adapter. Whatever JSON schema the
caller's prompt specifies, we return that JSON; we don't parse or rewrite
anything.

That means:
- If PR #4036 merges as-is, this adapter drops in directly.
- If ECoT's schema is restructured in review, this adapter still works
  unchanged — only the prompt string the caller passes changes.
- If someone builds a completely different annotation module on the same
  ``VlmClient`` protocol (a new reward-shaper, a task-verifier, whatever),
  this adapter serves that too.

See the README's ``Using as a lerobot VlmClient`` section for an end-to-end
standalone example that doesn't require lerobot to be installed.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence


def _extract_image_and_prompt(messages: Sequence[dict]) -> tuple[Any, str]:
    """Pull image + text prompt from the LAST user turn in OpenAI-style
    multimodal messages. Supports both ``type: image`` and ``type: image_url``
    block variants.

    Content can also be a plain string (text-only turn); returns (None, text).
    Earlier turns are treated as context and ignored for image/prompt.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return None, content
        if isinstance(content, list):
            image = None
            text_parts: list[str] = []
            for block in content:
                btype = block.get("type", "")
                if btype == "image" and image is None:
                    image = block.get("image")
                elif btype == "image_url" and image is None:
                    val = block.get("image_url")
                    image = val["url"] if isinstance(val, dict) and "url" in val else val
                elif btype == "text":
                    text_parts.append(block.get("text", ""))
            return image, "\n".join(t for t in text_parts if t)
        break
    return None, ""


def _strip_to_json(text: str) -> Any:
    """Extract a JSON object from LLM text output.

    Handles the same three cases lerobot's own ``vlm_client._strip_to_json``
    handles — <think> blocks (Qwen3 thinking-mode), ``` fences, and
    JSON-embedded-in-prose — so our output shape matches what their pipeline
    already tolerates.

    Raises ``ValueError`` if no balanced JSON object can be extracted.
    """
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        last_fence = text.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            text = text[first_nl + 1 : last_fence].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced { ... } block, respecting string literals.
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in text: {text[:200]!r}")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced JSON in text: {text[:200]!r}")


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from sync context.

    Safe whether or not an event loop is already running. If there IS one,
    dispatches to a fresh loop on a worker thread so we don't recurse into
    the caller's loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@dataclass
class SpatialAnnotatorVlmClient:
    """Satisfy lerobot's ``VlmClient`` protocol via ``SpatialAnnotator``.

    Drop-in replacement for the default Qwen-VL backend (or any other
    ``VlmClient`` implementation) in lerobot's steerable annotation pipeline.
    Every call runs the full ``SpatialAnnotator.annotate()`` loop — CodeAct
    reasoning + tool-grounded detection + metric depth + 3D distance where
    the LLM composes them — before the JSON response is returned.

    Standalone example (no lerobot required)::

        from experiments.nooa_agent.spatial_annotator import SpatialAnnotator
        from experiments.nooa_agent.lerobot_adapter import SpatialAnnotatorVlmClient

        agent = SpatialAnnotator(llm=llm, max_iterations=8)
        vlm = SpatialAnnotatorVlmClient(annotator=agent)

        result = vlm.generate_json([[{
            "role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": (
                    'Reply with strictly valid JSON: '
                    '{"summary": "<one sentence>", "worker_count": <int>}'
                )},
            ],
        }]])
        # → [{"summary": "...", "worker_count": 2}]

    ``max_new_tokens`` and ``temperature`` are accepted for protocol compat
    but ignored — those knobs belong to the underlying LLM client and are
    already configured through the ``SpatialAnnotator`` construction path.
    """

    annotator: Any   # SpatialAnnotator — Any to avoid an import cycle in tests

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        results: list[Any] = []
        for messages in messages_batch:
            image, prompt = _extract_image_and_prompt(list(messages))
            if image is None or not prompt:
                raise ValueError(
                    "SpatialAnnotatorVlmClient expects each messages list to "
                    "contain a user turn with BOTH an image block and a text "
                    f"block; got image={image!r}, prompt={prompt!r}"
                )
            answer = _run_sync(self.annotator.annotate(image, prompt))
            results.append(_strip_to_json(answer.answer))
        return results


__all__ = [
    "SpatialAnnotatorVlmClient",
    "_extract_image_and_prompt",
    "_strip_to_json",
    "_run_sync",
]
