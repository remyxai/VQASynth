"""Grounded chain-of-thought scaffolding.

Structures a reasoning trace into four causally ordered stages so that every
reasoning step is explicitly anchored to the visual entities and relations it
depends on, instead of a free-form ``<think>`` monologue that can drift into
fluent-but-unanchored text.

The four stages follow the supervision decomposition proposed in
*DRScaffold: Boosting Dense-Scene Reasoning in Lightweight Vision Language
Models* (https://arxiv.org/abs/2605.26038v1):

    1. Entity Grounding   -> <entities>   name the objects in play
    2. Relation Modeling  -> <relations>  state spatial relations between them
    3. Stepwise Reasoning -> <think>      reason, referring back to the above
    4. Answer             -> <answer>     the final answer

Stages 3 and 4 keep VQASynth's existing ``<think>``/``<answer>`` tags so the
produced trace stays drop-in compatible with the current reasoning format; the
two new leading stages are what enforce grounding. This module both *builds*
the system prompt that asks for that structure and *scores* a returned trace
for how well its reasoning is anchored to the grounded entities — the score is
usable as a downstream quality filter on synthetic CoT.
"""

import re

# Ordered (stage name, xml-ish tag) pairs. Order matters: it is the causal
# order the model must produce the stages in.
SCAFFOLD_STAGES = (
    ("entity_grounding", "entities"),
    ("relation_modeling", "relations"),
    ("stepwise_reasoning", "think"),
    ("answer", "answer"),
)

_TAG_BY_STAGE = {stage: tag for stage, tag in SCAFFOLD_STAGES}

# Short, lowercase tokens that carry no grounding signal when checking whether
# the reasoning references a named entity.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "is", "are",
    "it", "its", "with", "this", "that", "these", "those", "object", "objects",
    "scene", "image", "left", "right", "front", "back", "near", "far",
}


def build_scaffold_system_prompt(base_instructions: str = "") -> str:
    """Build the system prompt that asks for a four-stage grounded trace.

    Args:
        base_instructions: Domain instructions (e.g. VQASynth's quantitative
            distance guidance) prepended to the structural scaffold so the
            voice/task framing is preserved.

    Returns:
        The full system prompt string.
    """
    scaffold = (
        "Structure your response as four causally ordered stages, each wrapped "
        "in its own tag and produced strictly in this order:\n"
        "1. <entities>...</entities> — first ground the scene: list the "
        "specific objects you can see that are relevant to the question, with a "
        "distinguishing attribute (color, type, or location) for each.\n"
        "2. <relations>...</relations> — state the spatial relations between "
        "those grounded entities (relative position, ordering, distance) that "
        "the question depends on.\n"
        "3. <think>...</think> — reason step by step toward the answer, and in "
        "each step refer back to the entities and relations you just grounded "
        "rather than introducing unseen objects.\n"
        "4. <answer>...</answer> — give the final answer.\n"
        "Every claim in <think> must trace back to something named in "
        "<entities> or <relations>; do not invent objects that were not "
        "grounded.\n"
    )
    base = base_instructions.strip()
    if base:
        return base + "\n\n" + scaffold
    return scaffold


def parse_scaffold(reasoning_text: str) -> dict:
    """Extract each stage's content from a generated trace.

    Args:
        reasoning_text: The raw string returned by the reasoning model.

    Returns:
        Dict keyed by stage name; missing stages map to ``None``.
    """
    result = {}
    text = reasoning_text or ""
    for stage, tag in SCAFFOLD_STAGES:
        match = re.search(
            rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE
        )
        result[stage] = match.group(1).strip() if match else None
    return result


def _content_tokens(text: str) -> set:
    """Lowercased non-stopword word tokens of length >= 3."""
    tokens = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return {t for t in tokens if t not in _STOPWORDS}


def scaffold_grounding_score(reasoning_text: str) -> float:
    """Score how grounded a generated trace is, in ``[0, 1]``.

    The score blends two signals from DRScaffold's grounding objective:

    * **stage coverage** — how many of the four stages are present and
      non-empty (each stage absent or empty costs coverage); and
    * **anchoring** — whether the stepwise reasoning actually references
      entities/relations that were grounded earlier, measured as the fraction
      of reasoning content tokens that also appear in the grounding stages.

    A fluent but visually unanchored chain scores low on anchoring even when it
    fills every tag, which is exactly the failure mode the paper targets.
    """
    stages = parse_scaffold(reasoning_text)

    present = [s for s in (stages[name] for name, _ in SCAFFOLD_STAGES) if s]
    coverage = len(present) / len(SCAFFOLD_STAGES)

    grounding_tokens = _content_tokens(stages["entity_grounding"]) | _content_tokens(
        stages["relation_modeling"]
    )
    reasoning_tokens = _content_tokens(stages["stepwise_reasoning"])

    if not reasoning_tokens:
        # No reasoning content to anchor; grounding quality is undefined, so
        # the score is carried entirely by stage coverage.
        return round(coverage, 4)
    if not grounding_tokens:
        anchoring = 0.0
    else:
        overlap = reasoning_tokens & grounding_tokens
        anchoring = len(overlap) / len(reasoning_tokens)

    score = 0.5 * coverage + 0.5 * anchoring
    return round(score, 4)


def is_well_grounded(reasoning_text: str, threshold: float = 0.5) -> bool:
    """Whether a trace clears ``threshold`` on :func:`scaffold_grounding_score`."""
    return scaffold_grounding_score(reasoning_text) >= threshold
