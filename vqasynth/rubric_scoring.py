"""
Criterion-level rubric scoring for partially-verifiable spatial-reasoning QA.

Where ``vqasynth.evaluation.llm_judge`` returns a single binary verdict for a
whole prediction, this module decomposes scoring into a *rubric* of named
criteria and scores each one independently, then combines them with
hierarchical aggregation.

The design is adapted from "Reinforcement Learning with Robust Rubric Rewards"
(RLR^3, arXiv:2605.30244). Two ideas from that work are ported here as an
*evaluation* primitive (VQASynth uses the judge to score data, not to train):

  * **Criterion-level verification with two execution paths.** Verifiable
    criteria (a metric distance, a yes/no polarity, an object choice) are
    routed to the deterministic extractors+checkers already living in
    ``vqasynth.evaluation``. Non-verifiable criteria (grounding, logical flow)
    are routed to an LLM-as-a-Judge focused on that single criterion.
  * **Minimal exposure.** The deterministic path extracts answers from the
    prediction text only -- it never reveals the ground truth to a model. The
    judge path is given text only (no image). Both reduce exploitable false
    positives, as the paper reports.
  * **Hierarchical aggregation.** Essential criteria gate the score; additional
    criteria can only refine it within the ceiling the essentials set, so an
    answer cannot earn credit for nice reasoning while getting the spatial
    fact wrong.

This module is intentionally training-free: it consumes the same primitives the
benchmark scorers use and is invoked by ``BenchmarkRunner`` as a richer
replacement for the flat judge fallback.
"""

from dataclasses import dataclass

from vqasynth.evaluation import (
    classify_question,
    extract_numeric_with_unit,
    score_choice,
    score_distance,
    score_distance_mra,
    score_yes_no,
)

# Additional (non-essential) criteria can only pull a score down to this
# fraction of the essential ceiling -- they refine, they do not rescue.
ADDITIONAL_FLOOR = 0.5


@dataclass
class RubricCriterion:
    """
    One scoring criterion in a rubric.

    Attributes:
        name: Human-readable identifier (appears in the score breakdown).
        kind: Execution path. Deterministic verifiers:
            "distance", "distance_mra", "yes_no", "choice", "has_units".
            Non-verifiable: "judge" (uses ``prompt`` against an LLM).
        essential: Essential criteria gate the score; additional criteria
            (essential=False) only refine within the essential ceiling.
        weight: Relative weight within its tier (essential vs additional).
        prompt: For "judge" criteria, the single thing the judge must verify.
    """

    name: str
    kind: str
    essential: bool = True
    weight: float = 1.0
    prompt: str = ""


# Deterministic verifier dispatch. Each takes (prediction, ground_truth,
# question) and returns a score in [0, 1] or None when extraction fails.
def _verify_distance(pred, gt, question):
    return score_distance(pred, gt, tolerance=2.0)


def _verify_distance_mra(pred, gt, question):
    return score_distance_mra(pred, gt)


def _verify_yes_no(pred, gt, question):
    return score_yes_no(pred, gt)


def _verify_choice(pred, gt, question):
    return score_choice(pred, gt, question=question)


def _verify_has_units(pred, gt, question):
    """Object/measurement grounding: did the prediction commit to a unit?"""
    return 1.0 if extract_numeric_with_unit(pred) is not None else 0.0


_VERIFIERS = {
    "distance": _verify_distance,
    "distance_mra": _verify_distance_mra,
    "yes_no": _verify_yes_no,
    "choice": _verify_choice,
    "has_units": _verify_has_units,
}


CRITERION_JUDGE_SYSTEM = (
    "You verify ONE criterion of a spatial-reasoning answer.\n"
    "Criterion: {criterion}\n"
    "You are given only text (no image) so your scoring stays faithful.\n"
    "Answer 'True' only if the criterion is fully satisfied, otherwise 'False'.\n"
    "Respond with only 'True' or 'False'."
)


def _judge_criterion(client, model, criterion, question, prediction, ground_truth):
    """
    Score a single non-verifiable criterion via an LLM-as-a-Judge.

    Minimal exposure: text only, no image; the judge sees just this one
    criterion. Returns 1.0, 0.0, or None on failure / no client.
    """
    if client is None:
        return None
    system = CRITERION_JUDGE_SYSTEM.format(criterion=criterion.prompt)
    user = (
        f"Question: {question}\n"
        f"Prediction: {prediction}\n"
        f"Ground Truth: {ground_truth}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=10,
            temperature=0,
        )
        result = response.choices[0].message.content.strip().lower()
        return 1.0 if "true" in result else 0.0
    except Exception:
        return None


def _execute_criterion(criterion, prediction, ground_truth, question, client, model):
    """Route a criterion to its execution path. Returns score or None."""
    if criterion.kind == "judge":
        return _judge_criterion(
            client, model, criterion, question, prediction, ground_truth
        )
    verifier = _VERIFIERS.get(criterion.kind)
    if verifier is None:
        return None
    return verifier(prediction, ground_truth, question)


def _weighted_mean(pairs):
    """Mean of (score, weight) pairs; None if empty or zero total weight."""
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return None
    return sum(s * w for s, w in pairs) / total_w


def aggregate_rubric(essential, additional):
    """
    Hierarchical aggregation of (score, weight) pairs.

    Essential criteria set the ceiling; additional criteria refine within it
    (they can pull the score down to at most ``ADDITIONAL_FLOOR`` of the
    ceiling, never above it). Returns a score in [0, 1], or None if nothing
    was scorable.
    """
    base = _weighted_mean(essential)
    add = _weighted_mean(additional)

    if base is None and add is None:
        return None
    if base is None:
        # No essential criteria scorable: fall back to additional alone.
        base = 1.0
    if add is None:
        return base
    return base * (ADDITIONAL_FLOOR + (1.0 - ADDITIONAL_FLOOR) * add)


def score_rubric(
    criteria,
    prediction,
    ground_truth,
    question="",
    client=None,
    model="gpt-4o",
    return_breakdown=False,
):
    """
    Score a prediction against a rubric of criteria.

    Args:
        criteria: List of ``RubricCriterion``.
        prediction: Model prediction text.
        ground_truth: Reference answer text.
        question: Original question (used by some verifiers and the judge).
        client: Optional OpenAI-style client for "judge" criteria.
        model: Judge model name.
        return_breakdown: If True, also return the per-criterion breakdown.

    Returns a score in [0, 1] (or None if no criterion was scorable). When
    ``return_breakdown`` is True returns ``(score, breakdown_list)``.
    """
    essential = []
    additional = []
    breakdown = []

    for criterion in criteria:
        s = _execute_criterion(
            criterion, prediction, ground_truth, question, client, model
        )
        breakdown.append(
            {
                "name": criterion.name,
                "kind": criterion.kind,
                "essential": criterion.essential,
                "score": s,
            }
        )
        if s is None:
            continue
        if criterion.essential:
            essential.append((s, criterion.weight))
        else:
            additional.append((s, criterion.weight))

    overall = aggregate_rubric(essential, additional)
    if return_breakdown:
        return overall, breakdown
    return overall


_REASONING_PROMPT = (
    "Does the prediction give a coherent spatial justification (referencing "
    "the relevant objects and their relative positions) without contradicting "
    "its own conclusion?"
)


def build_spatial_rubric(question, ground_truth=""):
    """
    Build a default rubric for a VQASynth-style spatial question.

    The rubric is keyed off ``classify_question``: verifiable answer formats
    get a deterministic essential criterion plus refinement criteria; open /
    chain-of-thought questions get a judge-based essential answer check with
    grounding and logical-flow refinements -- the partially-verifiable,
    multi-criteria setting RLR^3 targets.
    """
    qtype = classify_question(question)

    if qtype in ("distance", "vertical_distance", "horizontal_distance", "measurement"):
        return [
            RubricCriterion("metric_accuracy", "distance", essential=True),
            RubricCriterion("units_grounded", "has_units", essential=False, weight=0.5),
            RubricCriterion(
                "reasoning_sound", "judge", essential=False, prompt=_REASONING_PROMPT
            ),
        ]

    if qtype == "comparison_yn":
        return [
            RubricCriterion("answer_polarity", "yes_no", essential=True),
            RubricCriterion(
                "reasoning_sound", "judge", essential=False, prompt=_REASONING_PROMPT
            ),
        ]

    if qtype == "comparison_choice":
        return [
            RubricCriterion("object_selection", "choice", essential=True),
            RubricCriterion(
                "reasoning_sound", "judge", essential=False, prompt=_REASONING_PROMPT
            ),
        ]

    # Open-ended / chain-of-thought: fully partially-verifiable -> judge rubric.
    return [
        RubricCriterion(
            "final_answer_correct",
            "judge",
            essential=True,
            prompt="Is the final answer correct given the ground truth?",
        ),
        RubricCriterion(
            "object_grounding",
            "judge",
            essential=False,
            prompt="Does the reasoning correctly identify and ground the objects "
            "the question is about?",
        ),
        RubricCriterion(
            "logical_flow",
            "judge",
            essential=False,
            prompt="Do the reasoning steps follow logically toward the stated answer?",
        ),
    ]
