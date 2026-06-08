"""
Tests for criterion-level rubric scoring and its wiring into BenchmarkRunner.

These exercise the integration between vqasynth.rubric_scoring (new) and
vqasynth.benchmarks.BenchmarkRunner (existing call site).
"""

from vqasynth.benchmarks import BenchmarkRunner
from vqasynth.rubric_scoring import (
    ADDITIONAL_FLOOR,
    RubricCriterion,
    aggregate_rubric,
    build_spatial_rubric,
    score_rubric,
)


class _StubMessage:
    def __init__(self, content):
        self.content = content


class _StubChoice:
    def __init__(self, content):
        self.message = _StubMessage(content)


class _StubResponse:
    def __init__(self, content):
        self.choices = [_StubChoice(content)]


class _StubCompletions:
    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _StubResponse(self._verdict)


class _StubClient:
    """Minimal OpenAI-style client that always returns a fixed verdict."""

    def __init__(self, verdict="True"):
        self.chat = type("chat", (), {"completions": _StubCompletions(verdict)})()


# --- Hierarchical aggregation -------------------------------------------------

def test_aggregate_essential_only():
    # No additional criteria: score is the essential mean.
    assert aggregate_rubric([(1.0, 1.0)], []) == 1.0
    assert aggregate_rubric([(0.0, 1.0)], []) == 0.0


def test_aggregate_essential_gates_additional():
    # A failed essential criterion caps the score, even with perfect extras.
    score = aggregate_rubric([(0.0, 1.0)], [(1.0, 1.0)])
    assert score == 0.0


def test_aggregate_additional_refines_within_ceiling():
    # Essentials pass; a failing additional criterion can only pull the score
    # down to ADDITIONAL_FLOOR of the ceiling, never above it.
    score = aggregate_rubric([(1.0, 1.0)], [(0.0, 1.0)])
    assert score == ADDITIONAL_FLOOR
    # A passing additional criterion leaves the ceiling intact.
    assert aggregate_rubric([(1.0, 1.0)], [(1.0, 1.0)]) == 1.0


def test_aggregate_none_when_unscorable():
    assert aggregate_rubric([], []) is None


# --- Deterministic (verifiable) criteria, no client ---------------------------

def test_score_rubric_verifiable_yes_no_no_client():
    rubric = [RubricCriterion("answer_polarity", "yes_no", essential=True)]
    assert score_rubric(rubric, "Yes, it is.", "yes") == 1.0
    assert score_rubric(rubric, "No.", "yes") == 0.0


def test_score_rubric_skips_judge_when_no_client():
    # Judge criterion returns None without a client and is dropped; the
    # deterministic essential criterion still produces a score.
    rubric = build_spatial_rubric("Is the cup to the left of the box?", "yes")
    score, breakdown = score_rubric(
        rubric, "Yes", "yes", question="Is the cup to the left of the box?",
        return_breakdown=True,
    )
    assert score == 1.0
    judge = [b for b in breakdown if b["kind"] == "judge"]
    assert judge and all(b["score"] is None for b in judge)


def test_build_spatial_rubric_routes_by_question_type():
    dist = build_spatial_rubric("What is the distance between the chair and the table?")
    assert dist[0].kind == "distance" and dist[0].essential
    openq = build_spatial_rubric("Describe the scene and reason about it.")
    # Open questions become a judge-essential rubric.
    assert openq[0].kind == "judge" and openq[0].essential


# --- Judge criteria with a stub client ----------------------------------------

def test_score_rubric_judge_path_with_stub_client():
    client = _StubClient(verdict="True")
    rubric = [
        RubricCriterion("answer_polarity", "yes_no", essential=True),
        RubricCriterion(
            "reasoning_sound", "judge", essential=False, prompt="ok?"
        ),
    ]
    # Essential passes (1.0) and judge approves the refinement (1.0) -> 1.0.
    score = score_rubric(rubric, "Yes", "yes", client=client)
    assert score == 1.0
    assert client.chat.completions.calls == 1


# --- Wiring into the existing BenchmarkRunner call site -----------------------

def _yes_no_items():
    return [
        {
            "id": "q1",
            "question": "Is the cup to the left of the box?",
            "answer": "yes",
            "category": "Spatial Relations",
            "subcategory": "left_right",
            "question_type": "judgment",
        }
    ]


def test_benchmark_runner_uses_rubric_when_enabled():
    runner = BenchmarkRunner(benchmarks="spatialscore", use_rubric=True)
    items = _yes_no_items()

    correct = runner.score("spatialscore", items, {"q1": "Yes, it is."})
    assert correct["per_item"][0]["score"] == 1.0

    wrong = runner.score("spatialscore", items, {"q1": "No, it is not."})
    assert wrong["per_item"][0]["score"] == 0.0


def test_benchmark_runner_rubric_off_by_default():
    # Default path leaves the existing flat-judge behavior untouched.
    runner = BenchmarkRunner(benchmarks="spatialscore")
    assert runner.use_rubric is False
