"""Structural tests for vqasynth.rubric_credit.

Verifies per-proposition partial credit (VF / RC / IF) and error localization
against hand-computed small examples — no CUDA, no model download, no network.
Mirrors the dependency-free style of tests/test_visual_credit.py.

Integration with the PRE-EXISTING ``vqasynth`` modules is asserted directly:
the decomposition must key off ``vqasynth.evaluation.classify_question`` and
agree with its extractors / scorers, and the wiring into
``vqasynth.benchmarks.BenchmarkRunner.score`` (this PR's call site) must attach
the rubric block to the report the multi-benchmark evaluation stage already
produces. Those modules are not created by this PR, so the composition tests
exercise real integrated behavior rather than self-testing the new file.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from vqasynth.rubric_credit import (
    COMPONENTS,
    breakdown_by,
    build_rubric,
    decompose,
    format_localizations,
    format_rubric_report,
    scalar_score,
    score_item,
    score_items,
    split_reasoning,
)

# --- decomposition -----------------------------------------------------------

YN_Q = "Is the cup to the left of the book from the viewer's perspective?"
YN_GOLD = "Yes, the cup is to the left of the book."
DIST_Q = "How far is the chair from the bookshelf?"
DIST_GOLD = "the chair is 1 meter from the bookshelf."


def test_decompose_lifts_relation_measurement_polarity():
    props = decompose(YN_GOLD)
    kinds = [p.kind for p in props]
    assert "relation" in kinds and "polarity" in kinds

    relation = next(p for p in props if p.kind == "relation")
    assert relation.relation == "left"
    assert relation.objects[0] == "cup"
    # The span points at the clause inside the decomposed text itself.
    assert YN_GOLD[relation.span[0] : relation.span[1]].lower().startswith(
        "the cup is to the left of the book"
    )

    measured = decompose(DIST_GOLD)
    measurement = next(p for p in measured if p.kind == "measurement")
    assert (measurement.value, measurement.unit) == (1.0, "meter")


def test_decompose_handles_hedged_template_wording():
    # A false-response template: "either to the right of or directly aligned
    # with the [B]" must still surface the 'right' relation.
    gold = "In fact, the cup is either to the right of or directly aligned with the book."
    relation = next(p for p in decompose(gold) if p.kind == "relation")
    assert relation.relation == "right"
    assert relation.objects[0] == "cup"


def test_build_rubric_falls_back_to_choice_object():
    # Comparative questions whose gold names the winner outright ("the table
    # is to the left") still yield a scorable proposition via the candidate
    # parse that vqasynth.evaluation.score_choice itself uses.
    question = "Which is more to the left, the chair or the table?"
    props = build_rubric(question, "Positioned to the left is the table.")
    assert any(p.kind == "choice" for p in props)
    assert any(p.objects == ("table",) for p in props)


# --- VF: partial credit + antonym localization -------------------------------

def test_vf_full_credit_when_relation_asserted():
    score = score_item(YN_Q, YN_GOLD, "Yes, the cup is to the left of the book.")
    assert score.vf == 1.0
    assert score.failures() == []


def test_vf_antonym_failure_localizes_to_the_offending_clause():
    response = "Yes. The cup is to the right of the book"
    score = score_item(YN_Q, YN_GOLD, response)
    relation_item = next(r for r in score.items if r.proposition.kind == "relation")
    assert relation_item.credit == 0.0
    assert "antonym 'right'" in relation_item.detail
    # The span points at the clause in the RESPONSE, not in the reference.
    start, end = relation_item.evidence_span
    assert response[start:end] == "The cup is to the right of the book"


def test_vf_partial_credit_beats_scalar_zero_on_a_near_miss_distance():
    # 1.2 m against a 1 m reference: the all-or-nothing MRA score is partial,
    # and the rubric adds the IF + RC items on top of it.
    score = score_item(DIST_Q, DIST_GOLD, "The chair is 1.2 meters from the bookshelf.")
    assert 0.0 < score.vf < 1.0
    assert score.instruction_following == 1.0  # unit-bearing number present
    assert score.rubric_score > 0.0


def test_instruction_following_flags_missing_unit():
    score = score_item(DIST_Q, DIST_GOLD, "pretty close")
    assert score.instruction_following == 0.0
    assert score.vf == 0.0


# --- RC: reasoning / final-answer consistency --------------------------------

CONTRADICTING = (
    "<think>The cup is to the right of the book</think> "
    "<answer>The cup is to the left of the book</answer>"
)


def test_rc_catches_a_chain_that_contradicts_its_answer():
    score = score_item(YN_Q, YN_GOLD, CONTRADICTING)
    assert score.rc == 0.0
    assert score.vf == 1.0  # the final answer itself is faithful
    localized = format_localizations(score)
    assert "RC" in localized
    assert "antonym 'right'" in localized
    assert "The cup is to the right of the book" in localized


def test_rc_localizes_into_the_full_response_not_the_reasoning_slice():
    score = score_item(YN_Q, YN_GOLD, CONTRADICTING)
    rc_item = next(r for r in score.items if r.component == "RC")
    start, end = rc_item.evidence_span
    assert CONTRADICTING[start:end] == "The cup is to the right of the book"


def test_rc_is_vacuous_without_a_reasoning_prefix():
    score = score_item(YN_Q, YN_GOLD, "Yes, the cup is to the left of the book.")
    assert score.rc is None  # no chain of thought -> RC not scored


def test_rc_consistent_chain_earns_full_credit():
    response = (
        "<think>The cup is to the left of the book.</think> "
        "<answer>The cup is to the left of the book.</answer>"
    )
    assert score_item(YN_Q, YN_GOLD, response).rc == 1.0


def test_rc_flags_a_distance_that_moves_between_reasoning_and_answer():
    response = "The chair is about 200 centimeters from the bookshelf. Answer: 1 meter"
    score = score_item(DIST_Q, DIST_GOLD, response)
    rc_item = next(r for r in score.items if r.component == "RC")
    assert rc_item.credit == 0.0
    assert "200 centimeter" in rc_item.detail


# --- split_reasoning: markers this repo already produces ---------------------

@pytest.mark.parametrize(
    "text,expected_final",
    [
        ("<think>a</think> <answer>b</answer>", "b"),
        ("Reasoning. <conclusion>yes</conclusion>", "yes"),
        ("Some reasoning. Answer: 1 meter", "1 meter"),
    ],
)
def test_split_reasoning_recognizes_repo_markers(text, expected_final):
    reasoning, final, offset = split_reasoning(text)
    assert final == expected_final
    assert reasoning
    assert text[offset : offset + len(final)] == final


def test_split_reasoning_last_sentence_fallback():
    reasoning, final, offset = split_reasoning("First sentence. Second sentence.")
    assert reasoning == "First sentence."
    assert final == "Second sentence."


# --- aggregation -------------------------------------------------------------

def _three_triples():
    return [
        (YN_Q, YN_GOLD, "Yes, the cup is to the left of the book."),      # all satisfied
        (YN_Q, YN_GOLD, "No, the cup is not to the left of the book."),   # VF partial
        (DIST_Q, DIST_GOLD, "pretty close"),                              # VF + IF fail
    ]


def test_score_items_aggregates_components():
    report = score_items(_three_triples())
    assert report.total == 3
    # Item 0 fully faithful, item 1 names the relation but flips the yes/no
    # polarity (0.5), item 2 asserts nothing grounded (0.0).
    assert report.vf == pytest.approx(0.5)
    assert report.instruction_following == pytest.approx(2 / 3)
    # Item 0 has no reasoning prefix -> RC not scored, and must not drag the mean.
    assert report.reasoning_consistency == 0.0
    assert 0.0 < report.rubric_score < 1.0
    assert report.propositions_per_item > 1.0


def test_rubric_score_bounds_scalar_when_partial_credit_applies():
    report = score_items(_three_triples())
    # The rubric grants partial credit the all-or-nothing scorer withholds.
    assert report.rubric_score > report.scalar_accuracy


def test_breakdown_by_groups_by_question_type():
    report = score_items(_three_triples())
    by_type = breakdown_by(report.per_item, key_fn=lambda s: s.question_type)
    assert set(by_type) == {"comparison_yn", "distance"}
    assert by_type["comparison_yn"]["count"] == 2
    assert by_type["comparison_yn"]["vf"] == pytest.approx(0.75)
    assert by_type["distance"]["count"] == 1


def test_format_rubric_report_lists_components():
    report = score_items(_three_triples())
    text = format_rubric_report(
        report, breakdown_by(report.per_item, key_fn=lambda s: s.question_type)
    )
    assert "RUBRIC CREDIT" in text
    for label in ("VF", "RC", "IF", "Rubric score", "Scalar"):
        assert label in text


# --- integration: the PRE-EXISTING vqasynth.evaluation module ---------------

def test_scalar_score_dispatches_to_evaluation_scorers():
    """The contrast score routes through the repo's existing scorers, not a
    parallel implementation."""
    from vqasynth.evaluation import score_distance_mra, score_yes_no  # pre-existing

    assert scalar_score(YN_Q, YN_GOLD, "Yes.") == score_yes_no("Yes.", YN_GOLD)
    assert scalar_score(DIST_Q, DIST_GOLD, "1.2 meters") == score_distance_mra(
        "1.2 meters", DIST_GOLD
    )


def test_instruction_following_keys_off_evaluation_classify_question():
    """The IF expectation is derived from the same question-type tagger the
    multi-benchmark evaluation stage already uses."""
    from vqasynth.evaluation import classify_question  # pre-existing

    assert classify_question(YN_Q) == "comparison_yn"
    assert classify_question(DIST_Q) == "distance"
    # A yes/no question answered with a bare number fails IF.
    assert score_item(YN_Q, YN_GOLD, "3 meters").instruction_following == 0.0


def test_components_are_the_paper_three():
    assert COMPONENTS == ("VF", "RC", "IF")


# --- integration: the call site in vqasynth.benchmarks ----------------------

def _runner():
    from vqasynth.benchmarks import BenchmarkRunner  # pre-existing module

    return BenchmarkRunner(benchmarks=["spatialscore"])


def _items():
    """Normalized items in the shape ``load_spatialscore`` emits (see the
    loader's normalization block) — synthetic, so the wiring test needs no
    HF Hub access."""
    return [
        {
            "id": "judg-0",
            "question": "Is the cup to the left of the book from the viewer's perspective?",
            "answer": "Yes, the cup is to the left of the book.",
            "question_type": "judgment",
            "category": "Spatial Relations",
            "subcategory": "left_right",
        },
        {
            "id": "dist-1",
            "question": "What is the distance between the chair and the bookshelf?",
            "answer": "the chair is 1 meter from the bookshelf.",
            "question_type": "open-ended",
            "category": "Object Distance",
            "subcategory": "distance",
        },
        {
            "id": "judg-2",
            "question": "Does the lamp appear over the desk?",
            "answer": "Yes, the lamp is above the desk.",
            "question_type": "judgment",
            "category": "Spatial Relations",
            "subcategory": "vertical",
        },
    ]


def test_runner_score_attaches_rubric_credit_block():
    """BenchmarkRunner.score — the multi-benchmark evaluation entry point —
    now carries per-proposition partial credit next to scalar accuracy."""
    items = _items()
    preds = {
        "judg-0": "Yes, the cup is to the left of the book.",
        "dist-1": "The chair is 200 centimeters from the bookshelf.",
        "judg-2": "No, the lamp is below the desk.",
    }
    result = _runner().score("spatialscore", items, preds)

    rubric = result["rubric_credit"]
    assert set(rubric) == {
        "vf", "reasoning_consistency", "instruction_following",
        "rubric_score", "scalar_accuracy", "propositions_per_item",
    }
    assert 0.0 <= rubric["vf"] <= 1.0
    assert 0.0 <= rubric["instruction_following"] <= 1.0
    assert rubric["propositions_per_item"] >= 1.0
    # Partial credit where the scalar scorer scores zero: judg-0 is right,
    # dist-1 is off by a factor of two, judg-2 is wrong outright.
    assert 0.0 < rubric["rubric_score"] < 1.0


def test_runner_scored_item_localizes_the_failing_proposition():
    """The same public API the runner calls localizes WHICH proposition of
    WHICH answer failed — the error-localization half of the rubric."""
    items = _items()
    preds = {"judg-2": "No, the lamp is below the desk."}
    result = _runner().score("spatialscore", [items[2]], preds)
    assert result["per_item"][0]["score"] == 0.0  # scalar: all-or-nothing

    score = score_item(items[2]["question"], items[2]["answer"], preds["judg-2"])
    localized = format_localizations(score)
    assert "VF" in localized
    assert "antonym 'below'" in localized
    assert "the lamp is below the desk" in localized


def test_format_benchmark_report_renders_the_rubric_block():
    """The printed report surfaces VF / RC / IF, not only scalar accuracy."""
    from vqasynth.benchmarks import format_benchmark_report  # pre-existing

    items = _items()
    preds = {item["id"]: "Yes." for item in items}
    with patch.object(
        _runner().__class__, "load", return_value=items
    ) as _mocked_load:
        report = _runner().run({"spatialscore": preds})
    text = format_benchmark_report(report)
    assert "Rubric credit (per proposition)" in text
    assert "reasoning_consistency" in text
