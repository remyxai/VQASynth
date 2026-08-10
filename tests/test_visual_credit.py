"""Structural tests for vqasynth.visual_credit.

Verifies the Visual Credit Audit (D-CC, correct-but-uncredited, label-free
decision-change) against hand-computed small examples — no CUDA, no model
download. Mirrors the dependency-free style of tests/test_judge_dataset.py.

Integration with the PRE-EXISTING ``vqasynth.evaluation`` module (the scorer
layer the multi-benchmark evaluation stage already ships) is asserted directly:
the audit's gold-alignment and forced-choice decision extraction must agree with
``extract_yes_no`` / ``extract_option``, and the per-question-type breakdown
must compose with ``classify_question``. Those modules are not created by this
PR, so the composition tests exercise real integrated behavior rather than
self-testing the new file.
"""
from __future__ import annotations

import pytest

from vqasynth.visual_credit import (
    CONTROL_CONDITIONS,
    VisualCreditItem,
    audit,
    breakdown_by,
    build_blank_control,
    build_text_only_control,
    decisions_differ,
    extract_decision,
    format_credit_report,
    gold_aligned,
    make_blank_image,
)


# --- forced-choice decision extraction --------------------------------------

def test_extract_decision_yes_no_option_text():
    assert extract_decision("Yes, it is.") == ("yesno", True)
    assert extract_decision("Nope.") == ("yesno", False)
    assert extract_decision("The answer is B.") == ("option", "B")
    # VQASynth "Actually, ..." contradiction -> treated as "no".
    assert extract_decision("Actually, the cup is on the right.") == ("yesno", False)
    assert extract_decision("The cup is to the left.") == ("text", "the cup is to the left.")


def test_decisions_differ_yesno():
    assert decisions_differ("Yes.", "No.") is True
    assert decisions_differ("Yes.", "Yes, exactly.") is False


def test_decisions_differ_option():
    assert decisions_differ("Answer: A", "Answer: B") is True
    # Both route through extract_option to the same letter "A".
    assert decisions_differ("(A)", "answer is a") is False


def test_decisions_differ_text_fallback():
    # Neither yes/no nor an option letter -> lowered text compared exactly.
    assert decisions_differ("The table.", "The chair.") is True
    assert decisions_differ("The Table", "the table") is False


# --- gold alignment ---------------------------------------------------------

def test_gold_aligned_cascades_yesno_then_option_then_text():
    assert gold_aligned("q", "Yes", "Yes, it is.") is True
    assert gold_aligned("q", "Yes", "No.") is False
    assert gold_aligned("q", "A", "The answer is A.") is True
    assert gold_aligned("q", "A", "B.") is False
    assert gold_aligned("q", "the table", "The Table") is True


# --- VisualCreditItem normalization ----------------------------------------

def test_item_normalizes_single_string_control():
    item = VisualCreditItem("q", "Yes", "Yes.", "Yes.")
    assert item.pred_controls == ["Yes."]


def test_item_rejects_empty_controls():
    with pytest.raises(ValueError, match="at least one control"):
        VisualCreditItem("q", "Yes", "Yes.", [])


# --- the audit: D-CC + correct-but-uncredited decomposition ----------------

def _three_items():
    return [
        # correct on real, ALSO correct under control -> correct but uncredited
        VisualCreditItem("Is A left of B?", "Yes", "Yes.", ["Yes."]),
        # correct on real, WRONG under control -> dependence-credited (D-CC)
        VisualCreditItem("Is A left of B?", "Yes", "Yes.", ["No."]),
        # wrong on real -> not correct
        VisualCreditItem("Is A left of B?", "Yes", "No.", ["No."]),
    ]


def test_audit_decomposition_matches_hand_computation():
    report = audit(_three_items())
    assert report.total == 3
    assert report.accuracy == pytest.approx(2 / 3)        # items 0, 1
    assert report.control_accuracy == pytest.approx(1 / 3)  # item 0
    assert report.d_cc == pytest.approx(1 / 3)            # item 1 only
    assert report.correct_but_uncredited == pytest.approx(1 / 3)  # item 0
    assert report.uncredited_of_correct == pytest.approx(1 / 2)  # item 0 of 2 correct
    assert report.image_gain == pytest.approx(1 / 3)      # 2/3 - 1/3


def test_audit_dcc_plus_uncredited_equals_accuracy():
    # For any dataset, credited and uncredited partition the correct items.
    report = audit(_three_items())
    assert report.d_cc + report.correct_but_uncredited == pytest.approx(report.accuracy)


def test_audit_decision_change_rate_is_label_free():
    # Only item 1 has a real-image decision ("Yes") that differs from its
    # control ("No"); items 0 and 2 agree with their controls.
    report = audit(_three_items())
    assert report.decision_change_rate == pytest.approx(1 / 3)


def test_audit_per_item_flags_credited_and_uncredited():
    report = audit(_three_items())
    assert report.per_item[0].correct_but_uncredited is True
    assert report.per_item[0].credited is False
    assert report.per_item[1].credited is True
    assert report.per_item[1].correct_but_uncredited is False
    assert report.per_item[2].real_correct is False


def test_audit_multiple_controls_any_match_makes_uncredited():
    # Gold is "Yes"; real is correct; text-only is wrong but blank is right.
    # Under ANY-control-matches-gold, the image is not uniquely responsible.
    item = VisualCreditItem("q", "Yes", "Yes.", ["No.", "Yes."])
    report = audit([item])
    assert report.per_item[0].control_correct is True
    assert report.per_item[0].credited is False
    assert report.per_item[0].correct_but_uncredited is True


def test_audit_custom_correctness_function_is_used():
    # A correctness fn that only matches an exact token proves the hook is
    # wired through (the default extractor would disagree here).
    def strict(question, gold, prediction):
        return prediction.strip() == "YES_TOKEN"

    item = VisualCreditItem("q", "Yes", "YES_TOKEN", ["YES_TOKEN"])
    report = audit([item], correctness=strict)
    assert report.per_item[0].real_correct is True
    assert report.per_item[0].correct_but_uncredited is True


def test_audit_empty_dataset_returns_zeros():
    report = audit([])
    assert report.total == 0
    assert report.accuracy == 0.0
    assert report.d_cc == 0.0


# --- integration with the PRE-EXISTING vqasynth.evaluation module ----------

def test_gold_aligned_agrees_with_evaluation_extract_yes_no():
    """The audit's correctness path is driven by the existing
    ``vqasynth.evaluation`` extractors, not a parallel implementation."""
    from vqasynth.evaluation import extract_yes_no  # pre-existing module

    for gold, pred in [("Yes", "Yes."), ("Yes", "No."), ("No", "Nope."), ("No", "Yes.")]:
        expected = extract_yes_no(gold) == extract_yes_no(pred)
        assert gold_aligned("q", gold, pred) is expected


def test_breakdown_by_composes_with_evaluation_classify_question():
    """The per-category credit breakdown keys off the existing
    ``vqasynth.evaluation.classify_question`` tagger (the same categories the
    multi-benchmark evaluation stage already uses)."""
    from vqasynth.evaluation import classify_question  # pre-existing module

    items = [
        VisualCreditItem("Is the cup left of the book?", "Yes", "Yes.", ["No."]),  # yn -> D-CC
        VisualCreditItem("How far is the car from the sign?", "3 meters", "3m", ["1m"]),  # distance
        VisualCreditItem("Is the lamp above the desk?", "Yes", "Yes.", ["Yes."]),  # yn -> uncredited
    ]
    report = audit(items)
    by_type = breakdown_by(items, report, key_fn=lambda it: classify_question(it.question))

    assert set(by_type) == {"comparison_yn", "distance"}
    yn = by_type["comparison_yn"]
    assert yn["count"] == 2
    assert yn["d_cc"] == pytest.approx(0.5)            # one of the two yn items is credited
    assert yn["correct_but_uncredited"] == pytest.approx(0.5)
    assert by_type["distance"]["count"] == 1


def test_decisions_differ_uses_evaluation_extract_option():
    """Decision comparison routes multi-choice answers through the existing
    option extractor, so "(A)" and "answer is a" are the same decision."""
    from vqasynth.evaluation import extract_option  # pre-existing module

    assert extract_option("(A)") == extract_option("answer is a")
    assert decisions_differ("(A)", "answer is a") is False


# --- controls + formatting --------------------------------------------------

def test_control_builders_keep_original_question():
    q = "Is A left of B?"
    assert build_text_only_control(q) == q
    assert build_blank_control(q) == q
    assert CONTROL_CONDITIONS == ("text_only", "blank")


def test_make_blank_image_is_guarded():
    pytest.importorskip("PIL")
    img = make_blank_image(size=(8, 8))
    assert img.size == (8, 8)


def test_format_credit_report_contains_headline_metrics():
    report = audit(_three_items())
    text = format_credit_report(report)
    assert "VISUAL CREDIT AUDIT" in text
    assert "D-CC" in text
    assert "Correct but uncredited" in text
    assert "Decision change rate" in text
