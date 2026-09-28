"""Structural tests for vqasynth.precision_parity.

Verifies the paired cross-precision audit (outcome flips, decision changes,
same-score-tradeoff flag, sign test + BH-FDR) against hand-computed small
examples — no CUDA, no model download, no benchmark download. Mirrors the
dependency-free style of tests/test_visual_credit.py.

Integration with the PRE-EXISTING evaluation stack is asserted directly:

* ``BenchmarkRunner.score_parity`` (vqasynth.benchmarks, the multi-benchmark
  evaluation stage this PR hooks into) must drive the audit through the
  benchmark's native scorer, offline, over items passed explicitly.
* ``vqasynth.inference.quantization_kwargs`` must map the GHOST-Q precision
  grid (FP16 / INT8 / NF4) onto transformers ``from_pretrained`` kwargs.
* The standalone audit's correctness path composes with
  ``vqasynth.evaluation.classify_question`` for per-question-type breakdowns.

Those modules are not created by this PR, so the composition tests exercise
real integrated behavior rather than self-testing the new file.
"""
from __future__ import annotations

import inspect

import pytest

from vqasynth.precision_parity import (
    BASELINE_ONLY,
    BOTH_CORRECT,
    BOTH_WRONG,
    QUANTIZED_ONLY,
    UNPAIRED,
    PrecisionParityReport,
    audit,
    benjamini_hochberg,
    breakdown_by,
    format_parity_report,
    sign_test_p,
)


def _items(questions, golds):
    return [
        {"id": f"q{i}", "question": q, "answer": g,
         "question_type": "judgment", "category": "Spatial Relations",
         "subcategory": "left_right"}
        for i, (q, g) in enumerate(zip(questions, golds))
    ]


# --- paired statistics ------------------------------------------------------

def test_sign_test_p_symmetric_and_exact():
    # No discordant pairs -> nothing to test.
    assert sign_test_p(0, 0) == 1.0
    # Balanced flips -> two-sided p caps at 1.0.
    assert sign_test_p(3, 3) == 1.0
    # 8 losses vs 0 rescues: p = 2 * C(8,0) / 2^8.
    assert sign_test_p(8, 0) == pytest.approx(2 / 256)
    assert sign_test_p(0, 8) == pytest.approx(2 / 256)


def test_benjamini_hochberg_matches_step_up_computation():
    assert benjamini_hochberg([]) == []
    # Hand-computed step-up: sorted .01/.03/.04 -> q .03/.04/.04.
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    # Never shrinks a p-value and never exceeds 1.
    raw = [0.9, 0.001, 0.5, 0.2]
    q_vals = benjamini_hochberg(raw)
    assert all(q >= p for q, p in zip(q_vals, raw))
    assert all(0.0 <= q <= 1.0 for q in q_vals)


# --- the audit: paired decomposition ---------------------------------------

def _five_item_fixture():
    questions = [
        "Is the cup to the left of the book?",
        "Is the lamp above the desk?",
        "Is the car in front of the sign?",
        "Is the box under the table?",
        "Is the rug next to the bed?",
    ]
    golds = ["Yes", "Yes", "No", "Yes", "Yes"]
    baseline = {
        "q0": "Yes.",          # both correct, same decision
        "q1": "Yes.",          # correct -> wrong (lost), decision changed
        "q2": "Yes.",          # wrong -> correct (rescued), decision changed
        "q3": "No.",           # both wrong, same decision
        "q4": "Yes",           # both correct ("Yes, exactly" -> same yes/no)
    }
    quantized = {
        "q0": "Yes, it is.",
        "q1": "No.",
        "q2": "No.",
        "q3": "Nope.",
        "q4": "Yes, exactly.",
    }
    return _items(questions, golds), baseline, quantized


def test_audit_decomposition_matches_hand_computation():
    items, baseline, quantized = _five_item_fixture()
    report = audit(baseline, quantized, items)

    assert isinstance(report, PrecisionParityReport)
    assert report.total == 5
    assert report.paired == 5
    assert report.baseline_accuracy == pytest.approx(3 / 5)   # q0, q1, q4
    assert report.quantized_accuracy == pytest.approx(3 / 5)  # q0, q2, q4
    assert report.net_delta == pytest.approx(0.0)
    assert report.agreement == pytest.approx(3 / 5)           # q0, q3, q4
    assert report.baseline_only_correct == 1                  # q1
    assert report.quantized_only_correct == 1                 # q2
    assert report.flip_rate == pytest.approx(2 / 5)
    # q0/q4 same decision, q3 same ("No." vs "Nope."), q1/q2 changed.
    assert report.decision_agreement == pytest.approx(3 / 5)
    assert [r.outcome for r in report.per_item] == [
        BOTH_CORRECT, BASELINE_ONLY, QUANTIZED_ONLY, BOTH_WRONG, BOTH_CORRECT,
    ]


def test_audit_flags_same_score_tradeoff():
    # Net delta is 0 (well inside ±2pp) yet two paired outcomes flipped:
    # exactly the regime GHOST-Q shows aggregate accuracy conceals.
    items, baseline, quantized = _five_item_fixture()
    report = audit(baseline, quantized, items)
    assert report.same_score_tradeoff is True
    # Balanced flips are not significant under the sign test.
    assert report.significant_asymmetry is False
    assert report.symmetry_p == pytest.approx(1.0)


def test_audit_no_tradeoff_when_aggregate_actually_moves():
    # Baseline correct on everything, quantized wrong on everything.
    # 6 losses vs 0 rescues -> sign-test p = 2/2^6 < 0.05.
    items = _items(["Is a left of b?"] * 6, ["Yes"] * 6)
    report = audit(
        {f"q{i}": "Yes." for i in range(6)},
        {f"q{i}": "No." for i in range(6)},
        items,
    )
    assert report.net_delta == pytest.approx(-1.0)
    assert report.same_score_tradeoff is False
    assert report.agreement == 0.0
    assert report.significant_asymmetry is True  # 6 losses, 0 rescues


def test_audit_skips_items_missing_a_prediction():
    items, baseline, quantized = _five_item_fixture()
    del quantized["q1"], quantized["q3"]
    report = audit(baseline, quantized, items)

    assert report.total == 5
    assert report.paired == 3
    assert report.per_item[1].outcome == UNPAIRED
    assert report.per_item[1].baseline_correct is None
    assert report.baseline_accuracy == pytest.approx(2 / 3)  # q0, q4


def test_audit_custom_correctness_none_drops_item_from_pair():
    from vqasynth.visual_credit import gold_aligned

    items = _items(["Is a left of b?", "Is c left of d?"], ["Yes", "Yes"])

    def correctness(item, prediction):
        if "unscorable" in prediction:
            return None
        return gold_aligned(item["question"], item["answer"], prediction)

    report = audit(
        {"q0": "Yes.", "q1": "unscorable"},
        {"q0": "No.", "q1": "unscorable"},
        items,
        correctness=correctness,
    )
    assert report.paired == 1
    assert report.per_item[1].outcome == UNPAIRED


def test_audit_accepts_list_predictions_aligned_with_items():
    items, baseline, _ = _five_item_fixture()
    report_dict = audit(baseline, list(baseline.values()), items)
    assert report_dict.paired == 5
    assert report_dict.baseline_accuracy == pytest.approx(report_dict.quantized_accuracy)
    assert report_dict.flip_rate == 0.0
    assert report_dict.same_score_tradeoff is False


def test_audit_empty_dataset_returns_zeros():
    report = audit({}, {}, [])
    assert report.total == 0
    assert report.paired == 0
    assert report.flip_rate == 0.0
    assert report.same_score_tradeoff is False


# --- breakdown + formatting -------------------------------------------------

def test_breakdown_flags_significant_group_after_fdr():
    from vqasynth.evaluation import classify_question  # pre-existing module

    items, baseline, quantized = [], {}, {}
    # comparison_yn group: quantization breaks 8, rescues 0 -> raw p ~ .008.
    for i in range(8):
        items.append(
            {"id": f"yn{i}", "question": f"Is the object{i} left of the wall?",
             "answer": "Yes"}
        )
        baseline[f"yn{i}"] = "Yes."
        quantized[f"yn{i}"] = "No."
    # distance group: identical answers -> no flips, p = 1.
    for i in range(4):
        items.append(
            {"id": f"d{i}", "question": f"How far is object{i} from the sign?",
             "answer": "2 meters"}
        )
        baseline[f"d{i}"] = "2 meters"
        quantized[f"d{i}"] = "2 meters"

    report = audit(baseline, quantized, items)
    by_type = breakdown_by(items, report, key_fn=lambda it: classify_question(it["question"]))

    assert set(by_type) == {"comparison_yn", "distance"}
    yn = by_type["comparison_yn"]
    assert yn["count"] == 8
    assert yn["baseline_only_correct"] == 8
    assert yn["quantized_accuracy"] == 0.0
    assert yn["p"] == pytest.approx(2 / 256)
    assert yn["q"] >= yn["p"]          # BH never shrinks a p-value
    assert yn["significant"] is True   # survives FDR correction
    assert by_type["distance"]["significant"] is False


def test_breakdown_skips_unpaired_items():
    items, baseline, quantized = _five_item_fixture()
    del quantized["q2"]
    report = audit(baseline, quantized, items)
    by_cat = breakdown_by(items, report, key_fn=lambda it: it["category"])
    assert by_cat["Spatial Relations"]["count"] == 4  # q2 dropped


def test_format_parity_report_contains_headline_metrics():
    items, baseline, quantized = _five_item_fixture()
    report = audit(baseline, quantized, items)
    text = format_parity_report(report)
    assert "PRECISION PARITY AUDIT" in text
    assert "SAME-SCORE TRADEOFF" in text
    assert "Flip rate" in text
    assert "5 paired of 5 items" in text

    # With a breakdown, the FDR column shows up.
    by_cat = breakdown_by(items, report, key_fn=lambda it: it["category"])
    assert "q(FDR)" in format_parity_report(report, breakdown=by_cat)


# --- integration: BenchmarkRunner.score_parity (pre-existing module) --------

# SpatialScore-shaped items exercising all three native scorer branches.
_SPATIALSCORE_ITEMS = [
    {"id": "q0", "question": "Is the cup to the left of the book?",
     "answer": "Yes", "question_type": "judgment",
     "category": "Spatial Relations", "subcategory": "left_right"},
    {"id": "q1", "question": "How far is the chair from the desk?",
     "answer": "3 meters", "question_type": "open-ended",
     "category": "Object Distance", "subcategory": "distance"},
    {"id": "q2", "question": "Which object is closer to the camera?",
     "answer": "B", "question_type": "multi-choice",
     "category": "Spatial Relations", "subcategory": "comparison"},
    {"id": "q3", "question": "Is the box under the table?",
     "answer": "No", "question_type": "judgment",
     "category": "Spatial Relations", "subcategory": "vertical"},
]


def test_score_parity_uses_benchmark_native_scorer():
    """The runner's parity hook must score through the SpatialScore scorer,
    not a text-exact comparison: "2 meters" is ratio-tolerance-correct against
    gold "3 meters" under score_distance, but wrong under exact match."""
    from vqasynth.benchmarks import BenchmarkRunner  # pre-existing module

    runner = BenchmarkRunner(benchmarks=["spatialscore"])
    baseline = {
        "q0": "Yes, the cup is to the left.",
        "q1": "2 meters",
        "q2": "(A)",
        "q3": "Yes.",
    }
    quantized = {
        "q0": "No, it is to the right.",
        "q1": "2.5 meters",
        "q2": "Answer: B",
        "q3": "Yes.",
    }

    report = runner.score_parity(
        "spatialscore", baseline, quantized, items=_SPATIALSCORE_ITEMS
    )

    assert isinstance(report, PrecisionParityReport)
    assert report.paired == 4
    assert [r.outcome for r in report.per_item] == [
        BASELINE_ONLY, BOTH_CORRECT, QUANTIZED_ONLY, BOTH_WRONG,
    ]
    assert report.baseline_accuracy == pytest.approx(0.5)
    assert report.quantized_accuracy == pytest.approx(0.5)
    assert report.same_score_tradeoff is True


def test_score_parity_accepts_list_predictions_and_breakdown():
    from vqasynth.benchmarks import BenchmarkRunner  # pre-existing module

    runner = BenchmarkRunner(benchmarks=["spatialscore"])
    baseline = ["Yes.", "2 meters", "(B)", "Yes."]
    quantized = ["Yes.", "2 meters", "(B)", "Yes."]

    report = runner.score_parity(
        "spatialscore", baseline, quantized, items=_SPATIALSCORE_ITEMS
    )
    assert report.paired == 4
    assert report.flip_rate == 0.0

    by_cat = breakdown_by(
        _SPATIALSCORE_ITEMS, report, key_fn=lambda it: it["category"]
    )
    assert set(by_cat) == {"Spatial Relations", "Object Distance"}
    assert by_cat["Object Distance"]["count"] == 1


def test_score_parity_rejects_unknown_benchmark():
    from vqasynth.benchmarks import BenchmarkRunner  # pre-existing module

    runner = BenchmarkRunner(benchmarks=["spatialscore"])
    with pytest.raises(ValueError, match="No scorer"):
        runner.score_parity("nope", {}, {}, items=[])


# --- integration: vqasynth.inference precision protocol (pre-existing) -----

def test_quantization_kwargs_maps_the_ghostq_precision_grid():
    import torch  # pre-existing dependency of vqasynth.inference

    from vqasynth.inference import quantization_kwargs  # pre-existing module

    fp16 = quantization_kwargs("fp16")
    assert fp16 == {"torch_dtype": torch.float16}
    assert quantization_kwargs("bf16") == {"torch_dtype": torch.bfloat16}
    # "auto" defers to the caller-provided dtype.
    assert quantization_kwargs("auto", dtype=torch.float32) == {
        "torch_dtype": torch.float32
    }

    int8 = quantization_kwargs("int8", dtype=torch.float16)
    assert int8["quantization_config"].load_in_8bit is True

    nf4 = quantization_kwargs("nf4", dtype=torch.float16)
    config = nf4["quantization_config"]
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.float16


def test_quantization_kwargs_rejects_unknown_precision():
    from vqasynth.inference import quantization_kwargs

    with pytest.raises(ValueError, match="Unknown precision"):
        quantization_kwargs("q8")


def test_supported_precisions_cover_float_and_quantized():
    from vqasynth.inference import FP_PRECISIONS, QUANTIZED_PRECISIONS

    assert set(FP_PRECISIONS) & set(QUANTIZED_PRECISIONS) == set()
    assert set(QUANTIZED_PRECISIONS) == {"int8", "nf4"}


def test_run_inference_on_benchmark_exposes_precision():
    """The two-precision protocol slots into the benchmark inference entry
    point: one run per precision, paired downstream by score_parity."""
    from vqasynth.inference import run_inference_on_benchmark  # pre-existing module

    signature = inspect.signature(run_inference_on_benchmark)
    assert "precision" in signature.parameters
    assert signature.parameters["precision"].default == "auto"
