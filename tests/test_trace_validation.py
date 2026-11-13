"""CPU tests for the Phase-4 data-validation gate (vqasynth.trace_validation).

Verifies the pre-training checks on generated traces: grounding-correctness
(hallucinated objects / out-of-bounds boxes), trace-faithfulness (answer
consistency, grounded-state reference, low-confidence hedging), chance/difficulty
calibration, dedup, and the end-to-end filter_dataset report. Pure CPU; no model.
"""
from __future__ import annotations

import json

import pytest

from vqasynth import trace_validation as tv

GS = {
    "objects": [
        {"label": "chair", "bbox": [10, 10, 50, 50], "depth_conf": 0.9},
        {"label": "mug", "bbox": [100, 100, 140, 140], "depth_conf": 0.3},  # low-confidence
    ],
    "pairs": [{"a": "chair", "b": "mug", "distance_m": 1.4}],
}
SIZE = (640, 480)


def _trace(objects, think, answer="(A)"):
    scene = json.dumps({"objects": objects, "relations": []})
    return f"<observe>x</observe><scene>{scene}</scene><think>{think}</think><answer>{answer}</answer>"


GOOD_OBJS = [
    {"id": 0, "label": "chair", "bbox": [10, 10, 50, 50]},
    {"id": 1, "label": "mug", "bbox": [100, 100, 140, 140]},
]
GOOD = _trace(GOOD_OBJS, "The chair is roughly 2 m away; I'm not sure about the mug.")


def test_grounding_correctness_good():
    r = tv.grounding_correctness(GOOD, GS, image_size=SIZE)
    assert r["score"] == pytest.approx(1.0) and r["verifiable"] and not r["issues"]


def test_grounding_flags_hallucinated_object():
    bad = _trace(
        [{"id": 0, "label": "banana", "bbox": [10, 10, 50, 50]},
         {"id": 1, "label": "mug", "bbox": [100, 100, 140, 140]}],
        "reasoning",
    )
    r = tv.grounding_correctness(bad, GS, image_size=SIZE)
    assert r["score"] < 1.0 and "hallucinated_objects" in r["issues"]


def test_grounding_flags_out_of_bounds_box():
    bad = _trace([{"id": 0, "label": "chair", "bbox": [600, 400, 700, 500]}], "r")
    r = tv.grounding_correctness(bad, GS, image_size=SIZE)
    assert "boxes_out_of_bounds" in r["issues"]


def test_grounding_no_scene():
    assert tv.grounding_correctness("<answer>(A)</answer>", GS)["score"] == 0.0


def test_grounding_unverifiable_without_reference():
    r = tv.grounding_correctness(GOOD, {"objects": []})  # no grounded labels, no image size
    assert r["verifiable"] is False and r["score"] == 1.0


def test_faithfulness_good():
    r = tv.trace_faithfulness(GOOD, "(A)", GS)
    assert r["score"] == pytest.approx(1.0) and not r["issues"]


def test_faithfulness_wrong_answer():
    r = tv.trace_faithfulness(GOOD, "(B)", GS)
    assert "answer_mismatch" in r["issues"] and r["score"] < 1.0


def test_faithfulness_unhedged_low_confidence():
    unhedged = _trace(GOOD_OBJS, "The chair is 2.1 meters away and the mug is 0.7 meters away.")
    r = tv.trace_faithfulness(unhedged, "(A)", GS)
    assert "unhedged_low_confidence" in r["issues"]


def test_validate_trace_pass_fail():
    assert tv.validate_trace(GOOD, "(A)", GS, image_size=SIZE)["passed"] is True
    assert tv.validate_trace(GOOD, "(B)", GS, image_size=SIZE)["passed"] is False


def test_chance_calibration():
    assert tv.chance_calibration(["(A)", "(A)", "(A)", "(A)"])["degenerate"] is True
    balanced = tv.chance_calibration(["(A)", "(B)", "(C)", "(A)"])
    assert balanced["degenerate"] is False
    assert balanced["majority_frac"] == pytest.approx(0.5)


def test_dedup():
    rows = [
        {"q": "How far is the chair?", "a": "(A)"},
        {"q": "how far is the chair?", "a": "(A)"},  # dup (normalized)
        {"q": "Where is the mug?", "a": "(B)"},
    ]
    kept, dropped = tv.dedup(rows, lambda r: r["q"], lambda r: r["a"])
    assert len(kept) == 2 and dropped == 1


def test_filter_dataset_end_to_end():
    rows = [
        {"out": GOOD, "ans": "(A)", "gs": GS, "q": "q1", "size": SIZE},  # passes
        {"out": _trace([{"id": 0, "label": "banana", "bbox": [10, 10, 50, 50]}], "r"),
         "ans": "(A)", "gs": GS, "q": "q2", "size": SIZE},  # fails grounding
        {"out": GOOD, "ans": "(A)", "gs": GS, "q": "q1", "size": SIZE},  # dup of row 1
    ]
    rep = tv.filter_dataset(
        rows,
        output_of=lambda r: r["out"], answer_of=lambda r: r["ans"],
        grounded_of=lambda r: r["gs"], question_of=lambda r: r["q"],
        image_size_of=lambda r: r["size"],
    )
    assert rep["n_in"] == 3
    assert rep["n_dropped_quality"] == 1 and "hallucinated_objects" in rep["drop_reasons"]
    assert rep["n_dropped_dup"] == 1
    assert rep["n_kept"] == 1
