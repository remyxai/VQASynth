"""Tests for the Visual Credit Audit integration.

Exercises BenchmarkRunner.audit_visual_credit -- the wiring that lives in
vqasynth.benchmarks (a non-new module) -- against a fake VLM, so no CUDA and no
model download are required. Also covers the D-CC primitives directly.
Real end-to-end auditing on a GPU host belongs elsewhere, like the existing
vggt_speedups smoke tests.
"""
from __future__ import annotations

from vqasynth.benchmarks import BenchmarkRunner
from vqasynth.visual_credit_audit import (
    audit_decisions,
    blank_control_image,
    credit_decision,
)


class _FakeVLM:
    """Mimics VLMInference.predict(image, question) -> str from a fixed map."""

    def __init__(self, controls):
        self.controls = controls
        self.calls = 0

    def predict(self, image, question):
        self.calls += 1
        return self.controls[question]


def _judgment_item(item_id, question, answer):
    return {
        "id": item_id,
        "question": question,
        "answer": answer,
        "question_type": "judgment",
        "category": "c",
        "subcategory": "s",
    }


def test_audit_visual_credit_wiring_via_benchmark_runner():
    """D-CC via BenchmarkRunner: credited vs. correct-but-uncredited vs. incorrect."""
    items = [
        _judgment_item("i1", "Q1", "yes"),  # image=yes control=no -> credited
        _judgment_item("i2", "Q2", "yes"),  # image=yes control=yes -> uncredited
        _judgment_item("i3", "Q3", "no"),   # image=yes control=no -> incorrect
    ]
    image_predictions = {"i1": "yes", "i2": "yes", "i3": "yes"}
    fake_vlm = _FakeVLM({"Q1": "no", "Q2": "yes", "Q3": "no"})

    runner = BenchmarkRunner(benchmarks=["spatialscore"])
    report = runner.audit_visual_credit(items, image_predictions, vlm=fake_vlm)

    assert report["audit"] == "visual_credit_dcc"
    assert report["n"] == 3
    assert report["accuracy"] == 2 / 3               # i1, i2 correct
    assert report["d_cc"] == 1 / 3                   # only i1 credited
    assert report["correct_uncredited_rate"] == 0.5  # 1 of 2 correct uncredited
    assert fake_vlm.calls == 3                       # one blank control per item


def test_audit_uses_precomputed_controls_and_skips_without_vlm():
    """Precomputed controls are reused; items lacking a control are skipped."""
    items = [
        _judgment_item("i1", "Q1", "yes"),
        _judgment_item("i2", "Q2", "yes"),  # no control, no vlm -> skipped
    ]
    image_predictions = {"i1": "yes", "i2": "yes"}

    runner = BenchmarkRunner(benchmarks=["spatialscore"])
    report = runner.audit_visual_credit(
        items, image_predictions, control_predictions={"i1": "no"}
    )

    assert report["n"] == 1
    assert report["d_cc"] == 1.0


def test_audit_skips_open_ended_items():
    """Only forced-choice items are audited; open-ended items are ignored."""
    items = [
        {
            "id": "d1", "question": "Q", "answer": "2 meters",
            "question_type": "open-ended", "category": "c", "subcategory": "s",
        },
        _judgment_item("j1", "Q1", "no"),
    ]
    runner = BenchmarkRunner(benchmarks=["spatialscore"])
    report = runner.audit_visual_credit(
        items, {"j1": "no"}, control_predictions={"j1": "yes"}
    )
    assert report["n"] == 1


def test_credit_decision_multi_choice():
    assert credit_decision("A", "B", "A", "multi-choice") == 1.0  # credited
    assert credit_decision("A", "A", "A", "multi-choice") == 0.0  # uncredited
    assert credit_decision("B", "A", "A", "multi-choice") == 0.0  # incorrect


def test_credit_decision_unextractable_is_none():
    assert credit_decision("maybe", "yes", "yes", "judgment") is None


def test_blank_control_image_is_solid_rgb():
    img = blank_control_image()
    assert img.size == (224, 224)
    assert img.mode == "RGB"
    assert img.getpixel((0, 0)) == img.getpixel((100, 100))


def test_audit_decisions_empty():
    assert audit_decisions([])["n"] == 0
