"""Structural tests for the SpaceCast-Bench integration.

Mirrors the philosophy of ``tests/test_correspondence.py``: verify the
loader + scorer mechanics and their registration in ``vqasynth.benchmarks``
without network access. Rows mirror the released ``hongxingli/SpaceCast-Bench``
schema (question_uid / level / type / options / option_labels / answer /
multi_select / correct_values / image_paths); ``load_dataset`` and
``hf_hub_download`` are monkeypatched so nothing downloads.

Scoring is asserted against the existing ``vqasynth.evaluation`` primitives
(the shared multi-choice contract), and the registry wiring is exercised
end-to-end through ``BenchmarkRunner`` whenever ``vqasynth.benchmarks`` is
importable (its module-level ``datasets`` import needs pyarrow, so that path
is guarded like the cv2-guarded tests elsewhere in this suite).
"""
from __future__ import annotations

import pytest

from vqasynth import spacecast
from vqasynth.evaluation import extract_option
from vqasynth.spacecast import (
    load_spacecast,
    score_spacecast_item,
    spacecast_macro_type_accuracy,
)


def make_row(**overrides):
    """One row in the released SpaceCast-Bench schema (an L3 rotation task)."""
    row = {
        "question_uid": "q1",
        "dataset": "scannet",
        "scene_id": "scene0025_01",
        "level": "L3",
        "type": "Rotate Camera Direction",
        "question_type": "Rotate Camera Direction",
        "task": "Rotate Camera Direction",
        "question": "In which direction is the book relative to the pillow?",
        "options": ["back-left", "front-left", "back-right", "front-right"],
        "option_labels": ["A", "B", "C", "D"],
        "answer": ["C"],
        "multi_select": False,
        "correct_values": ["back-right"],
        "image_paths": [
            "scannet/scene0025_01/492.jpg",
            "scannet/scene0025_01/522.jpg",
            "scannet/scene0025_01/564.jpg",
        ],
        "image_roles": ["main_a", "bridge", "main_b"],
    }
    row.update(overrides)
    return row


def make_item(**overrides):
    """A normalized item as produced by load_spacecast()."""
    item = {
        "id": "q1",
        "question": "In which direction is the book relative to the pillow?",
        "answer": "C",
        "question_type": "multi-choice",
        "category": "Rotate Camera Direction",
        "subcategory": "Global Prediction",
        "options": ["back-left", "front-left", "back-right", "front-right"],
        "images": [],
        "source": "SpaceCast-Bench",
        "multi_select": False,
        "correct_values": ["back-right"],
        "level": "L3",
    }
    item.update(overrides)
    return item


def _patch_downloads(monkeypatch, rows):
    """Patch dataset loading + hub downloads; returns the resolved image path."""
    monkeypatch.setattr(
        spacecast, "load_dataset",
        lambda repo_id, name=None, split="test", streaming=False: rows,
        raising=False,
    )
    monkeypatch.setattr(
        spacecast, "hf_hub_download",
        lambda repo_id, filename, repo_type="dataset": f"/hf-cache/{repo_id}/{filename}",
    )


def _benchmarks():
    """Import the wired call-site module, skipping when its deps are missing."""
    try:
        from vqasynth import benchmarks
    except ImportError as e:
        pytest.skip(f"vqasynth.benchmarks not importable here: {e}")
    return benchmarks


# ---------------------------------------------------------------------------
# Scorer (against the shared vqasynth.evaluation contract)
# ---------------------------------------------------------------------------

def test_single_select_scores_through_shared_extract_option():
    item = make_item()
    for pred in ["C", "Answer: C", "The answer is (C).", "(C)", "C."]:
        assert extract_option(pred) == "C"
        assert score_spacecast_item(item, pred) == 1.0
    assert score_spacecast_item(item, "A") == 0.0
    assert score_spacecast_item(item, "") == 0.0


def test_single_select_text_fallback():
    item = make_item()
    assert score_spacecast_item(item, "back-right") == 1.0
    assert score_spacecast_item(item, "front-left") == 0.0


def test_multi_select_requires_exact_set_match():
    item = make_item(
        multi_select=True,
        answer="A,C",
        correct_values=["back-left", "back-right"],
    )
    assert score_spacecast_item(item, "A and C") == 1.0
    assert score_spacecast_item(item, "C, A") == 1.0  # order-insensitive
    assert score_spacecast_item(item, "A") == 0.0  # no partial credit
    assert score_spacecast_item(item, "A, B") == 0.0
    # CoT that mentions option letters is ignored once an answer marker appears
    cot = "Views A and B identify the objects; the answer is A, C"
    assert score_spacecast_item(item, cot) == 1.0


def test_wide_option_sets_score_beyond_shared_extractor_range():
    options = [f"option_{i}" for i in range(8)]
    item = make_item(options=options, answer="G")
    assert extract_option("G") is None  # shared cascade only covers A-F
    assert score_spacecast_item(item, "G") == 1.0
    assert score_spacecast_item(item, "B") == 0.0


# ---------------------------------------------------------------------------
# Loader normalization
# ---------------------------------------------------------------------------

def test_loader_normalizes_rows_without_downloads(monkeypatch):
    _patch_downloads(monkeypatch, [make_row()])
    items = load_spacecast()
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "q1"
    assert item["source"] == "SpaceCast-Bench"
    assert item["question_type"] == "multi-choice"
    assert item["category"] == "Rotate Camera Direction"  # task type
    assert item["subcategory"] == "Global Prediction"  # L3 capability level
    assert item["answer"] == "C"
    assert item["multi_select"] is False
    assert item["correct_values"] == ["back-right"]
    # Images resolve from the hub cache in protocol order (main_a, bridge, main_b)
    assert item["images"] == [
        f"/hf-cache/hongxingli/SpaceCast-Bench/scannet/scene0025_01/{name}"
        for name in ["492.jpg", "522.jpg", "564.jpg"]
    ]


def test_loader_prefers_embedded_images(monkeypatch):
    embedded = [
        {"path": "scannet/scene0025_01/492.jpg", "bytes": b"fake-main-a"},
        {"path": "scannet/scene0025_01/564.jpg", "bytes": b"fake-main-b"},
    ]
    _patch_downloads(monkeypatch, [make_row(images=embedded)])
    monkeypatch.setattr(
        spacecast, "hf_hub_download",
        lambda *a, **k: pytest.fail("embedded images must not hit the hub"),
    )
    assert load_spacecast()[0]["images"] == embedded


def test_loader_maps_answer_values_when_schema_drifts(monkeypatch):
    _patch_downloads(
        monkeypatch,
        [make_row(answer=["back-right"])],  # values instead of labels
    )
    assert load_spacecast()[0]["answer"] == "C"


# ---------------------------------------------------------------------------
# Registry wiring in vqasynth.benchmarks (the call site)
# ---------------------------------------------------------------------------

def test_spacecast_registered_in_benchmark_registries():
    benchmarks = _benchmarks()
    assert benchmarks.BENCHMARK_LOADERS["spacecast"] is load_spacecast
    assert benchmarks.BENCHMARK_SCORERS["spacecast"] is score_spacecast_item
    # "all" picks it up (docker/eval_stage defaults this to every registered
    # benchmark's keys)
    assert "spacecast" in benchmarks.BenchmarkRunner(benchmarks="all").benchmarks


def test_runner_scores_spacecast_end_to_end(monkeypatch):
    benchmarks = _benchmarks()
    rows = [
        make_row(),
        make_row(
            question_uid="q2", level="L1", type="Camera Direction",
            answer=["A"], correct_values=["back-left"],
        ),
        make_row(
            question_uid="q3", level="L1", type="Camera Direction",
            answer=["B"], correct_values=["front-left"],
        ),
    ]
    _patch_downloads(monkeypatch, rows)

    runner = benchmarks.BenchmarkRunner(benchmarks=["spacecast"])
    report = runner.run({"spacecast": {"q1": "Answer: C", "q2": "A", "q3": "no idea"}})

    assert report["summary"]["spacecast"] == pytest.approx(2 / 3)
    by_category = report["benchmarks"]["spacecast"]["by_category"]
    assert set(by_category) == {"Rotate Camera Direction", "Camera Direction"}
    assert by_category["Camera Direction"]["accuracy"] == pytest.approx(0.5)
    by_subcategory = report["benchmarks"]["spacecast"]["by_subcategory"]
    assert set(by_subcategory) == {"Global Prediction", "Static Perception"}

    # Paper headline metric: unweighted mean over (level, task-type) groups
    # = mean(1.0 for (L3, Rotate Camera), 0.5 for (L1, Camera Direction))
    scored = runner.score("spacecast", runner.load("spacecast"),
                          {"q1": "Answer: C", "q2": "A", "q3": "no idea"})
    assert spacecast_macro_type_accuracy(scored) == pytest.approx(0.75)

    formatted = benchmarks.format_benchmark_report(report)
    assert "spacecast" in formatted
    assert "Global Prediction" in formatted
