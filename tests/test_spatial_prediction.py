"""Structural tests for vqasynth.spatial_prediction.

Verifies the SpaceCast-Bench loader/scorer (predictive spatial reasoning,
observe-transform-infer) against hand-computed small examples — no CUDA, no
model download, no Hub fetch. Mirrors the dependency-free style of
tests/test_visual_credit.py.

Integration with the PRE-EXISTING evaluation stage is asserted directly:
``spacecast`` must be registered in ``vqasynth.benchmarks``'
``BENCHMARK_LOADERS`` / ``BENCHMARK_SCORERS``, must be picked up by
``BenchmarkRunner``'s default benchmark list, and must round-trip through
``BenchmarkRunner.run`` so the paper's L1/L2/L3 level ladder lands in
``by_category``. Those modules are not created by this PR, so the composition
tests exercise real integrated behavior rather than self-testing the new file.
"""
from __future__ import annotations

import subprocess
import sys
import types

import pytest

from vqasynth import spatial_prediction
from vqasynth.spatial_prediction import (
    SPACECAST_LEVELS,
    extract_option_letters,
    normalize_spacecast_entry,
    score_spacecast_item,
)


def benchmarks_module():
    """
    Import ``vqasynth.benchmarks`` even when ``datasets`` cannot be imported.

    The registry module imports ``load_dataset`` at module top; in minimal
    environments (e.g. the repo's numpy 1.26 / pyarrow>=2 pin mismatch) that
    import fails. The registry wiring under test has nothing to do with
    HuggingFace datasets, so fall back to a never-called stub rather than
    skipping the integration tests.
    """
    try:
        import vqasynth.benchmarks as benchmarks
    except ImportError:
        stub = types.ModuleType("datasets")
        stub.load_dataset = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("datasets stub: no Hub access in these tests")
        )
        sys.modules.setdefault("datasets", stub)
        import vqasynth.benchmarks as benchmarks
    return benchmarks


# One raw row of hongxingli/SpaceCast-Bench (test split), abridged: a
# multi-image camera path, a rotation premise, four lettered options.
RAW_ROW = {
    "question_uid": "9876ae5a033b64b3ab1a35a894f898dc21182ca8",
    "dataset": "scannet",
    "scene_id": "scene0025_01",
    "level": "L3",
    "type": "Rotate Camera Direction",
    "source_type": "coordinate_rotation_agent",
    "question": (
        "Identify the book in the first photo and the pillow in the last photo. "
        "If the room were rotated 180 degrees clockwise about the camera position "
        "in the first photo, in which direction would the book be from the pillow?"
    ),
    "options": ["back-left", "front-left", "back-right", "front-right"],
    "option_labels": ["A", "B", "C", "D"],
    "answer": ["C"],
    "correct_values": ["back-right"],
    "multi_select": False,
    "images": ["492.jpg", "522.jpg", "564.jpg"],
    "image_roles": ["main_a", "bridge", "main_b"],
    "image_paths": [
        "scene0025_01/492.jpg",
        "scene0025_01/522.jpg",
        "scene0025_01/564.jpg",
    ],
    "main_image_a_name": "492.jpg",
    "main_image_b_name": "564.jpg",
    "bridge_view_names": ["522.jpg"],
    "relation_unchanged": False,
}


def _row(**overrides):
    row = dict(RAW_ROW)
    row.update(overrides)
    return row


# --- normalization ----------------------------------------------------------

def test_normalize_maps_paper_taxonomy_onto_report_categories():
    item = normalize_spacecast_entry(_row())
    assert item["question_type"] == "multi-choice"
    assert item["category"] == "L3 global prediction"
    assert item["subcategory"] == "Rotate Camera Direction"
    assert item["source"] == "SpaceCast-Bench"
    # Gold is exposed both as letters (for option scoring) and as text (for the
    # LLM-judge fallback), mirroring the OmniSpatial loader.
    assert item["answers"] == ["C"]
    assert item["answer"] == "C"
    assert item["answer_text"] == "back-right"
    assert len(item["images"]) == 3


def test_normalize_maps_text_answers_back_to_option_labels():
    item = normalize_spacecast_entry(_row(answer=["back-right"], correct_values=[]))
    assert item["answers"] == ["C"]
    # Nothing left to map -> no option letters, only free-text gold.
    assert normalize_spacecast_entry(_row(answer=["2 meters"], options=[], option_labels=[]))[
        "answers"
    ] == []


def test_level_ladder_sorts_in_report_order():
    assert [SPACECAST_LEVELS[key] for key in sorted(SPACECAST_LEVELS)] == [
        "L1 static perception",
        "L2 local prediction",
        "L3 global prediction",
    ]


def test_bridge_view_ablation_drops_only_bridging_views():
    full = normalize_spacecast_entry(_row(), include_bridge_views=True)
    ablated = normalize_spacecast_entry(_row(), include_bridge_views=False)
    assert full["images"] == ["492.jpg", "522.jpg", "564.jpg"]
    # Observe (main_a) and infer (main_b) views stay; the connecting view goes.
    assert ablated["images"] == ["492.jpg", "564.jpg"]
    assert ablated["image_roles"] == ["main_a", "main_b"]
    # Everything the model is scored on is unchanged by the ablation.
    for key in ("question", "options", "answers", "correct_values"):
        assert ablated[key] == full[key]


def test_bridge_view_ablation_without_roles_uses_view_names():
    row = _row(image_roles=[])
    ablated = normalize_spacecast_entry(row, include_bridge_views=False)
    assert ablated["images"] == ["492.jpg", "564.jpg"]


# --- answer extraction + scoring --------------------------------------------

def test_extract_option_letters_reads_only_the_answer_span():
    assert extract_option_letters("Answer: A and C") == ["A", "C"]
    assert extract_option_letters("The answer is B, D.") == ["B", "D"]
    # Duplicates collapse, letters outside the answer span are ignored.
    assert extract_option_letters("A or A, then C") == ["A", "C"]
    assert extract_option_letters("the book is to the back-right") == []
    assert extract_option_letters("") == []


def test_score_single_select_letter_match_and_text_fallback():
    item = normalize_spacecast_entry(_row())
    assert score_spacecast_item(item, "The answer is C.") == 1.0
    assert score_spacecast_item(item, "(A)") == 0.0
    # A model that answers with the option text instead of its letter.
    assert score_spacecast_item(item, "The book would be to the back-right.") == 1.0
    assert score_spacecast_item(item, "The book would be to the back-left.") == 0.0


def test_score_multi_select_requires_the_exact_set():
    item = normalize_spacecast_entry(
        _row(answer=["A", "C"], correct_values=[], multi_select=True)
    )
    assert score_spacecast_item(item, "Answer: C and A") == 1.0
    assert score_spacecast_item(item, "Answer: A") == 0.0
    assert score_spacecast_item(item, "Answer: A, B, C") == 0.0
    assert score_spacecast_item(item, "I cannot tell") == 0.0


def test_score_open_ended_numeric_reuses_mra():
    item = normalize_spacecast_entry(
        _row(options=[], option_labels=[], answer=["2 meters"], correct_values=["2 meters"])
    )
    assert item["question_type"] == "open-ended"
    assert score_spacecast_item(item, "About 2.1 meters") == 1.0
    assert score_spacecast_item(item, "About 9 meters") == 0.0


def test_scorer_returns_none_only_when_there_is_no_gold():
    item = normalize_spacecast_entry(
        _row(options=[], option_labels=[], answer=[], correct_values=[])
    )
    # None is what lets BenchmarkRunner fall back to its LLM judge.
    assert score_spacecast_item(item, "somewhere to the right") is None


# --- registry integration (pre-existing vqasynth.benchmarks) ----------------

def test_spacecast_registered_in_benchmark_runner_defaults():
    benchmarks = benchmarks_module()
    assert "spacecast" in benchmarks.BENCHMARK_LOADERS
    assert "spacecast" in benchmarks.BENCHMARK_SCORERS
    assert benchmarks.BENCHMARK_LOADERS["spacecast"] is spatial_prediction.load_spacecast
    assert benchmarks.BENCHMARK_SCORERS["spacecast"] is score_spacecast_item
    # The default ("all") benchmark list picks the registration up.
    assert "spacecast" in benchmarks.BenchmarkRunner().benchmarks
    assert "spacecast" in benchmarks.BenchmarkRunner(benchmarks="all").benchmarks


def test_runner_scores_spacecast_with_level_breakdown(monkeypatch):
    benchmarks = benchmarks_module()
    rows = [
        _row(question_uid="q1", level="L1", type="Camera Direction", answer=["A"]),
        _row(question_uid="q2", level="L2", type="Object State Change",
             answer=["A", "C"], multi_select=True),
        _row(question_uid="q3", level="L3", answer=["C"]),
    ]
    monkeypatch.setattr(
        spatial_prediction, "_fetch_spacecast", lambda split, streaming: iter(rows)
    )

    runner = benchmarks.BenchmarkRunner(benchmarks=["spacecast"])
    # Predictions as a list exercise the order -> item-id mapping in score().
    report = runner.run({"spacecast": ["Answer: A", "Answer: A", "The answer is C"]})

    assert report["summary"]["spacecast"] == pytest.approx(2 / 3)
    by_category = report["benchmarks"]["spacecast"]["by_category"]
    assert by_category["L1 static perception"]["accuracy"] == 1.0
    assert by_category["L2 local prediction"]["accuracy"] == 0.0
    assert by_category["L3 global prediction"]["accuracy"] == 1.0
    assert list(by_category) == [
        "L1 static perception",
        "L2 local prediction",
        "L3 global prediction",
    ]

    formatted = benchmarks.format_benchmark_report(report)
    assert "spacecast" in formatted
    assert "L3 global prediction" in formatted
    assert "Rotate Camera Direction" in formatted  # task type as subcategory

    # The inference-facing view hides ground truth but keeps the observe views.
    eval_items = runner.get_benchmark_items("spacecast")
    assert [item["id"] for item in eval_items] == ["q1", "q2", "q3"]
    assert all("answer" not in item for item in eval_items)
    assert eval_items[0]["images"] == ["492.jpg", "522.jpg", "564.jpg"]


def test_runner_loads_spacecast_without_bridging_views(monkeypatch):
    benchmarks = benchmarks_module()
    monkeypatch.setattr(
        spatial_prediction, "_fetch_spacecast", lambda split, streaming: iter([_row()])
    )
    runner = benchmarks.BenchmarkRunner(benchmarks=["spacecast"])
    # Loader kwargs flow through the runner, so the ablation is reachable from
    # the eval stage without a bespoke code path.
    items = runner.load("spacecast", include_bridge_views=False)
    assert [item["images"] for item in items] == [["492.jpg", "564.jpg"]]


def test_module_imports_without_datasets():
    """``datasets`` is a runtime-only dep of the Hub fetch path and must never
    be imported at module import time (see vqasynth.detection_3d's lazy open3d /
    numpy imports). Checked in a fresh interpreter so test ordering cannot
    pollute sys.modules."""
    code = "import vqasynth.spatial_prediction as m; assert m.load_spacecast"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="."
    )
    assert result.returncode == 0, result.stderr
