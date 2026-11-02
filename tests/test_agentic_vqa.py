"""Structural tests for vqasynth.agentic_vqa — AgentVQA in the eval stage.

Mirrors the philosophy of ``tests/test_correspondence.py``: verify the loader,
scorer and registry wiring without a GPU, model download, or network. The
HuggingFace client is stubbed, and the integration assertions go through the
pre-existing ``vqasynth.benchmarks`` stage (``BENCHMARK_LOADERS``,
``BenchmarkRunner``, ``format_benchmark_report``) rather than through
``vqasynth.agentic_vqa`` alone.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

from vqasynth.agentic_vqa import (
    AGENTVQA_BENCHMARK_NAME,
    AGENTVQA_DOMAINS,
    AGENTVQA_DOMAIN_BY_SUBSET,
    AGENTVQA_SUBSETS,
    _format_option,
    load_agentvqa,
    register_agentvqa,
    score_agentvqa_item,
    subsets_for_domains,
)

# Rows shaped like the AgentVQA/AgentVQA parquet schema (Image/Question/
# Options/Answer). Pointing options arrive as [x, y] pixel coordinates.
ROBOT_ROWS = [
    {
        "Question": "Which action should the robot take next?",
        "Options": ["grasp the block", "open the drawer"],
        "Answer": "A",
        "Image": "<pil-image-0>",
    },
    {
        "Question": "Where should the robot grasp?",
        "Options": [[124, 356], [88, 91]],
        "Answer": "B",
        "Image": "<pil-image-1>",
    },
]

WEB_ROWS = [
    {
        "Question": "Which button submits the form?",
        "Options": ["Settings", "Submit", "Cancel"],
        "Answer": "C",
        "Image": "<pil-image-2>",
    },
]

VIDEO_ROW = {
    "Question": "How many objects were moved by the end of the video?",
    "Options": ["one", "two", "three"],
    "Answer": "A",
    # Video subsets reference a media/ file instead of an embedded still.
    "file_name": "media/VSI-Bench/vid_0001.mp4",
}


@pytest.fixture(scope="module")
def benchmarks_module():
    """The pre-existing evaluation stage, ``vqasynth.benchmarks``.

    Some environments ship a ``datasets`` wheel whose pyarrow build mismatches
    the installed numpy, so importing it fails outright. The registry wiring
    under test does not depend on the real HF client — the loader below is
    exercised against a stubbed ``load_dataset`` — so import through a stand-in
    module rather than skipping the integration assertions entirely.
    """
    try:
        return importlib.import_module("vqasynth.benchmarks")
    except ImportError:
        saved = sys.modules.get("datasets")
        stub = types.ModuleType("datasets")
        stub.load_dataset = None
        sys.modules["datasets"] = stub
        try:
            return importlib.import_module("vqasynth.benchmarks")
        finally:
            if saved is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = saved


@pytest.fixture
def stub_load_dataset(monkeypatch):
    """Replace ``datasets.load_dataset`` with an in-memory subset -> rows map."""
    calls = []
    rows_by_subset = {}

    def load_dataset(repo_id, subset, split="train", streaming=False):
        calls.append((repo_id, subset, split))
        return list(rows_by_subset.get(subset, []))

    stub = types.ModuleType("datasets")
    stub.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", stub)
    return rows_by_subset, calls


# ── domain / subset taxonomy ───────────────────────────────────────────


def test_domain_mapping_covers_every_subset():
    assert len(AGENTVQA_SUBSETS) == 23
    assert set(AGENTVQA_SUBSETS) == set(AGENTVQA_DOMAIN_BY_SUBSET)
    assert set(AGENTVQA_DOMAIN_BY_SUBSET.values()) == set(AGENTVQA_DOMAINS)
    assert len(AGENTVQA_DOMAINS) == 5


def test_subsets_for_domains_selects_the_spatial_cut():
    spatial = subsets_for_domains(["Spatial Planning"])
    assert "3DSRBench" in spatial and "SpaCE-10" in spatial
    assert "AitW" not in spatial
    # Two of the spatial subsets are re-curated cuts of benchmarks this repo
    # already loads under its own name.
    assert {"OmniSpatial", "SpaCE-10"} <= set(spatial)

    with pytest.raises(ValueError, match="Unknown AgentVQA domain"):
        subsets_for_domains(["Web Agents", "Nope"])


# ── loader ─────────────────────────────────────────────────────────────


def test_loader_normalizes_rows_to_the_common_schema(stub_load_dataset):
    rows_by_subset, calls = stub_load_dataset
    rows_by_subset["ERQA"] = ROBOT_ROWS

    items = load_agentvqa(subsets=["ERQA"])

    assert calls == [("AgentVQA/AgentVQA", "ERQA", "train")]
    assert [item["id"] for item in items] == ["ERQA_0", "ERQA_1"]
    first, second = items
    assert first["question"] == "Which action should the robot take next?"
    assert first["answer"] == "A"
    assert first["options"] == ["grasp the block", "open the drawer"]
    assert first["images"] == ["<pil-image-0>"]
    assert first["question_type"] == "multi-choice"
    assert first["category"] == "Robotics"
    assert first["subcategory"] == "ERQA"
    assert first["source"] == "AgentVQA"
    # Pointing options are rendered as text so (A)/(B) prompting still works.
    assert second["options"] == ["(124, 356)", "(88, 91)"]


def test_loader_skips_video_only_rows_unless_asked(stub_load_dataset):
    rows_by_subset, _ = stub_load_dataset
    rows_by_subset["VSI-Bench"] = [dict(ROBOT_ROWS[0]), VIDEO_ROW]

    kept = load_agentvqa(subsets=["VSI-Bench"])
    assert [item["id"] for item in kept] == ["VSI-Bench_0"]

    all_items = load_agentvqa(subsets=["VSI-Bench"], require_image=False)
    assert [item["id"] for item in all_items] == ["VSI-Bench_0", "VSI-Bench_1"]
    assert all_items[1]["images"] == []


def test_loader_rejects_unknown_subset_before_touching_huggingface():
    with pytest.raises(ValueError, match="Unknown AgentVQA subset"):
        load_agentvqa(subsets=["NotASubset"])


def test_format_option_renders_coordinates_as_text():
    assert _format_option([124, 356]) == "(124, 356)"
    assert _format_option([124.0, 356.5]) == "(124, 356.5)"
    assert _format_option("  open the drawer ") == "open the drawer"


# ── scorer ─────────────────────────────────────────────────────────────


def _mc_item(answer="B", options=("grasp the block", "open the drawer")):
    return {"answer": answer, "options": list(options), "question_type": "multi-choice"}


def test_scorer_matches_extracted_letter():
    assert score_agentvqa_item(_mc_item(), "The answer is (B).") == 1.0
    assert score_agentvqa_item(_mc_item(), "A") == 0.0


def test_scorer_falls_back_to_option_text():
    # No letter in the prediction — the gold option's text still counts.
    assert score_agentvqa_item(_mc_item(), "The robot should open the drawer.") == 1.0
    assert score_agentvqa_item(_mc_item(), "The robot should grasp the block.") == 0.0


def test_scorer_handles_pointing_answers():
    item = _mc_item(answer="A", options=("(124, 356)", "(88, 91)"))
    assert score_agentvqa_item(item, "(124, 356)") == 1.0
    assert score_agentvqa_item(item, "(88, 91)") == 0.0
    assert score_agentvqa_item(item, "A") == 1.0


# ── integration with the existing benchmark stage ─────────────────────


def test_register_agentvqa_extends_existing_registry(benchmarks_module):
    # Start from a clean registry so the test holds regardless of test order.
    benchmarks_module.BENCHMARK_LOADERS.pop(AGENTVQA_BENCHMARK_NAME, None)
    benchmarks_module.BENCHMARK_SCORERS.pop(AGENTVQA_BENCHMARK_NAME, None)

    assert register_agentvqa() == AGENTVQA_BENCHMARK_NAME
    assert benchmarks_module.BENCHMARK_LOADERS[AGENTVQA_BENCHMARK_NAME] is load_agentvqa
    assert benchmarks_module.BENCHMARK_SCORERS[AGENTVQA_BENCHMARK_NAME] is score_agentvqa_item
    # BenchmarkRunner's "all" default now carries the fifth benchmark.
    assert AGENTVQA_BENCHMARK_NAME in benchmarks_module.BenchmarkRunner().benchmarks

    register_agentvqa()  # idempotent
    assert list(benchmarks_module.BENCHMARK_LOADERS).count(AGENTVQA_BENCHMARK_NAME) == 1


def test_load_benchmark_routes_to_the_agentvqa_loader(benchmarks_module, stub_load_dataset):
    rows_by_subset, _ = stub_load_dataset
    rows_by_subset["ERQA"] = ROBOT_ROWS
    register_agentvqa()

    items = benchmarks_module.load_benchmark(AGENTVQA_BENCHMARK_NAME, subsets=["ERQA"])

    assert [item["id"] for item in items] == ["ERQA_0", "ERQA_1"]
    assert all(item["category"] == "Robotics" for item in items)


def test_benchmark_runner_scores_agentvqa_by_domain(benchmarks_module, stub_load_dataset):
    rows_by_subset, _ = stub_load_dataset
    rows_by_subset["ERQA"] = ROBOT_ROWS
    rows_by_subset["AitW"] = WEB_ROWS
    register_agentvqa()

    runner = benchmarks_module.BenchmarkRunner(AGENTVQA_BENCHMARK_NAME)
    items = runner.load(AGENTVQA_BENCHMARK_NAME, subsets=["ERQA", "AitW"])
    predictions = {
        "ERQA_0": "A",                       # correct letter
        "ERQA_1": "The answer is (B).",      # correct letter, prose
        "AitW_0": "no letter here",          # unparseable -> 0
    }

    result = runner.score(AGENTVQA_BENCHMARK_NAME, items, predictions)

    assert result["total"] == 3
    assert result["overall_accuracy"] == pytest.approx(2 / 3)
    assert result["by_category"]["Robotics"] == {
        "accuracy": pytest.approx(1.0),
        "count": 2,
    }
    assert result["by_category"]["Web Agents"] == {
        "accuracy": pytest.approx(0.0),
        "count": 1,
    }
    # Source-dataset breakdown is preserved for the cross-domain read.
    assert set(result["by_subcategory"]) == {"ERQA", "AitW"}

    report = benchmarks_module.format_benchmark_report({
        "summary": {result["benchmark"]: result["overall_accuracy"]},
        "benchmarks": {result["benchmark"]: {
            key: value for key, value in result.items() if key != "per_item"
        }},
    })
    assert "Robotics" in report and "Web Agents" in report
    assert "agentvqa" in report
