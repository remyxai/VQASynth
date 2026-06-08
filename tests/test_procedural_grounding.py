"""
Tests for the PGT procedural-grounding diagnostic and its wiring into the
existing benchmark registry (vqasynth.benchmarks).
"""

from vqasynth.benchmarks import (
    BENCHMARK_LOADERS,
    BENCHMARK_SCORERS,
    BenchmarkRunner,
    load_benchmark,
)
from vqasynth.evaluation import extract_yes_no


def test_pgt_registered_in_benchmark_registries():
    # The call-site edit must expose PGT through the existing dispatch tables.
    assert "pgt" in BENCHMARK_LOADERS
    assert "pgt" in BENCHMARK_SCORERS


def test_load_pgt_through_unified_loader():
    items = load_benchmark("pgt", num_items=9, seed=1, image_size=128)
    assert len(items) == 9
    for item in items:
        # Conforms to the shared normalized item schema.
        assert item["source"] == "PGT"
        assert item["question"]
        assert item["answer"]
        assert item["qa_type"] in {"relation_yn", "relation_choice", "count"}
        assert isinstance(item["primitives"], list) and item["primitives"]


def test_pgt_ground_truth_is_self_consistent():
    # A model that echoes the ground-truth answer must score a perfect 1.0,
    # proving the generated answers parse through vqasynth.evaluation scorers.
    runner = BenchmarkRunner(benchmarks="pgt")
    items = runner.load("pgt", num_items=30, seed=7, image_size=128)
    predictions = {item["id"]: item["answer"] for item in items}

    result = runner.score("pgt", items, predictions)
    assert result["overall_accuracy"] == 1.0
    assert result["total"] == 30


def test_pgt_scorer_distinguishes_wrong_answers():
    runner = BenchmarkRunner(benchmarks="pgt")
    items = runner.load("pgt", num_items=12, seed=3, image_size=128)

    # Flip yes/no predictions so the relational-judgment items are wrong.
    predictions = {}
    for item in items:
        if item["qa_type"] == "relation_yn":
            truth = extract_yes_no(item["answer"])
            predictions[item["id"]] = "No." if truth else "Yes."
        else:
            predictions[item["id"]] = item["answer"]

    result = runner.score("pgt", items, predictions)
    yn_scores = [
        pi["score"] for pi, it in zip(result["per_item"], items)
        if it["qa_type"] == "relation_yn"
    ]
    assert yn_scores  # there is at least one yes/no item
    assert all(s == 0.0 for s in yn_scores)


def test_pgt_runs_end_to_end_via_runner():
    # Exercises BenchmarkRunner.run, the same entrypoint used for the external
    # spatial-reasoning benchmarks, now driving the PGT diagnostic.
    runner = BenchmarkRunner(benchmarks="pgt")
    items = runner.load("pgt", num_items=6, seed=0, image_size=96)
    predictions = {item["id"]: item["answer"] for item in items}

    report = runner.run(
        {"pgt": predictions},
        load_kwargs={"pgt": {"num_items": 6, "seed": 0, "image_size": 96}},
    )
    assert "pgt" in report["summary"]
    assert report["summary"]["pgt"] == 1.0
