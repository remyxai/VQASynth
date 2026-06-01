"""
Tests for cross-view stratification wired into BenchmarkRunner.score.

Exercises the integration end-to-end through the existing
``vqasynth.benchmarks`` module (not just the new module in isolation):
synthetic single-view and cross-view items are scored, and we assert the
cross-view breakdown the scoring path now emits.
"""

from vqasynth.benchmarks import BenchmarkRunner, format_benchmark_report
from vqasynth.cross_view_consistency import (
    cross_view_breakdown,
    is_cross_view,
    view_count,
    view_stratum,
)


def _items():
    # Two single-view items (one image each) and two cross-view items
    # (multiple images / multi-view category), so each stratum is populated.
    return [
        {
            "id": "sv1", "question": "Is the cup left of the plate?", "answer": "A",
            "question_type": "multi-choice", "category": "relation",
            "subcategory": "relation", "options": ["A", "B"], "images": ["a.png"],
        },
        {
            "id": "sv2", "question": "Is the cup left of the plate?", "answer": "B",
            "question_type": "multi-choice", "category": "relation",
            "subcategory": "relation", "options": ["A", "B"], "images": ["b.png"],
        },
        {
            "id": "xv1", "question": "Across the two views, which is closer?", "answer": "A",
            "question_type": "multi-choice", "category": "rotation",
            "subcategory": "rotation", "options": ["A", "B"], "images": ["v0.png", "v1.png"],
        },
        {
            "id": "xv2", "question": "Across the two views, which is closer?", "answer": "B",
            "question_type": "multi-choice", "category": "rotation",
            "subcategory": "rotation", "options": ["A", "B"], "images": ["w0.png", "w1.png"],
        },
    ]


def test_view_helpers():
    items = _items()
    assert view_count(items[0]) == 1
    assert view_count(items[2]) == 2
    assert not is_cross_view(items[0])
    assert is_cross_view(items[2])
    # Category keyword alone marks cross-view even with a single image.
    stitched = {"category": "rotation", "subcategory": "rotation", "images": ["one.png"]}
    assert view_stratum(stitched) == "cross-view"


def test_score_emits_cross_view_breakdown():
    runner = BenchmarkRunner(benchmarks="mindcube")
    items = _items()
    # Single-view answered perfectly; cross-view answered wrong -> a clear gap.
    predictions = {"sv1": "A", "sv2": "B", "xv1": "B", "xv2": "A"}

    result = runner.score("mindcube", items, predictions)

    assert "cross_view" in result
    cv = result["cross_view"]
    assert cv["by_stratum"]["single-view"] == {"accuracy": 1.0, "count": 2}
    assert cv["by_stratum"]["cross-view"] == {"accuracy": 0.0, "count": 2}
    # CrossViewBench signal: full degradation from single-view to cross-view.
    assert cv["cross_view_gap"] == 1.0
    assert cv["by_view_count"][1]["count"] == 2
    assert cv["by_view_count"][2]["count"] == 2


def test_breakdown_gap_none_without_both_strata():
    items = _items()[:2]  # only single-view
    per_item = [{"id": "sv1", "score": 1.0}, {"id": "sv2", "score": 1.0}]
    cv = cross_view_breakdown(items, per_item)
    assert cv["by_stratum"]["cross-view"]["count"] == 0
    assert cv["cross_view_gap"] is None


def test_report_renders_cross_view_lines():
    runner = BenchmarkRunner(benchmarks="mindcube")
    # Build a report dict directly to avoid network loaders, then format it.
    result = runner.score("mindcube", _items(), {"sv1": "A", "sv2": "B", "xv1": "B", "xv2": "A"})
    report = {
        "summary": {"mindcube": result["overall_accuracy"]},
        "benchmarks": {
            "mindcube": {
                "overall_accuracy": result["overall_accuracy"],
                "total": result["total"],
                "by_category": result["by_category"],
                "by_subcategory": result["by_subcategory"],
                "cross_view": result["cross_view"],
            }
        },
    }
    text = format_benchmark_report(report)
    assert "Cross-view" in text
    assert "Cross-view gap" in text
