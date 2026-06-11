"""
Cross-view stratification for spatial reasoning benchmark results.

Adapted from CrossView Suite (CrossViewBench), which argues that a model's
single-view perception accuracy systematically overstates its real spatial
intelligence: the hard, distinguishing capability is reasoning *consistently
about the same objects across multiple viewpoints*. Their benchmark is built
to surface that gap.

This module brings that evaluation lens to VQASynth's existing benchmark
scoring. Given the normalized benchmark items and the per-item scores produced
by ``vqasynth.benchmarks.BenchmarkRunner.score``, it splits results into
single-view vs. cross-view strata and reports the cross-view performance gap
(single-view accuracy minus cross-view accuracy). A large positive gap is the
signal the paper highlights: the model handles isolated images but fails to
align objects across views.

It is deliberately a metric/stratification layer only. The paper's CrossViewer
model (adaptive region tokenizer, explicit multi-view object alignment) and its
1.6M-sample CrossViewSet training corpus are out of scope here — VQASynth is a
data-generation and evaluation pipeline, not a trainer.
"""

# Benchmark categories/subcategories that denote multi-viewpoint scenes even
# when the views are stitched into a single image (e.g. MindCube settings).
CROSS_VIEW_KEYWORDS = (
    "rotation",
    "around",
    "among",
    "translation",
    "cross-view",
    "cross view",
    "multi-view",
    "multi view",
    "viewpoint",
    "perspective",
)


def view_count(item):
    """Number of viewpoints (images) attached to a benchmark item."""
    images = item.get("images") or []
    return len(images)


def is_cross_view(item, min_views=2):
    """
    Whether an item exercises cross-view reasoning.

    True when the item carries at least ``min_views`` images, or when its
    category/subcategory names a multi-viewpoint setting (some benchmarks
    compose several views into one image but tag the scene structure).
    """
    if view_count(item) >= min_views:
        return True
    tags = "{} {}".format(item.get("category", ""), item.get("subcategory", "")).lower()
    return any(kw in tags for kw in CROSS_VIEW_KEYWORDS)


def view_stratum(item, min_views=2):
    """Return ``"cross-view"`` or ``"single-view"`` for a benchmark item."""
    return "cross-view" if is_cross_view(item, min_views=min_views) else "single-view"


def _agg(scores):
    return {
        "accuracy": sum(scores) / len(scores) if scores else 0.0,
        "count": len(scores),
    }


def cross_view_breakdown(items, per_item, min_views=2):
    """
    Stratify per-item scores by cross-view structure.

    Args:
        items: List of normalized benchmark items (must carry ``id`` and
            ``images``; ``category``/``subcategory`` used as a fallback signal).
        per_item: List of per-item score dicts from ``BenchmarkRunner.score``,
            each with at least ``id`` and ``score``.
        min_views: Minimum image count to treat an item as cross-view.

    Returns a dict with:
        - ``by_stratum``: accuracy/count for "single-view" and "cross-view"
        - ``by_view_count``: accuracy/count keyed by exact number of views
        - ``cross_view_gap``: single-view accuracy minus cross-view accuracy
          (None if either stratum is empty), the CrossViewBench-style signal
          of how much performance degrades across viewpoints.
    """
    item_by_id = {item["id"]: item for item in items}

    stratum_scores = {"single-view": [], "cross-view": []}
    view_count_scores = {}

    for record in per_item:
        item = item_by_id.get(record["id"])
        if item is None:
            continue
        score = record["score"]

        stratum_scores[view_stratum(item, min_views=min_views)].append(score)
        view_count_scores.setdefault(view_count(item), []).append(score)

    by_stratum = {name: _agg(scores) for name, scores in stratum_scores.items()}

    single_acc = by_stratum["single-view"]["accuracy"]
    cross_acc = by_stratum["cross-view"]["accuracy"]
    if by_stratum["single-view"]["count"] and by_stratum["cross-view"]["count"]:
        cross_view_gap = single_acc - cross_acc
    else:
        cross_view_gap = None

    return {
        "by_stratum": by_stratum,
        "by_view_count": {vc: _agg(s) for vc, s in sorted(view_count_scores.items())},
        "cross_view_gap": cross_view_gap,
    }
