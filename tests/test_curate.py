"""Structural tests for vqasynth.curate.

Verifies the max-min (farthest-point) selection algorithm and the random
baseline against small synthetic embedding matrices -- no CLIP, no CUDA, just
numpy. One composition test exercises the pre-existing ``vqasynth.datasets``
I/O layer (Dataloader save/load round-trip) that the curate stage composes with.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vqasynth.curate import (
    Curator,
    build_manifest,
    farthest_point_selection,
    random_selection,
    select_indices,
    write_manifest,
)


# --- pure-numpy selection algorithm -----------------------------------------

def test_farthest_selection_is_deterministic_with_start_index():
    emb = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [9.0, 9.0], [3.0, 1.0]])
    a = farthest_point_selection(emb, 3, seed=0, start_index=0)
    b = farthest_point_selection(emb, 3, seed=0, start_index=0)
    assert a.tolist() == b.tolist()


def test_farthest_first_point_is_seed_driven_and_reproducible():
    emb = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [9.0, 9.0], [3.0, 1.0]])
    a = farthest_point_selection(emb, 3, seed=42)
    b = farthest_point_selection(emb, 3, seed=42)
    assert a.tolist() == b.tolist()  # same seed -> identical, including first point


def test_farthest_max_min_invariant():
    # Four 1-D points with a unique greedy path and no distance ties.
    emb = np.array([[0.0], [1.0], [2.0], [10.0]])
    sel = farthest_point_selection(emb, 3, seed=0, start_index=0)

    assert sel[0] == 0          # start
    assert sel[1] == 3          # farthest from {0} is the point at 10
    assert sel[2] == 2          # min-dist to {0,3}: [_,1,2,_] -> argmax is 2

    # Generic invariant: each step is the argmax over remaining points of the
    # minimum distance to the already-selected prefix.
    for t in range(1, len(sel)):
        prefix = sel[:t].tolist()
        remaining = [i for i in range(len(emb)) if i not in prefix]
        best = max(
            remaining,
            key=lambda i: min(abs(emb[i, 0] - emb[j, 0]) for j in prefix),
        )
        assert sel[t] == best


def test_farthest_k_equals_n_returns_permutation_of_all():
    emb = np.array([[0.0], [1.0], [2.0], [10.0]])
    sel = farthest_point_selection(emb, 4, seed=0, start_index=0)
    assert sorted(sel.tolist()) == [0, 1, 2, 3]
    assert len(set(sel.tolist())) == 4


def test_farthest_k_zero_returns_empty():
    sel = farthest_point_selection(np.zeros((3, 2)), 0, seed=0)
    assert sel.shape == (0,)


def test_farthest_rejects_k_too_large():
    with pytest.raises(ValueError, match="exceeds"):
        farthest_point_selection(np.zeros((3, 2)), 4, seed=0)


def test_farthest_rejects_unknown_metric():
    with pytest.raises(ValueError, match="metric"):
        farthest_point_selection(np.zeros((3, 2)), 1, seed=0, metric="bogus")


def test_cosine_metric_runs_and_is_deterministic():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    a = farthest_point_selection(emb, 2, seed=0, metric="cosine", start_index=0)
    b = farthest_point_selection(emb, 2, seed=0, metric="cosine", start_index=0)
    assert a.tolist() == b.tolist()
    # the point most dissimilar (cosine) to [1,0] is [0,1]
    assert a[1] == 1


def test_random_selection_reproducible_distinct_and_in_range():
    a = random_selection(10, 4, seed=0)
    b = random_selection(10, 4, seed=0)
    assert a.tolist() == b.tolist()      # reproducible
    assert len(a) == 4
    assert len(set(a.tolist())) == 4     # distinct
    assert all(0 <= i < 10 for i in a)   # in range


def test_random_selection_k_zero_returns_empty():
    assert random_selection(5, 0, seed=0).shape == (0,)


def test_select_indices_dispatches_by_strategy():
    emb = np.array([[0.0], [1.0], [2.0], [10.0]])
    r = select_indices(emb, 2, strategy="random", seed=0)
    assert len(r) == 2 and len(set(r.tolist())) == 2
    # farthest dispatch is reproducible and respects the seed
    f1 = select_indices(emb, 2, strategy="farthest", seed=0)
    f2 = select_indices(emb, 2, strategy="farthest", seed=0)
    assert f1.tolist() == f2.tolist() and len(f1) == 2


def test_select_indices_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        select_indices(np.zeros((3, 2)), 1, strategy="bogus")


# --- manifest ---------------------------------------------------------------

def test_build_manifest_records_selection_metadata():
    manifest = build_manifest(
        source_repo_id="remyxai/SpaceThinker",
        split="train",
        source_count=1000,
        selected_ids=np.array([3, 7, 42]),
        k=3,
        strategy="farthest",
        seed=0,
        metric="euclidean",
        embedding_source="openai/clip:ViT-B/32",
        fraction=0.25,
    )
    assert manifest["selected_ids"] == [3, 7, 42]
    assert manifest["selected_count"] == 3
    assert manifest["source_count"] == 1000
    assert manifest["fraction"] == 0.25
    assert manifest["strategy"] == "farthest"
    assert manifest["seed"] == 0
    assert "arXiv:2506.24120" in manifest["method"]


def test_write_manifest_creates_parents_and_roundtrips(tmp_path):
    manifest = build_manifest(
        source_repo_id="r/s",
        split="train",
        source_count=10,
        selected_ids=[0, 1],
        k=2,
        strategy="random",
        seed=1,
        metric="euclidean",
        embedding_source="precomputed:embedding",
        fraction=None,
    )
    path = tmp_path / "nested" / "variant" / "curate_manifest.json"
    write_manifest(manifest, str(path))
    assert path.exists()
    assert json.loads(path.read_text())["selected_ids"] == [0, 1]


# --- composition with the pre-existing vqasynth.datasets I/O layer ----------

def test_curator_reads_precomputed_embedding_column():
    """Curator reads the `embedding` column (the schema the embeddings stage
    writes) without loading any model. Pure structural composition."""
    datasets = pytest.importorskip("datasets")
    # Embedding vectors hand-picked so the greedy path is well-defined.
    ds = datasets.Dataset.from_dict({
        "embedding": [[0.0], [1.0], [2.0], [10.0]],
        "label": ["a", "b", "c", "d"],
    })
    curator = Curator(embedding_source=None, seed=0)
    subset, manifest = curator.curate(ds, count=3, strategy="farthest")

    assert manifest["embedding_source"] == "precomputed:embedding"
    assert manifest["selected_count"] == 3
    assert manifest["source_count"] == 4
    assert subset.num_rows == 3
    assert len(set(manifest["selected_ids"])) == 3
    assert all(0 <= i < 4 for i in manifest["selected_ids"])
    # subset rows are a true slice of the source
    assert set(subset["label"]).issubset({"a", "b", "c", "d"})


def test_curated_subset_roundtrips_through_dataloader(tmp_path):
    """The curated subset composes with the pre-existing vqasynth.datasets
    Dataloader: save it, then reload it through Dataloader's local-cache branch
    (offline, no Hub). This is the exact I/O contract the curate docker stage
    relies on."""
    datasets = pytest.importorskip("datasets")
    from vqasynth.datasets import Dataloader

    ds = datasets.Dataset.from_dict({
        "embedding": [[0.0], [1.0], [2.0], [10.0]],
        "label": ["a", "b", "c", "d"],
    })
    curator = Curator(embedding_source=None, seed=0)
    subset, _ = curator.curate(ds, count=2, strategy="farthest")
    assert subset.num_rows == 2

    cache = tmp_path / "cache"
    cache.mkdir()
    dl = Dataloader(str(cache))
    dl.dataset_name = "curated_subset"
    # In the real pipeline the source dataset is loaded into the cache first, so
    # the final path already exists when save_to_disk overwrites it.
    (cache / "curated_subset").mkdir()

    dl.save_to_disk(subset)

    # Reload via the public Dataloader.load_dataset local-cache path (offline).
    dl2 = Dataloader(str(cache))
    reloaded = dl2.load_dataset("org/curated_subset")
    assert reloaded.num_rows == 2
    assert set(reloaded["label"]).issubset({"a", "b", "c", "d"})


def test_curator_rejects_fraction_and_count_together():
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"embedding": [[0.0], [1.0], [2.0], [10.0]]})
    curator = Curator(embedding_source=None, seed=0)
    with pytest.raises(ValueError, match="fraction OR count"):
        curator.curate(ds, fraction=0.5, count=2)
