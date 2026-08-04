"""Uniform-sampling data curation for synthetic training sets.

Implements the max-min pairwise-distance (farthest-point) selection idea from
SafeRL-Lab/data-uniformity (arXiv:2506.24120) as a portable curation stage:
greedily pick the K samples that maximize the minimum distance to the
already-selected set in embedding space. A random-sampling baseline ships in
the same module so a downstream ablation can switch strategies with one flag.

The selection algorithm is pure numpy and dependency-light (no CLIP, no CUDA);
it consumes any ``(N, D)`` embedding matrix. For a HuggingFace dataset the
default embedding source is OpenAI CLIP ViT-B/32 image embeddings, reused from
``vqasynth.embeddings.EmbeddingGenerator`` (CPU-friendly, dependency-light). If
the dataset already carries an ``embedding`` column (written by the embeddings
stage), that column is read directly and no model is loaded.

The curated subset is a standalone artifact: it composes with a downstream LoRA
training config by pointing ``dataset.repo_id`` at the curated variant.

Refs: https://github.com/remyxai/VQASynth/issues/28
"""
import os
import json

import numpy as np

# Column written by vqasynth.embeddings.EmbeddingGenerator.apply_transform.
# The curator reads it directly when present, avoiding any model load.
EMBEDDING_COLUMN = "embedding"

# CLIP ViT-B/32 image embeddings via vqasynth.embeddings.EmbeddingGenerator.
DEFAULT_EMBEDDING_SOURCE = "openai/clip:ViT-B/32"

VALID_STRATEGIES = ("farthest", "random")
VALID_METRICS = ("euclidean", "cosine")
DEFAULT_SEED = 0


def _to_embeddings_array(embeddings):
    """Coerce embeddings to a 2D float64 array of shape (N, D).

    float64 is used internally so greedy selection is stable across numpy/BLAS
    builds; CLIP produces float32 and the cast is harmless for distance ranking.
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N, D); got shape {arr.shape}")
    return arr


def _dist_to_point(work, idx, metric):
    """Distance from every row of ``work`` to row ``idx`` under ``metric``."""
    if metric == "cosine":
        # ``work`` is L2-normalized for cosine; cosine distance = 1 - similarity
        return 1.0 - work @ work[idx]
    diff = work - work[idx]
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def farthest_point_selection(embeddings, k, seed=DEFAULT_SEED, metric="euclidean", start_index=None):
    """Greedy max-min pairwise-distance selection (farthest-point sampling).

    Picks ``k`` row indices from ``embeddings`` (N, D). The first point is the
    RNG-drawn index (driven by ``seed``) unless ``start_index`` overrides it;
    each subsequent point maximizes the minimum distance to the already-selected
    set. Deterministic given fixed ``embeddings`` + ``seed``.

    Args:
        embeddings: array-like of shape (N, D).
        k: number of samples to select.
        seed: seed for the first-point RNG.
        metric: "euclidean" or "cosine".
        start_index: optional explicit first index (skips the RNG draw).

    Returns:
        np.ndarray[int64] of length ``k`` in selection order.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"unsupported metric {metric!r}; choose from {VALID_METRICS}")
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}")

    emb = _to_embeddings_array(embeddings)
    n = emb.shape[0]
    if k > n:
        raise ValueError(f"k={k} exceeds number of samples n={n}")
    if k == 0:
        return np.empty(0, dtype=np.int64)

    if metric == "cosine":
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        work = emb / norms
    else:
        work = emb

    rng = np.random.default_rng(seed)
    if start_index is None:
        first = int(rng.integers(n))
    else:
        first = int(start_index)
        if not (0 <= first < n):
            raise ValueError(f"start_index {first} out of range [0, {n})")

    selected = [first]
    chosen = np.zeros(n, dtype=bool)
    chosen[first] = True
    # min distance from each point to the current selected set
    min_dist = _dist_to_point(work, first, metric)

    while len(selected) < k:
        # already-chosen points can't be re-picked
        candidates = np.where(chosen, -np.inf, min_dist)
        nxt = int(np.argmax(candidates))
        selected.append(nxt)
        chosen[nxt] = True
        min_dist = np.minimum(min_dist, _dist_to_point(work, nxt, metric))

    return np.array(selected, dtype=np.int64)


def random_selection(n_samples, k, seed=DEFAULT_SEED):
    """Random-sampling baseline: ``k`` distinct indices drawn from ``range(n_samples)``."""
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 0:
        raise ValueError(f"n_samples must be a non-negative integer, got {n_samples!r}")
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}")
    if k > n_samples:
        raise ValueError(f"k={k} exceeds n_samples={n_samples}")
    if k == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(int(n_samples), size=int(k), replace=False).astype(np.int64)


def select_indices(embeddings, k, strategy="farthest", seed=DEFAULT_SEED, metric="euclidean"):
    """Dispatch to the requested selection ``strategy`` and return ``k`` indices.

    Args:
        embeddings: array-like (N, D); used directly for "farthest" and only to
            count N for "random".
        k: number of samples to select.
        strategy: "farthest" (max-min pairwise distance) or "random" (baseline).
        seed: RNG seed.
        metric: distance metric for the "farthest" strategy.
    """
    if strategy == "farthest":
        return farthest_point_selection(embeddings, k, seed=seed, metric=metric)
    if strategy == "random":
        n = _to_embeddings_array(embeddings).shape[0]
        return random_selection(n, k, seed=seed)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {VALID_STRATEGIES}")


def build_manifest(*, source_repo_id, split, source_count, selected_ids, k,
                   strategy, seed, metric, embedding_source, fraction=None):
    """Assemble the curation manifest dict recorded alongside a curated subset."""
    if strategy == "farthest":
        method = ("farthest-point (max-min pairwise distance) selection per "
                  "SafeRL-Lab/data-uniformity, arXiv:2506.24120")
    else:
        method = "random sampling baseline"
    return {
        "source_repo_id": source_repo_id,
        "split": split,
        "source_count": int(source_count),
        "selected_count": int(len(selected_ids)),
        "fraction": None if fraction is None else float(fraction),
        "selected_ids": [int(i) for i in selected_ids],
        "strategy": strategy,
        "seed": int(seed),
        "metric": metric,
        "embedding_source": embedding_source,
        "method": method,
    }


def write_manifest(manifest, path):
    """Write ``manifest`` as JSON to ``path``, creating parent dirs."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def clip_embedding_source(model_name="ViT-B/32", device=None):
    """Return a callable ``image -> np.ndarray`` using CLIP via EmbeddingGenerator.

    Lazily imports ``vqasynth.embeddings`` so the selection algorithm and tests
    never require CLIP/torch at import time. The callable matches the output of
    ``EmbeddingGenerator.run`` (L2-normalized float32 CLIP image embedding).
    """
    from vqasynth.embeddings import EmbeddingGenerator

    generator = EmbeddingGenerator(model_name=model_name, device=device)

    def _embed(image):
        return generator.run(image)

    return _embed


def _resolve_split(dataset, split):
    """Return ``(dataset_split, split_name)`` for a Dataset or DatasetDict."""
    from datasets import DatasetDict

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"split {split!r} not found; available: {list(dataset.keys())}")
        return dataset[split], split
    return dataset, split


class Curator:
    """Curate a HuggingFace dataset down to a uniform (or random) subset.

    Embeddings are resolved in priority order:
      1. a precomputed ``embedding`` column on the dataset (written by the
         embeddings stage / ``EmbeddingGenerator.apply_transform``) — no model;
      2. an explicit ``embedding_source`` callable (image -> np.ndarray);
      3. CLIP ViT-B/32 via ``clip_embedding_source`` (default, lazy).

    Args:
        embedding_source: callable image -> np.ndarray, or None to read the
            precomputed column / fall back to CLIP.
        seed: RNG seed for selection.
        metric: distance metric for farthest-point selection.
        images: image column name used when computing embeddings on the fly.
        embedding_source_name: label recorded in the manifest when an explicit
            ``embedding_source`` is used.
    """

    def __init__(self, embedding_source=None, seed=DEFAULT_SEED, metric="euclidean",
                 images="image", embedding_source_name=None):
        if metric not in VALID_METRICS:
            raise ValueError(f"unsupported metric {metric!r}; choose from {VALID_METRICS}")
        self.embedding_source = embedding_source
        self.seed = seed
        self.metric = metric
        self.images = images
        self.embedding_source_name = embedding_source_name or "custom"

    def _collect_embeddings(self, ds):
        """Return ``(embeddings (N, D), embedding_source_label)`` for one split."""
        if EMBEDDING_COLUMN in ds.column_names:
            column = ds[EMBEDDING_COLUMN]
            if any(v is None for v in column):
                raise ValueError(
                    f"column {EMBEDDING_COLUMN!r} contains nulls; run the embeddings "
                    "stage and filter nulls before curation"
                )
            vectors = [np.asarray(v, dtype=np.float64) for v in column]
            return np.stack(vectors), f"precomputed:{EMBEDDING_COLUMN}"

        if self.embedding_source is not None:
            name = self.embedding_source_name
            source = self.embedding_source
        else:
            name = DEFAULT_EMBEDDING_SOURCE
            source = clip_embedding_source()

        vectors = []
        for row in ds:
            image = row[self.images]
            image = image[0] if isinstance(image, list) else image
            vectors.append(np.asarray(source(image), dtype=np.float64))
        if not vectors:
            raise ValueError("no embeddings could be collected from the dataset")
        return np.stack(vectors), name

    @staticmethod
    def _resolve_k(n, fraction, count):
        if fraction is None and count is None:
            raise ValueError("provide either fraction or count")
        if fraction is not None and count is not None:
            raise ValueError("provide fraction OR count, not both")
        if fraction is not None:
            if not (0.0 < fraction <= 1.0):
                raise ValueError(f"fraction must be in (0.0, 1.0]; got {fraction}")
            k = int(round(fraction * n))
        else:
            if count <= 0:
                raise ValueError(f"count must be positive; got {count}")
            k = int(count)
        if k < 1:
            raise ValueError(f"resolved k={k} (< 1); increase fraction/count")
        if k > n:
            raise ValueError(f"resolved k={k} exceeds dataset size n={n}")
        return k

    def curate(self, dataset, fraction=None, count=None, strategy="farthest",
               split="train", source_repo_id=None):
        """Select a subset of ``dataset`` and return ``(subset, manifest)``.

        Args:
            dataset: a ``datasets.Dataset`` or ``DatasetDict``.
            fraction: fraction of the split to keep (e.g. 0.25 for 25%).
            count: absolute number of samples to keep. Mutually exclusive with
                ``fraction``.
            strategy: "farthest" or "random".
            split: split name to curate when ``dataset`` is a DatasetDict.
            source_repo_id: optional source repo id recorded in the manifest.

        Returns:
            ``(subset_dataset, manifest_dict)``. The subset is a slice of the
            selected split via ``Dataset.select``.
        """
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {VALID_STRATEGIES}")

        ds, split_name = _resolve_split(dataset, split)
        embeddings, embedding_source_name = self._collect_embeddings(ds)
        n = embeddings.shape[0]
        k = self._resolve_k(n, fraction, count)

        indices = select_indices(
            embeddings, k, strategy=strategy, seed=self.seed, metric=self.metric
        )
        subset = ds.select([int(i) for i in indices])
        manifest = build_manifest(
            source_repo_id=source_repo_id,
            split=split_name,
            source_count=n,
            selected_ids=indices,
            k=k,
            strategy=strategy,
            seed=self.seed,
            metric=self.metric,
            embedding_source=embedding_source_name,
            fraction=fraction,
        )
        return subset, manifest
