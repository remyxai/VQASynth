"""Smoke tests for vqasynth.embeddings backend plumbing (issue #33).

Exercises the backend registry, dispatch, and the normalization / tag-ranking
logic against a registered *fake* backend — no OpenAI ``clip`` install, no
``transformers`` download, no CUDA. Real end-to-end embedding quality belongs
on a GPU host with the actual checkpoints.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from PIL import Image

from vqasynth.embeddings import (
    EmbeddingGenerator,
    EmbeddingBackend,
    MultiModalEmbeddingModel,
    TagFilter,
    list_embedding_backends,
    register_embedding_backend,
)


@register_embedding_backend("fake_test")
class _FakeBackend(EmbeddingBackend):
    """Deterministic orthogonal embedding space for ranking assertions.

    ``encode_text`` maps tag ``i`` to basis vector ``e_i``; ``encode_image``
    maps to ``e_{image_index}``. The closest tag to an image is therefore the
    one whose index equals ``image_index``.
    """

    DIM = 8

    def __init__(self, model_name=None, device=None, image_index=0, **_):
        super().__init__(model_name=model_name, device=device)
        self.image_index = image_index

    def encode_image(self, image):
        v = torch.zeros(self.DIM)
        v[self.image_index] = 1.0
        return v.unsqueeze(0)  # (1, D)

    def encode_text(self, tags):
        out = torch.zeros(len(tags), self.DIM)
        for i in range(len(tags)):
            out[i, i % self.DIM] = 1.0
        return out  # (len(tags), D)


def _rgb_image():
    return Image.new("RGB", (4, 4), color=(0, 0, 0))


def test_clip_import_is_lazy():
    """Importing vqasynth.embeddings must not require the `clip` package.

    The OpenAI CLIP backend imports `clip` inside its constructor, so merely
    importing the module (which we did above) proves the top-level import was
    removed. Belt-and-braces: assert `clip` was never pulled into sys.modules.
    """
    assert "clip" not in sys.modules


def test_default_backend_is_clip():
    assert "clip" in list_embedding_backends()
    assert "transformers" in list_embedding_backends()
    # The default backend name on the base class resolves to clip without
    # instantiating it (which would need the clip package).
    import inspect

    sig = inspect.signature(MultiModalEmbeddingModel.__init__)
    assert sig.parameters["backend"].default == "clip"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        MultiModalEmbeddingModel(backend="does-not-exist")


def test_embedding_generator_run_returns_normalized_float32():
    gen = EmbeddingGenerator(backend="fake_test", image_index=2)
    emb = gen.run(_rgb_image())

    assert isinstance(emb, np.ndarray)
    assert emb.dtype == np.float32
    assert emb.shape == (1, _FakeBackend.DIM)
    # L2-normalized
    np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-6)
    # Matches the configured basis vector.
    np.testing.assert_allclose(emb[0, 2], 1.0, atol=1e-6)


def test_tag_filter_ranks_closest_tag():
    tags = ["cat", "dog", "car", "tree"]
    tag_filter = TagFilter(backend="fake_test", image_index=2)

    # Build an image embedding in the same fake space (e_2) and rank tags.
    img_emb = np.zeros((1, _FakeBackend.DIM), dtype=np.float32)
    img_emb[0, 2] = 1.0

    assert tag_filter.get_best_matching_tag(img_emb, tags) == "car"

    # Different image_index -> different winner (proves ranking isn't hardcoded).
    tag_filter_0 = TagFilter(backend="fake_test", image_index=0)
    img_emb_0 = np.zeros((1, _FakeBackend.DIM), dtype=np.float32)
    img_emb_0[0, 0] = 1.0
    assert tag_filter_0.get_best_matching_tag(img_emb_0, tags) == "cat"


def test_filter_by_tag_include_exclude_logic():
    tag_filter = TagFilter(backend="fake_test")

    assert tag_filter.filter_by_tag("cat", include_tags=["cat", "dog"], exclude_tags=[]) is True
    assert tag_filter.filter_by_tag("car", include_tags=["cat", "dog"], exclude_tags=[]) is False
    assert tag_filter.filter_by_tag("cat", include_tags=[], exclude_tags=["cat"]) is False
    # Neither list active -> everything passes.
    assert tag_filter.filter_by_tag("anything", include_tags=[], exclude_tags=[]) is True


def test_tag_filter_apply_transform_single_row():
    tag_filter = TagFilter(backend="fake_test", image_index=1)
    img_emb = np.zeros((1, _FakeBackend.DIM), dtype=np.float32)
    img_emb[0, 1] = 1.0

    example = {"embedding": img_emb}
    out = tag_filter.apply_transform(example, tags=["a", "b", "c"])
    assert out["tag"] == "b"


def test_backend_kwargs_forwarded():
    """Extra kwargs flow through MultiModalEmbeddingModel to the backend."""
    # The factory path forwards image_index via **backend_kwargs:
    model = MultiModalEmbeddingModel(backend="fake_test", image_index=3)
    assert model._backend.image_index == 3
