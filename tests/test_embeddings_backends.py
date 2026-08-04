"""Smoke tests for vqasynth.embeddings pluggable backends.

Verifies backend dispatch + the EmbeddingGenerator / TagFilter contracts against
a minimal fake backend — no CLIP install, no transformers download, no CUDA.
Real end-to-end embedding quality belongs on a GPU host (same stance as
tests/test_vggt_speedups.py).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from PIL import Image

from vqasynth.embeddings import (
    CLIPBackend,
    EmbeddingBackend,
    EmbeddingGenerator,
    MultiModalEmbeddingModel,
    TagFilter,
    TransformersBackend,
    build_embedding_backend,
    register_embedding_backend,
)


class _FakeBackend(EmbeddingBackend):
    """Deterministic fake backend: fixed-dim features, no model weights.

    tokenize -> (N, L) long tensor; encode_text -> (N, dim) one-hot of
    (sum(token_ids) % dim); preprocess -> single float per image; encode_image
    -> (1, dim) one-hot of (int(scalar) % dim). Enough to exercise the
    EmbeddingGenerator / TagFilter plumbing end to end.
    """

    name = "fake"
    dim = 4

    def __init__(self, model_name="fake", device=None):
        super().__init__(device=device)
        self.image_calls = 0
        self.text_calls = 0

    def preprocess(self, image):
        return torch.tensor([float(np.asarray(image).sum())])

    def encode_image(self, image_input):
        self.image_calls += 1
        scalar = int(image_input.flatten()[0].item())
        feat = torch.zeros(self.dim)
        feat[scalar % self.dim] = 1.0
        return feat.unsqueeze(0).to(self.device)

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for t in texts:
            ids = [ord(c) for c in t]
            ids = (ids + [0] * 8)[:8]
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long)

    def encode_text(self, text_input):
        self.text_calls += 1
        t = text_input.to(self.device).float()
        sums = t.sum(dim=-1)
        feats = torch.zeros(t.shape[0], self.dim, device=self.device)
        for i, s in enumerate(sums.tolist()):
            feats[i, int(s) % self.dim] = 1.0
        return feats


# Register once so name-based resolution works in the tests below.
register_embedding_backend("fake", _FakeBackend)


def _img(seed=0):
    arr = np.full((4, 4, 3), seed, dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


def test_clip_import_is_lazy():
    """Importing the module must NOT pull in the optional `clip` package."""
    import vqasynth.embeddings as emb_mod

    assert not hasattr(emb_mod, "clip"), (
        "clip should be imported lazily inside CLIPBackend so the module is "
        "importable without the CLIP package installed"
    )


def test_default_backend_is_clip():
    sig = inspect.signature(MultiModalEmbeddingModel.__init__)
    assert sig.parameters["backend"].default == "clip"


def test_builtin_backends_registered():
    """The built-in backends map to the right classes (without constructing them,
    which would require their optional packages / checkpoints)."""
    import vqasynth.embeddings as emb

    assert emb._EMBEDDING_BACKENDS["clip"] is CLIPBackend
    assert emb._EMBEDDING_BACKENDS["transformers"] is TransformersBackend
    # "fake" was registered at this test module's import.
    assert emb._EMBEDDING_BACKENDS["fake"] is _FakeBackend


def test_register_and_resolve_named_backend():
    backend = build_embedding_backend("fake", device="cpu")
    assert isinstance(backend, _FakeBackend)


def test_accepts_backend_class_and_instance():
    by_class = build_embedding_backend(_FakeBackend, device="cpu")
    assert isinstance(by_class, _FakeBackend)
    instance = _FakeBackend(device="cpu")
    by_instance = build_embedding_backend(instance)
    assert by_instance is instance


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        build_embedding_backend("definitely-not-a-backend", device="cpu")


def test_register_rejects_non_backend():
    with pytest.raises(TypeError, match="EmbeddingBackend"):
        register_embedding_backend("bogus", object)


def test_embedding_generator_runs_through_backend():
    """End-to-end through the real EmbeddingGenerator.run via a fake backend."""
    gen = EmbeddingGenerator(backend=_FakeBackend, device="cpu")
    assert gen.backend_name == "fake"

    out = gen.run(_img(seed=3))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, _FakeBackend.dim)
    assert out.dtype == np.float32
    # L2-normalized
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5
    assert gen._backend.image_calls == 1


def test_embedding_generator_apply_transform_batched():
    gen = EmbeddingGenerator(backend=_FakeBackend, device="cpu")
    batch = {"image": [_img(seed=1), _img(seed=2), _img(seed=3)]}
    result = gen.apply_transform(batch, images="image")
    assert "embedding" in result
    assert len(result["embedding"]) == 3
    assert all(e is not None for e in result["embedding"])
    assert result["embedding"][0].shape == (1, _FakeBackend.dim)


def test_tag_filter_picks_a_registered_tag():
    """End-to-end through the real TagFilter.get_best_matching_tag."""
    tf = TagFilter(backend=_FakeBackend, device="cpu")
    img_emb = np.zeros((1, _FakeBackend.dim), dtype=np.float32)
    img_emb[0, 2] = 1.0

    tags = ["cat", "dog", "bird"]
    best = tf.get_best_matching_tag(img_emb, tags)
    assert best in tags
    assert tf._backend.text_calls == 1


def test_tag_filter_apply_transform_handles_nulls():
    tf = TagFilter(backend=_FakeBackend, device="cpu")
    batch = {"embedding": [None, np.zeros((1, _FakeBackend.dim), dtype=np.float32)]}
    result = tf.apply_transform(batch, tags=["cat", "dog"])
    assert result["tag"][0] is None
    assert result["tag"][1] in ["cat", "dog"]


def test_clip_backend_preserves_uniform_interface():
    """CLIPBackend must expose the same four primitives as every backend."""
    for method in ("preprocess", "encode_image", "encode_text", "tokenize"):
        assert callable(getattr(CLIPBackend, method)), method
    for method in ("preprocess", "encode_image", "encode_text", "tokenize"):
        assert callable(getattr(TransformersBackend, method)), method
