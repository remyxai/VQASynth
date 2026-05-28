"""Tests for vqasynth.segcompass_integration.

Pure-Python tests — no numpy / torch / spacy needed. The integration
module exposes lexical concept primitives that the orchestrator can run
in the project's minimal CI env.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vqasynth.segcompass_integration import (
    SegCompassAligner,
    SegCompassConfig,
    align_concepts_to_captions,
    build_vocabulary,
    concepts_to_sparse_vector,
    cosine_sparse,
    detect_alignment_inconsistencies,
    extract_concepts_from_cot,
    jaccard,
    mask_iou,
    strip_reasoning_tags,
    tokenize,
)


# ---------- text utilities ----------

def test_strip_reasoning_tags_with_think_block():
    text = "<think>The forklift is red.</think><answer>Yes.</answer>"
    assert strip_reasoning_tags(text).strip() == "The forklift is red."


def test_strip_reasoning_tags_multiple_blocks_concatenated():
    text = "<think>one</think> filler <think>two</think>"
    out = strip_reasoning_tags(text)
    assert "one" in out and "two" in out


def test_strip_reasoning_tags_no_tag_returns_input():
    assert strip_reasoning_tags("plain text").strip() == "plain text"


def test_strip_reasoning_tags_empty():
    assert strip_reasoning_tags("") == ""
    assert strip_reasoning_tags(None) == ""


def test_tokenize_basic():
    assert tokenize("Red Forklift, near boxes!") == ["red", "forklift", "near", "boxes"]


def test_tokenize_keeps_hyphens_and_apostrophes():
    assert tokenize("low-light scene; can't") == ["low-light", "scene", "can't"]


def test_tokenize_empty_inputs():
    assert tokenize("") == []
    assert tokenize(None) == []


# ---------- concept extraction ----------

def test_extract_concepts_drops_stopwords_and_short_tokens():
    text = "The red forklift is on the floor."
    concepts = extract_concepts_from_cot(text, include_bigrams=False)
    assert "red" in concepts
    assert "forklift" in concepts
    assert "floor" in concepts
    assert "the" not in concepts
    assert "is" not in concepts


def test_extract_concepts_includes_bigrams_when_requested():
    text = "A red forklift moves boxes."
    concepts = extract_concepts_from_cot(text, include_bigrams=True)
    assert "red forklift" in concepts


def test_extract_concepts_preserves_order_and_dedups():
    text = "Forklift. Forklift. Forklift."
    concepts = extract_concepts_from_cot(text, include_bigrams=False)
    assert concepts == ["forklift"]


def test_extract_concepts_from_think_block_only():
    text = "<think>The pallet is empty.</think><answer>Yes.</answer>"
    concepts = extract_concepts_from_cot(text, include_bigrams=False)
    assert "pallet" in concepts
    assert "empty" in concepts


def test_extract_concepts_empty_input():
    assert extract_concepts_from_cot("") == []
    assert extract_concepts_from_cot(None) == []


# ---------- sparse vector primitives ----------

def test_build_vocabulary_assigns_stable_ids():
    vocab = build_vocabulary(["a", "b", "a", "c"])
    assert vocab == {"a": 0, "b": 1, "c": 2}


def test_concepts_to_sparse_vector_drops_oov():
    vocab = {"forklift": 0, "pallet": 1}
    v = concepts_to_sparse_vector(["forklift", "ladder", "forklift"], vocab)
    assert v == {0: 2.0}


def test_cosine_sparse_identical_is_one():
    v = {0: 1.0, 1: 2.0, 3: 4.0}
    assert cosine_sparse(v, dict(v)) == pytest.approx(1.0)


def test_cosine_sparse_disjoint_is_zero():
    assert cosine_sparse({0: 1.0}, {1: 1.0}) == 0.0


def test_cosine_sparse_partial_overlap():
    a = {0: 1.0, 1: 1.0}
    b = {1: 1.0, 2: 1.0}
    # dot=1, |a|=|b|=sqrt(2), cos=1/2
    assert cosine_sparse(a, b) == pytest.approx(0.5)


def test_cosine_sparse_handles_empty():
    assert cosine_sparse({}, {0: 1.0}) == 0.0
    assert cosine_sparse({0: 1.0}, {}) == 0.0
    assert cosine_sparse({}, {}) == 0.0


# ---------- set + mask primitives ----------

def test_jaccard_full_overlap():
    assert jaccard({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)


def test_jaccard_no_overlap():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_jaccard_two_empty_is_zero():
    assert jaccard([], []) == 0.0


def test_mask_iou_identical():
    m = [[1, 0], [1, 1]]
    assert mask_iou(m, [row[:] for row in m]) == pytest.approx(1.0)


def test_mask_iou_disjoint():
    a = [[1, 0], [0, 0]]
    b = [[0, 1], [0, 0]]
    assert mask_iou(a, b) == 0.0


def test_mask_iou_partial():
    a = [[1, 1], [0, 0]]
    b = [[1, 0], [1, 0]]
    # inter = 1 (pos (0,0)); union = 3 (pos (0,0),(0,1),(1,0))
    assert mask_iou(a, b) == pytest.approx(1 / 3)


def test_mask_iou_all_zero_is_zero():
    z = [[0, 0], [0, 0]]
    assert mask_iou(z, z) == 0.0


def test_mask_iou_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mask_iou([[1]], [[1, 0]])
    with pytest.raises(ValueError):
        mask_iou([[1]], [[1], [0]])


# ---------- alignment ----------

def test_align_concepts_to_captions_matches_best_caption():
    concepts = ["forklift", "pallet", "ladder"]
    captions = ["a red forklift", "wooden pallet with boxes"]
    out = align_concepts_to_captions(concepts, captions)
    assert out[0][0] == 0
    assert out[1][0] == 1
    assert out[2] == (-1, 0.0)
    assert out[0][1] > 0
    assert out[1][1] > 0


def test_align_concepts_to_captions_empty_concept_list():
    assert align_concepts_to_captions([], ["a"]) == []


def test_align_concepts_to_captions_empty_captions():
    out = align_concepts_to_captions(["forklift"], [])
    assert out == [(-1, 0.0)]


def test_detect_alignment_inconsistencies_partitions_correctly():
    cot = ["forklift", "pallet", "ladder"]
    captions = ["forklift", "pallet", "shelf"]
    parts = detect_alignment_inconsistencies(cot, captions)
    assert "ladder" in parts["cot_only"]
    assert "shelf" in parts["caption_only"]
    assert "forklift" in parts["shared"]
    assert "pallet" in parts["shared"]
    # Sets are disjoint.
    assert set(parts["cot_only"]).isdisjoint(parts["caption_only"])
    assert set(parts["cot_only"]).isdisjoint(parts["shared"])
    assert set(parts["caption_only"]).isdisjoint(parts["shared"])


# ---------- config + aligner ----------

def test_segcompass_config_defaults_present():
    cfg = SegCompassConfig()
    assert cfg.sae_dim > 0
    assert cfg.sae_top_k > 0
    assert cfg.codebook_size > 0
    assert cfg.num_slots > 0
    assert cfg.heatmap_resolution > 0
    assert 0.0 < cfg.rl_loss_weight
    assert 0.0 < cfg.seg_loss_weight
    assert cfg.concept_min_token_len >= 2
    assert "the" in cfg.stopwords


def test_segcompass_config_overrides():
    cfg = SegCompassConfig(num_slots=4, sae_top_k=16)
    assert cfg.num_slots == 4
    assert cfg.sae_top_k == 16


def test_aligner_no_checkpoint_smoke():
    aligner = SegCompassAligner()
    assert aligner.has_checkpoint is False
    assert isinstance(aligner.config, SegCompassConfig)


def test_aligner_encode_text_concepts_uses_lexical_fallback():
    aligner = SegCompassAligner()
    concepts = aligner.encode_text_concepts(
        "<think>A red forklift lifts a pallet.</think>"
    )
    assert "forklift" in concepts
    assert "pallet" in concepts


def test_aligner_encode_visual_concepts_dedups_across_captions():
    aligner = SegCompassAligner()
    out = aligner.encode_visual_concepts(["red forklift", "another forklift"])
    assert out.count("forklift") == 1


def test_aligner_align_endtoend_returns_expected_shape():
    aligner = SegCompassAligner()
    cot = "<think>The red forklift is near the wooden pallet.</think>"
    captions = ["red forklift", "wooden pallet", "yellow ladder"]
    masks = [[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [0, 1]]]
    result = aligner.align(cot, captions, masks=masks)
    assert "concepts" in result
    assert "concept_to_caption" in result
    assert "inconsistencies" in result
    assert result["num_masks"] == 3
    assert len(result["concept_to_caption"]) == len(result["concepts"])
    # "ladder" was in a caption but not in CoT -> caption_only should include it.
    assert "ladder" in result["inconsistencies"]["caption_only"]


def test_aligner_align_handles_no_masks():
    aligner = SegCompassAligner()
    result = aligner.align("<think>pallet</think>", ["pallet"])
    assert result["num_masks"] is None
    assert "pallet" in result["inconsistencies"]["shared"]
