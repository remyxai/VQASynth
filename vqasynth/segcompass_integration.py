"""SegCompass integration — interpretable CoT-to-mask alignment scaffold.

Scaffolds the experiment proposed for SegCompass
(https://arxiv.org/abs/2605.22658v1) against VQASynth's existing pipeline:

  "Take a few generated CoT reasoning traces from VQASynth's pipeline.
   Investigate how SegCompass's SAE-driven alignment mechanism could map
   these textual CoT concepts to SAM2-generated visual masks. Analyze
   whether the alignment reveals inconsistencies between the CoT and the
   visual segmentation, suggesting ways to improve VQASynth's CoT
   generation for better visual grounding and segmentation quality."

What this module ships today (concrete, no external checkpoints required):
  - SegCompassConfig: paper-aligned hyperparameter container
  - Concept extraction from CoT traces (stopwords + bigrams)
  - Sparse concept vectorization and cosine similarity
  - Jaccard / mask IoU primitives
  - CoT-concept to SAM2-caption alignment scoring
  - Inconsistency detection (CoT-only / mask-only / shared concepts)
  - SegCompassAligner: top-level entry point combining the above

What is intentionally NOT implemented (requires checkpoints from the
paper authors and a training pipeline that does not yet exist in this
repo):
  - The neural SAE encoder over CoT and visual tokens
  - The learned query codebook
  - The slot-mapper that lifts sparse concepts to a multi-slot heatmap
  - The mask decoder that consumes the heatmap
  - The joint RL + segmentation training loop

`SegCompassAligner` runs in "no-checkpoint" mode by default, using
lexical concept overlap as a stand-in for the SAE's learned concept
space. This is enough to surface concept-level inconsistencies between
existing CoT traces (vqasynth.r1_reasoning) and existing SAM2 masks
(vqasynth.localize) without depending on weights that aren't released.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence


_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
    "from", "by", "for", "with", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "there", "here",
    "it", "its", "i", "we", "you", "they", "them", "he", "she", "his",
    "her", "their", "our", "your", "my", "do", "does", "did", "done",
    "have", "has", "had", "will", "would", "could", "should", "can",
    "may", "might", "must", "shall", "if", "then", "than", "so",
    "because", "while", "when", "where", "what", "which", "who", "whom",
    "whose", "how", "why", "not", "no", "yes", "also", "very", "just",
    "only", "more", "most", "less", "least", "some", "any", "all",
    "each", "every", "both", "either", "neither", "above", "below",
    "between", "into", "onto", "over", "under", "through", "across",
    "image", "scene", "picture",
})

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


@dataclass
class SegCompassConfig:
    """Scaffold hyperparameters for SegCompass alignment.

    Defaults follow common SAE-on-VLM setups; tune to the actual checkpoint
    when wiring up `SegCompassAligner` with a real SAE.
    """

    sae_dim: int = 16384
    sae_top_k: int = 64
    codebook_size: int = 512
    num_slots: int = 8
    heatmap_resolution: int = 64
    rl_loss_weight: float = 1.0
    seg_loss_weight: float = 1.0
    concept_min_token_len: int = 3
    include_bigrams: bool = True
    stopwords: frozenset = field(default_factory=lambda: _STOPWORDS)


def strip_reasoning_tags(text: str) -> str:
    """Return the concatenation of <think> blocks if present, else the text.

    VQASynth's CoT outputs wrap reasoning in <think>...</think>; concept
    extraction should focus on that section, not the final <answer>.
    """
    if not text:
        return ""
    thinks = _THINK_RE.findall(text)
    if thinks:
        return " ".join(t.strip() for t in thinks)
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (letters, hyphens, apostrophes); no filtering."""
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def extract_concepts_from_cot(
    text: str,
    stopwords: Iterable[str] = _STOPWORDS,
    min_token_len: int = 3,
    include_bigrams: bool = True,
) -> list[str]:
    """Extract candidate object/concept phrases from a CoT trace.

    A lightweight stand-in for the paper's SAE concept readout. Lowercased,
    stopwords removed, optional bigrams to capture multi-word objects
    ("red forklift"). Order is preserved and duplicates are removed.
    """
    if not text:
        return []
    sw = set(stopwords)
    body = strip_reasoning_tags(text)
    toks = [
        t for t in tokenize(body)
        if len(t) >= min_token_len and t not in sw
    ]
    seen: set[str] = set()
    out: list[str] = []
    for tok in toks:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    if include_bigrams:
        for a, b in zip(toks, toks[1:]):
            bg = f"{a} {b}"
            if bg not in seen:
                seen.add(bg)
                out.append(bg)
    return out


def build_vocabulary(concepts: Iterable[str]) -> dict[str, int]:
    """Assign a stable integer index to each unique concept (insertion order)."""
    vocab: dict[str, int] = {}
    for c in concepts:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def concepts_to_sparse_vector(
    concepts: Iterable[str], vocabulary: dict[str, int]
) -> dict[int, float]:
    """Encode a list of concepts as a sparse vector {idx: count}.

    Concepts absent from `vocabulary` are dropped — mirroring what the
    paper's SAE encoder does at inference: project into the shared concept
    space and read off only the active dimensions.
    """
    vec: dict[int, float] = {}
    for c in concepts:
        idx = vocabulary.get(c)
        if idx is None:
            continue
        vec[idx] = vec.get(idx, 0.0) + 1.0
    return vec


def cosine_sparse(v1: dict[int, float], v2: dict[int, float]) -> float:
    """Cosine similarity between two sparse {idx: weight} vectors."""
    if not v1 or not v2:
        return 0.0
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    dot = 0.0
    for k, val in v1.items():
        other = v2.get(k)
        if other is not None:
            dot += val * other
    n1 = math.sqrt(sum(v * v for v in v1.values()))
    n2 = math.sqrt(sum(v * v for v in v2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def jaccard(set_a: Iterable, set_b: Iterable) -> float:
    """Jaccard similarity between two iterables of hashables."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def mask_iou(
    mask_a: Sequence[Sequence], mask_b: Sequence[Sequence]
) -> float:
    """IoU between two binary 2D masks (lists of lists, or any 2D sequence).

    Non-zero entries are treated as positive. Both masks must share shape.
    Works on pure-Python lists so it's testable without numpy; in production
    callers should pass numpy arrays through `mask_iou_array` for speed.
    """
    if len(mask_a) != len(mask_b):
        raise ValueError("Mask height mismatch")
    inter = 0
    union = 0
    for row_a, row_b in zip(mask_a, mask_b):
        if len(row_a) != len(row_b):
            raise ValueError("Mask width mismatch")
        for a, b in zip(row_a, row_b):
            pa = bool(a)
            pb = bool(b)
            if pa and pb:
                inter += 1
                union += 1
            elif pa or pb:
                union += 1
    if union == 0:
        return 0.0
    return inter / union


def align_concepts_to_captions(
    concepts: Sequence[str], captions: Sequence[str]
) -> list[tuple[int, float]]:
    """For each concept, return (best_caption_idx, jaccard_score).

    Captions in VQASynth come from `vqasynth.localize.Localizer.run()` —
    one caption per SAM2 mask. This produces a soft assignment from CoT
    concepts to masks via their captions; concepts with no token overlap
    against any caption map to (-1, 0.0).
    """
    caption_tokens = [
        {t for t in tokenize(c) if len(t) >= 2 and t not in _STOPWORDS}
        for c in captions
    ]
    out: list[tuple[int, float]] = []
    for concept in concepts:
        concept_toks = {
            t for t in tokenize(concept)
            if len(t) >= 2 and t not in _STOPWORDS
        }
        if not concept_toks:
            out.append((-1, 0.0))
            continue
        best_idx, best_score = -1, 0.0
        for i, ctoks in enumerate(caption_tokens):
            score = jaccard(concept_toks, ctoks)
            if score > best_score:
                best_idx, best_score = i, score
        out.append((best_idx, best_score))
    return out


def detect_alignment_inconsistencies(
    cot_concepts: Sequence[str], caption_concepts: Sequence[str]
) -> dict[str, list[str]]:
    """Partition concept tokens into CoT-only / caption-only / shared.

    A human reviewer can scan `cot_only` for CoT hallucinations (objects
    discussed in reasoning but not segmented), and `caption_only` for
    objects the segmentor surfaced that the CoT never used.
    """
    cot_tokens: set[str] = set()
    for c in cot_concepts:
        cot_tokens.update(tokenize(c))
    cap_tokens: set[str] = set()
    for c in caption_concepts:
        cap_tokens.update(tokenize(c))
    cot_tokens = {t for t in cot_tokens if len(t) >= 2 and t not in _STOPWORDS}
    cap_tokens = {t for t in cap_tokens if len(t) >= 2 and t not in _STOPWORDS}
    return {
        "cot_only": sorted(cot_tokens - cap_tokens),
        "caption_only": sorted(cap_tokens - cot_tokens),
        "shared": sorted(cot_tokens & cap_tokens),
    }


class SegCompassAligner:
    """CoT-to-mask alignment scaffold from the SegCompass paper.

    In "no-checkpoint" mode (the default), alignment uses the lexical
    primitives above — a usable approximation for inspecting existing
    VQASynth outputs without external dependencies.

    To wire up the paper's full method, a future PR would:
      1. Load the trained SAE encoder + query codebook + slot mapper +
         mask decoder from `checkpoint_path` (HF repo or local).
      2. Replace `encode_text_concepts` and `encode_visual_concepts`
         with calls into the SAE encoder over CoT tokens and SAM2 vision
         features respectively.
      3. Replace `align` with a forward pass that produces a multi-slot
         heatmap per concept and refines SAM2 masks with the decoder.
    These are flagged as TODO below to keep the scope of this scaffold
    honest.
    """

    def __init__(
        self,
        config: SegCompassConfig | None = None,
        checkpoint_path: str | None = None,
    ):
        self.config = config or SegCompassConfig()
        self.checkpoint_path = checkpoint_path
        self._sae = None
        self._codebook = None
        self._slot_mapper = None
        self._mask_decoder = None
        if checkpoint_path is not None:
            # TODO: load SAE encoder, query codebook, slot mapper, and mask
            # decoder from `checkpoint_path`. The paper's GitHub
            # (ZhenyuLU-Heliodore/SegCompass) is the source of truth for
            # the weight layout; until that repo is depended on directly,
            # downstream calls fall back to the lexical path.
            pass

    @property
    def has_checkpoint(self) -> bool:
        return self._sae is not None

    def encode_text_concepts(self, cot_text: str) -> list[str]:
        """CoT trace -> concept list. Lexical fallback when has_checkpoint is False."""
        return extract_concepts_from_cot(
            cot_text,
            stopwords=self.config.stopwords,
            min_token_len=self.config.concept_min_token_len,
            include_bigrams=self.config.include_bigrams,
        )

    def encode_visual_concepts(self, captions: Sequence[str]) -> list[str]:
        """Per-mask captions -> concept list. Lexical fallback when has_checkpoint is False."""
        concepts: list[str] = []
        seen: set[str] = set()
        for cap in captions:
            for tok in extract_concepts_from_cot(
                cap,
                stopwords=self.config.stopwords,
                min_token_len=self.config.concept_min_token_len,
                include_bigrams=False,
            ):
                if tok not in seen:
                    seen.add(tok)
                    concepts.append(tok)
        return concepts

    def align(
        self,
        cot_text: str,
        captions: Sequence[str],
        masks: Sequence | None = None,
    ) -> dict:
        """Align one CoT trace against a list of per-mask captions.

        Returns:
          {
            'concepts':            CoT-derived concept list,
            'concept_to_caption':  [(caption_idx, score), ...] per concept,
            'inconsistencies':     {'cot_only': [...], 'caption_only': [...], 'shared': [...]},
            'num_masks':           len(masks) or None,
          }
        """
        cot_concepts = self.encode_text_concepts(cot_text)
        cap_concepts = self.encode_visual_concepts(captions)
        return {
            "concepts": cot_concepts,
            "concept_to_caption": align_concepts_to_captions(cot_concepts, captions),
            "inconsistencies": detect_alignment_inconsistencies(cot_concepts, cap_concepts),
            "num_masks": (len(masks) if masks is not None else None),
        }
