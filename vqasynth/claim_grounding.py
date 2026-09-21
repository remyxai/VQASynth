"""
Claim Grounding — semantic-spatial agreement verification for object claims.

Adapted from "Semantic-Spatial Agreement Verification for Mitigating Object
Hallucination in Multimodal Large Language Models" (arXiv:2609.17269). A VLM
reasoning trace (e.g. the ``output`` column written by the ``r1_reasoning``
stage) may mention objects the image does not contain. SSAV's training-free
check: a visually grounded claim should (a) be *semantically supported* —
paraphrased pointing queries about the object keep finding *something* — and
(b) show *spatial agreement* — those queries repeatedly localize to the *same*
image region rather than dispersing or firing once in isolation.

The paper's two evidence branches are kept at full fidelity:

  * semantic support  — the share of paraphrased queries that localize the
                       claim at all. Aggregating over paraphrases makes the
                       signal robust to any single prompt's wording. (The
                       reference SSAV averages per-query detection *scores*;
                       the localize stage emits Molmo ``<point>`` outputs that
                       carry no confidence, so this branch uses the
                       localized-query *fraction* as the point-native analog.);
  * QIRV              — Query-Induced Regional Verification over the per-query
                       localizations, combining region *persistence* (how many
                       distinct queries agree on one region), spatial *overlap*
                       (how many distinct-query pairs land within one region),
                       and relative *candidate dominance* (how far the modal
                       region stands above the runner-up) as their **product**
                       — the AND-semantics of the reference's
                       ``persistence * spatial_agreement * candidate_dominance``,
                       so dispersed localizations and isolated single-query
                       responses collapse the branch;
  * fusion            — the geometric mean of the two branches, so a claim
                       scores high only when BOTH semantic and spatial
                       evidence support it.

This is an ADAPTED PORT (Mode 2). Substituted target-natively: the paper's
MLLM-inference + prompt-aggregation harness is replaced by the pointing-query
JSONL split used by ``experiments/visual_credit_audit`` — this module scores
localizations the caller collected, parsed through the existing
``vqasynth.localize.extract_points_and_descriptions`` contract (Molmo ``<point>``
tags, pixel coordinates). Object-claim extraction is a parameter-free
article-phrase heuristic matched against the pipeline's own ``captions``
inventory (the localize stage's detected objects) instead of any tagging model.
The paper's POPE / CHAIR evaluation protocols are intentionally out of scope:
this is the verification signal for flagging synthetic CoT, not an evaluation
framework.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from itertools import combinations

# Radius, as a fraction of the normalized image frame, within which two
# localized points count as the SAME region.
DEFAULT_RADIUS = 0.15

# Verification score at/above which a claim counts as grounded.
DEFAULT_THRESHOLD = 0.5

# Paraphrased pointing queries per claim. Semantic support is estimated over
# ALL of them, so no single wording can pass or sink a claim on its own.
POINTING_QUERIES = (
    "Point to the {claim}.",
    "Where is the {claim} in the image?",
    "Click on the {claim}.",
    "Locate the {claim} and mark it with a point.",
    "Identify the {claim} in the image and point to it.",
)

# ---------------------------------------------------------------------------
# Object-claim extraction (parameter-free substitute for a tagging model)
# ---------------------------------------------------------------------------

_ARTICLES = {"the", "a", "an"}
# Words that never extend an object phrase ("the man in the red hat" -> runs
# "man" and "red hat", not "man in the red hat").
_FILLER = _ARTICLES | {
    "this", "that", "these", "those", "of", "in", "on", "at", "to", "and",
    "or", "is", "are", "was", "were", "it", "its", "with", "for", "from",
    "by", "as", "him", "his", "her", "hers", "them", "their", "our", "my",
    "your", "you", "we", "us", "between", "into", "onto", "over", "under",
    "near", "behind", "above", "below", "beside", "across", "around",
}
# Phrase heads that name scene geometry or reasoning bookkeeping rather than
# an object ("the left side", "the distance", "the image").
_NON_OBJECT_HEADS = {
    "left", "right", "front", "back", "rear", "top", "bottom", "side",
    "middle", "center", "centre", "corner", "edge", "distance", "gap",
    "height", "width", "depth", "length", "size", "amount", "number",
    "position", "location", "perspective", "viewpoint", "viewer", "image",
    "picture", "photo", "scene", "frame", "background", "foreground",
    "question", "answer", "estimate", "estimation", "measurement", "unit",
    "meter", "meters", "foot", "feet", "inch", "inches", "centimeter",
    "centimeters", "way", "time", "point", "points", "end", "line", "row",
    "column", "grid", "two", "three", "several", "other", "same", "next",
    "first", "second", "total", "pair", "arrangement", "layout", "structure",
    "setup", "combination", "difference", "relationship",
}
# Longest run of modifier + head tokens accepted in one claim.
_MAX_CLAIM_WORDS = 4

_WORD_RE = re.compile(r"[A-Za-z]+")


def _singularize(token):
    """Cheap plural fold so "boxes" can match a caption's "box"."""
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_tokens(text):
    """Lowercased content tokens (articles/prepositions dropped, plurals folded)."""
    return [
        _singularize(tok)
        for tok in _WORD_RE.findall((text or "").lower())
        if tok not in _FILLER
    ]


def claim_in_inventory(claim, vocabulary, min_overlap=0.5):
    """
    Does the claim refer to one of the scene's detected objects?

    Uses the overlap coefficient between the claim's and each caption's
    content tokens, so a colloquial "hat" matches the caption "a man wearing
    a red hat" (the caption's tokens are a superset of the claim's).
    ``vocabulary`` is the localize stage's ``captions`` list for the image.
    """
    if not vocabulary:
        return False
    claim_tokens = set(_content_tokens(claim))
    if not claim_tokens:
        return False
    for caption in vocabulary:
        caption_tokens = set(_content_tokens(caption))
        if not caption_tokens:
            continue
        shared = len(claim_tokens & caption_tokens)
        if shared and shared / min(len(claim_tokens), len(caption_tokens)) >= min_overlap:
            return True
    return False


def extract_object_claims(text, vocabulary=None, min_overlap=0.5):
    """
    Pull candidate object mentions out of free-form reasoning text.

    For each article ("the red hat", "a pallet"), take the maximal run of
    consecutive non-filler alphabetic tokens that follows it — stopping at
    punctuation, the next filler word, or ``_MAX_CLAIM_WORDS`` — and keep it
    when its head noun is not scene-geometry/reasoning vocabulary. When
    ``vocabulary`` (the pipeline ``captions`` for the image) is given, each
    claim is marked with whether it refers to a detected object; claims that
    do NOT match are the prime hallucination suspects the pointing queries
    should test.

    Returns a list of :class:`ObjectClaim` in order of first mention.
    """
    text = text or ""
    tokens = [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    claims, seen = [], set()
    for index, (word, _, _) in enumerate(tokens):
        if word not in _ARTICLES:
            continue
        run = []
        position = index + 1
        while position < len(tokens) and len(run) < _MAX_CLAIM_WORDS:
            next_word, next_start, _ = tokens[position]
            prev_end = tokens[position - 1][2]
            if text[prev_end:next_start].strip() or next_word in _FILLER:
                break
            run.append(next_word)
            position += 1
        if not run or run[-1] in _NON_OBJECT_HEADS:
            continue
        claim = " ".join(run)
        if claim in seen:
            continue
        seen.add(claim)
        claims.append(
            ObjectClaim(
                claim=claim,
                in_inventory=claim_in_inventory(claim, vocabulary, min_overlap)
                if vocabulary
                else False,
            )
        )
    return claims


# ---------------------------------------------------------------------------
# Paraphrased pointing queries
# ---------------------------------------------------------------------------


def build_pointing_queries(claim, templates=POINTING_QUERIES):
    """
    The paraphrased pointing prompts for one claim (one per template).

    These are what the pointing VLM (Molmo in this pipeline) answers with
    ``<point>`` tags; the collected responses are parsed back into pixel
    coordinates by ``vqasynth.localize.extract_points_and_descriptions``.
    """
    return [template.format(claim=claim) for template in templates]


# ---------------------------------------------------------------------------
# Semantic support + QIRV scoring
# ---------------------------------------------------------------------------


def _normalize_points(points, image_size=None):
    """
    Coerce one query's localized points to normalized [0, 1] frame coords.

    Pixel coordinates (the localize parser's output) are divided by
    ``image_size`` = (width, height). Without ``image_size`` the points are
    assumed normalized already; anything outside [0, 1] then raises.
    """
    normalized = []
    for x, y in points:
        if image_size is not None:
            width, height = image_size
            if width <= 0 or height <= 0:
                raise ValueError(f"image_size must be positive, got {image_size}")
            x, y = x / width, y / height
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"point ({x}, {y}) outside the [0, 1] frame — pass image_size "
                "for pixel coordinates"
            )
        normalized.append((float(x), float(y)))
    return normalized


def semantic_support(query_points):
    """
    Share of paraphrased queries that localized the claim at all.

    This is the semantic branch: an object that is really there keeps being
    *found* under reworded queries, while a hallucinated one is only found
    (if ever) under whichever wording leaked it into the trace.
    """
    if not query_points:
        return 0.0
    found = sum(1 for points in query_points if len(points) > 0)
    return found / len(query_points)


def qirv(query_points, radius=DEFAULT_RADIUS):
    """
    Query-Induced Regional Verification over per-query localizations.

    Args:
        query_points: one list of normalized [0, 1] points per paraphrased
            query (empty list when that query found nothing).
        radius: normalized-frame distance within which two points count as
            the same region.

    Returns a :class:`SpatialAgreement` with the paper's three components:

      * persistence — distinct queries agreeing on the modal region / K;
      * overlap     — distinct-query pairs whose point sets come within
                      ``radius`` of each other / all localizing pairs (this is
                      what drops for dispersed localizations);
      * dominance   — (modal support − runner-up support) / modal support,
                      where the runner-up is the strongest region outside the
                      modal one (isolated extra responses push this down).
    """
    per_query = [list(points) for points in query_points]
    total_queries = len(per_query)
    if total_queries == 0:
        return SpatialAgreement(0.0, 0.0, 0.0, 0.0, None)

    flat = [
        (query_index, x, y)
        for query_index, points in enumerate(per_query)
        for x, y in points
    ]
    if not flat:
        return SpatialAgreement(0.0, 0.0, 0.0, 0.0, None)

    def _support(center_x, center_y, candidates):
        return len(
            {
                query_index
                for query_index, x, y in candidates
                if math.hypot(x - center_x, y - center_y) <= radius
            }
        )

    # Modal region: the localization with the widest distinct-query agreement.
    modal_support, modal_center = 0, None
    for _, x, y in flat:
        support = _support(x, y, flat)
        if support > modal_support:
            modal_support, modal_center = support, (x, y)
    persistence = modal_support / total_queries

    # Overlap: agreement between distinct-query point sets, over all pairs of
    # localizing queries. Fewer than two localizing queries leaves no
    # cross-query overlap evidence.
    localizing = [i for i, points in enumerate(per_query) if points]
    pairs = list(combinations(localizing, 2))
    if pairs:
        near = sum(
            1
            for i, j in pairs
            if any(
                math.hypot(x1 - x2, y1 - y2) <= radius
                for x1, y1 in per_query[i]
                for x2, y2 in per_query[j]
            )
        )
        overlap = near / len(pairs)
    else:
        overlap = 0.0

    # Dominance: best distinct-query support outside the modal region.
    outside = [
        (query_index, x, y)
        for query_index, x, y in flat
        if math.hypot(x - modal_center[0], y - modal_center[1]) > radius
    ]
    runner_up = max((_support(x, y, outside) for _, x, y in outside), default=0)
    dominance = (modal_support - runner_up) / modal_support if modal_support else 0.0

    # Combine the three components as a PRODUCT, matching the reference SSAV's
    # ``qirv_evidence = proposal_persistence * spatial_agreement *
    # candidate_dominance`` (zihengren/SSAV, ``ssav/core.py::qirv_features``).
    # The product gives AND-semantics — any one component near zero collapses
    # the spatial evidence — which is the paper's intent (a claim is spatially
    # grounded only when it persists AND overlaps AND dominates). An average
    # would soften that, letting two strong components mask a failed one.
    score = persistence * overlap * dominance
    return SpatialAgreement(
        persistence=persistence,
        overlap=overlap,
        dominance=dominance,
        score=score,
        modal_point=modal_center,
    )


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class ObjectClaim:
    """One object mention lifted from a reasoning trace.

    ``in_inventory`` records whether the mention refers to an object the
    localize stage actually detected (the ``captions`` column); False claims
    are the ones the pointing queries should stress-test.
    """

    claim: str
    in_inventory: bool = False


@dataclass
class SpatialAgreement:
    """The QIRV branch scores for one claim."""

    persistence: float
    overlap: float
    dominance: float
    score: float                       # mean of the three components
    modal_point: tuple                 # normalized center of the modal region


@dataclass
class ClaimGroundingResult:
    """Full SSAV verdict for one claim."""

    claim: str
    in_inventory: bool
    queries_total: int
    queries_localized: int
    semantic_support: float
    persistence: float
    overlap: float
    dominance: float
    spatial_agreement: float
    score: float                       # geometric mean of the two branches
    grounded: bool


@dataclass
class GroundingReport:
    total_claims: int
    grounded_claims: int
    grounded_rate: float
    mean_score: float
    mean_semantic_support: float
    mean_spatial_agreement: float
    ungrounded: list = field(default_factory=list)
    per_claim: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_claim(
    claim,
    query_points,
    image_size=None,
    radius=DEFAULT_RADIUS,
    threshold=DEFAULT_THRESHOLD,
    in_inventory=False,
):
    """
    Verify one object claim from its per-query localizations.

    Args:
        claim: the object phrase being verified.
        query_points: one list of points per paraphrased query — pixel coords
            (the localize parser's output) when ``image_size`` is given,
            normalized [0, 1] coords otherwise. An empty list means that
            query found nothing.
        image_size: (width, height) of the image the points refer to.
        radius: normalized-frame region radius (see :func:`qirv`).
        threshold: score at/above which the claim counts as grounded.
        in_inventory: whether the claim matched the pipeline's captions.

    With fewer than two queries there is no cross-query evidence to verify
    against, so the claim is reported unverified (score 0.0, grounded False) —
    SSAV's premise is aggregation over paraphrases.
    """
    per_query = [_normalize_points(points, image_size) for points in query_points]
    semantic = semantic_support(per_query)
    spatial = qirv(per_query, radius=radius)

    if len(per_query) < 2:
        score = 0.0
    else:
        # Geometric mean: low when EITHER branch lacks support.
        score = math.sqrt(max(semantic, 0.0) * max(spatial.score, 0.0))

    return ClaimGroundingResult(
        claim=claim,
        in_inventory=in_inventory,
        queries_total=len(per_query),
        queries_localized=sum(1 for points in per_query if points),
        semantic_support=semantic,
        persistence=spatial.persistence,
        overlap=spatial.overlap,
        dominance=spatial.dominance,
        spatial_agreement=spatial.score,
        score=score,
        grounded=bool(score >= threshold),
    )


def grounding_report(results, threshold=DEFAULT_THRESHOLD):
    """
    Aggregate per-claim verification results into a :class:`GroundingReport`.

    ``ungrounded`` lists the claim strings scoring below ``threshold`` — the
    mentions a curation step would flag or drop before the trace is used as
    training data.
    """
    results = list(results)
    total = len(results)

    def _mean(values):
        return sum(values) / total if total else 0.0

    return GroundingReport(
        total_claims=total,
        grounded_claims=sum(1 for result in results if result.grounded),
        grounded_rate=_mean([float(result.grounded) for result in results]),
        mean_score=_mean([result.score for result in results]),
        mean_semantic_support=_mean([result.semantic_support for result in results]),
        mean_spatial_agreement=_mean([result.spatial_agreement for result in results]),
        ungrounded=[result.claim for result in results if not result.grounded],
        per_claim=results,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_grounding_report(report):
    """
    Format a :class:`GroundingReport` as a human-readable string.

    Mirrors the layout of ``vqasynth.visual_credit.format_credit_report``.
    """
    lines = [
        "",
        "=" * 70,
        "CLAIM GROUNDING (SEMANTIC-SPATIAL AGREEMENT)",
        "=" * 70,
        "",
        f"  ({report.total_claims} object claims verified)",
        "",
        f"  {'Metric':<34} {'Value':>12}",
        f"  {'-' * 48}",
        f"  {'Grounded claims':<34} {report.grounded_rate:>11.1%}",
        f"  {'Mean verification score':<34} {report.mean_score:>11.3f}",
        f"  {'Mean semantic support':<34} {report.mean_semantic_support:>11.3f}",
        f"  {'Mean spatial agreement':<34} {report.mean_spatial_agreement:>11.3f}",
    ]
    if report.ungrounded:
        lines.append("")
        lines.append("  Ungrounded claims (below threshold):")
        for claim in report.ungrounded:
            lines.append(f"    - {claim}")
    lines.extend(["", "=" * 70, ""])
    return "\n".join(lines)
