"""Pure-logic tests for the semantic grounding in vqasynth.correspondence.

VQASynth++ Phase 2c: cross-view correspondence QA must reference actual
objects from the localize stage (a matched pixel inside mask A grounds the
match to label A) instead of the anonymous "feature", and the RANSAC inlier
ratio must be surfaced as a per-example column. Acceptance bar from the
brief — everything here is CPU-only and dependency-light: masks are nested
bool lists (same ``[y][x]`` indexing as the localize stage's numpy arrays),
matches are hand-built ``CorrespondenceMatch`` dataclasses, and the
extractor path is stubbed — no SIFT, no cv2, no model weights. Mirrors the
structural-test philosophy of ``tests/test_correspondence.py``.
"""
from __future__ import annotations

import re

import pytest

from vqasynth.correspondence import (
    CorrespondenceExtractor,
    CorrespondenceMatch,
    CorrespondenceResult,
    aggregate_correspondences,
    build_qa_pair,
    ground_correspondences,
    ground_point_to_object,
)

# Exact regex from vqasynth.localize.extract_points_and_descriptions — the
# contract our <point> output must satisfy to round-trip through the pipeline.
LOCALIZE_POINT_RE = re.compile(
    r'<point\s+x="\s*([0-9]+(?:\.[0-9]+)?)"\s+y="\s*([0-9]+(?:\.[0-9]+)?)"\s+alt="([^"]+)">'
)


def _rect_mask(h, w, y0, y1, x0, x1):
    """2D boolean mask with rows [y0, y1) x cols [x0, x1) set."""
    return [[y0 <= y < y1 and x0 <= x < x1 for x in range(w)] for y in range(h)]


# Two 10x10 views of the same scene: a chair top-left in view A, top-right in
# view B (the viewpoint shifted); a desk bottom-right in both views.
_SIZE = (10, 10)  # (W, H)
_MASKS_A = [_rect_mask(10, 10, 0, 5, 0, 5), _rect_mask(10, 10, 5, 10, 5, 10)]
_CAPTIONS_A = ["wooden chair", "desk"]
_MASKS_B = [_rect_mask(10, 10, 0, 5, 5, 10), _rect_mask(10, 10, 5, 10, 5, 10)]
_CAPTIONS_B = ["wooden chair", "desk"]


def _result(matches, inlier_count=0, raw_match_count=0):
    return CorrespondenceResult(
        matches=matches,
        view_a_size=_SIZE,
        view_b_size=_SIZE,
        inlier_count=inlier_count,
        raw_match_count=raw_match_count,
    )


# ── point -> object lookup ──────────────────────────────────────────────

def test_point_inside_mask_grounds_to_its_label():
    # (x=2, y=2) sits in view A's chair region (rows 0-4, cols 0-4).
    assert ground_point_to_object((2, 2), _MASKS_A, _CAPTIONS_A) == "wooden chair"
    # (x=7, y=7) sits in the desk region (rows 5-9, cols 5-9).
    assert ground_point_to_object((7, 7), _MASKS_A, _CAPTIONS_A) == "desk"


def test_point_outside_all_masks_returns_none():
    # (x=8, y=0): top-right corner — inside neither region of view A.
    assert ground_point_to_object((8, 0), _MASKS_A, _CAPTIONS_A) is None
    # Negative and out-of-bounds coordinates contain nothing.
    assert ground_point_to_object((-1, 2), _MASKS_A, _CAPTIONS_A) is None
    assert ground_point_to_object((2, 999), _MASKS_A, _CAPTIONS_A) is None
    # Missing localize outputs (None / empty) ground nothing.
    assert ground_point_to_object((2, 2), None, None) is None
    assert ground_point_to_object((2, 2), [], []) is None


def test_float_pixels_are_rounded_before_lookup():
    # (2.4, 2.6) rounds to (2, 3) — still inside the chair region.
    assert ground_point_to_object((2.4, 2.6), _MASKS_A, _CAPTIONS_A) == "wooden chair"


# ── grounding a result's matches ────────────────────────────────────────

def test_ground_correspondences_labels_each_match_from_either_view():
    result = _result(
        [
            CorrespondenceMatch(pt_a=(2, 2), pt_b=(7, 2)),    # chair via view A
            CorrespondenceMatch(pt_a=(3, 8), pt_b=(7, 7)),    # desk via view B
            CorrespondenceMatch(pt_a=(8, 0), pt_b=(0, 9)),    # inside no object
        ],
        inlier_count=3,
        raw_match_count=6,
    )
    grounded = ground_correspondences(
        result, masks_a=_MASKS_A, captions_a=_CAPTIONS_A,
        masks_b=_MASKS_B, captions_b=_CAPTIONS_B,
    )
    assert [m.caption for m in grounded.matches] == ["wooden chair", "desk", "feature"]
    # (3, 8) is outside view A's objects but its view-B pixel (7, 7) sits in
    # the desk — grounding falls through to the destination view.
    assert grounded.matches[1].caption == "desk"
    # Purity: the input result is not mutated (caller can re-ground it).
    assert [m.caption for m in result.matches] == ["feature", "feature", "feature"]


def test_ground_correspondences_preserves_explicit_captions():
    # A caption supplied by a downstream localizer must not be clobbered.
    result = _result([CorrespondenceMatch(pt_a=(2, 2), pt_b=(7, 2), caption="red mug")])
    grounded = ground_correspondences(
        result, masks_a=_MASKS_A, captions_a=_CAPTIONS_A,
        masks_b=_MASKS_B, captions_b=_CAPTIONS_B,
    )
    assert grounded.matches[0].caption == "red mug"


def test_ground_correspondences_without_localize_output_keeps_feature():
    result = _result([CorrespondenceMatch(pt_a=(2, 2), pt_b=(7, 2))])
    grounded = ground_correspondences(result)
    assert grounded.matches[0].caption == "feature"


# ── grounded QA renders the object label ────────────────────────────────

def test_grounded_match_renders_object_label_in_qa():
    """The grounded label must flow into the question text AND both <point>
    tags, in localize-parser-compatible form (round-trip contract)."""
    result = _result([CorrespondenceMatch(pt_a=(2, 2), pt_b=(7, 2))])
    grounded = ground_correspondences(
        result, masks_a=_MASKS_A, captions_a=_CAPTIONS_A,
        masks_b=_MASKS_B, captions_b=_CAPTIONS_B,
    )
    question, answer = build_qa_pair(grounded.matches[0], _SIZE, _SIZE)

    # Question references the object by name, not "feature".
    assert "wooden chair" in question
    assert "feature" not in question
    # Source tag (view A: 2/10 -> 20) and answer tag (view B: 7/10 -> 70)
    # both carry the object label and parse through localize's regex.
    src = LOCALIZE_POINT_RE.search(question)
    dst = LOCALIZE_POINT_RE.search(answer)
    assert src is not None and dst is not None
    assert float(src.group(1)) == 20.0 and src.group(3) == "wooden chair"
    assert float(dst.group(1)) == 70.0 and dst.group(3) == "wooden chair"


# ── inlier-ratio signal ─────────────────────────────────────────────────

def test_inlier_ratio_property_is_keep_rate():
    assert _result([], inlier_count=8, raw_match_count=10).inlier_ratio == pytest.approx(0.8)
    # Division guarded when nothing survived the ratio test.
    assert _result([], inlier_count=0, raw_match_count=0).inlier_ratio == 0.0


def test_aggregate_correspondences_populates_inlier_ratio_column():
    results = [
        _result([CorrespondenceMatch(pt_a=(1, 1), pt_b=(2, 2))], 8, 10),
        _result([CorrespondenceMatch(pt_a=(3, 3), pt_b=(4, 4))], 5, 10),
    ]
    row = aggregate_correspondences(results)
    assert set(row.keys()) == {"messages", "inlier_ratio"}
    assert row["inlier_ratio"] == pytest.approx(0.65)  # mean of 0.8 and 0.5
    assert len(row["messages"]) == 4  # two QA turns folded into one conversation
    # No pairs (single-view clip) -> empty conversation, zero similarity.
    empty = aggregate_correspondences([])
    assert empty == {"messages": [], "inlier_ratio": 0.0}


# ── apply_transform wiring (extractor stubbed — no cv2 needed) ──────────

class _FakePIL:
    """Duck-typed stand-in for PIL.Image — has .mode/.size/.convert."""

    def __init__(self, mode="RGB", size=(10, 10)):
        self.mode = mode
        self.size = size

    def convert(self, mode):
        return _FakePIL(mode=mode, size=self.size)


def test_apply_transform_emits_grounded_messages_and_inlier_ratio(monkeypatch):
    """End-to-end wiring: localize columns in -> grounded messages + the
    inlier_ratio column out. ``extract`` is stubbed so no cv2 is needed."""
    canned = _result(
        [CorrespondenceMatch(pt_a=(2, 2), pt_b=(7, 2))], inlier_count=4, raw_match_count=8
    )
    extractor = CorrespondenceExtractor()
    monkeypatch.setattr(extractor, "extract", lambda a, b: canned)

    example = {
        "image": [_FakePIL(), _FakePIL()],
        "masks": [_MASKS_A, _MASKS_B],          # localize stage's per-view output
        "captions": [_CAPTIONS_A, _CAPTIONS_B],
    }
    out = extractor.apply_transform(example, images="image")

    # View-similarity column populated from the stubbed RANSAC accounting.
    assert out["inlier_ratio"] == pytest.approx(0.5)
    # Messages grounded: the question names the object, not "feature".
    question = out["messages"][0]["content"][-1]["text"]
    answer = out["messages"][1]["content"][0]["text"]
    assert "wooden chair" in question
    assert LOCALIZE_POINT_RE.search(answer).group(3) == "wooden chair"


def test_apply_transform_without_localize_columns_stays_back_compat(monkeypatch):
    """No masks/captions columns -> un-grounded "feature" QA, ratio column
    still emitted."""
    canned = _result([CorrespondenceMatch(pt_a=(2, 2), pt_b=(7, 2))], 4, 8)
    extractor = CorrespondenceExtractor()
    monkeypatch.setattr(extractor, "extract", lambda a, b: canned)

    out = extractor.apply_transform({"image": [_FakePIL(), _FakePIL()]}, images="image")

    assert out["inlier_ratio"] == pytest.approx(0.5)
    answer = out["messages"][1]["content"][0]["text"]
    assert LOCALIZE_POINT_RE.search(answer).group(3) == "feature"
