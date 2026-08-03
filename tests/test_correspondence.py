"""Structural tests for vqasynth.correspondence.

Mirrors the philosophy of ``tests/test_vggt_speedups.py``: verify the stage +
converter mechanics without a GPU, model download, or (for the converter) even
cv2/numpy/PIL installed. The OpenCV extractor path is exercised by a
cv2-guarded end-to-end test that runs only when opencv is importable.

Format compatibility with the existing pointing-VLM pipeline is asserted two
ways: (1) an always-on check against the exact ``<point>`` regex that
``vqasynth.localize`` uses, and (2) a guarded round-trip through
``vqasynth.localize.extract_points_and_descriptions`` itself when its
heavy deps (sam2/transformers) are importable.
"""
from __future__ import annotations

import re

import pytest

from vqasynth import prompt_templates  # pre-existing module — template vocabulary
from vqasynth.correspondence import (
    CorrespondenceExtractor,
    CorrespondenceMatch,
    CorrespondenceResult,
    _substitute,
    build_qa_message,
    build_qa_pair,
    correspondences_to_messages,
    correspondences_to_point_qa,
    point_to_molmo_tag,
    to_molmo_coord,
)

# Exact regex from vqasynth.localize.extract_points_and_descriptions — the
# contract our <point> output must satisfy to round-trip through the pipeline.
LOCALIZE_POINT_RE = re.compile(
    r'<point\s+x="\s*([0-9]+(?:\.[0-9]+)?)"\s+y="\s*([0-9]+(?:\.[0-9]+)?)"\s+alt="([^"]+)">'
)


def _first_point_tag(text: str):
    """Return the first <point ...> match in text, or None."""
    return LOCALIZE_POINT_RE.search(text)


# ── coordinate / tag helpers ────────────────────────────────────────────

def test_to_molmo_coord_scales_pixels_to_0_100():
    assert to_molmo_coord(160, 320) == 50.0
    assert to_molmo_coord(0, 320) == 0.0
    assert to_molmo_coord(320, 320) == 100.0


def test_to_molmo_coord_clamps_out_of_range():
    # localize rejects tags whose max coord > 100; clamping guarantees compliance.
    assert to_molmo_coord(400, 320) == 100.0
    assert to_molmo_coord(-10, 320) == 0.0


def test_to_molmo_coord_rounds_and_rejects_bad_extent():
    assert to_molmo_coord(1, 3) == 33.33
    with pytest.raises(ValueError, match="extent_px"):
        to_molmo_coord(10, 0)


def test_point_tag_matches_localize_regex():
    tag = point_to_molmo_tag(160, 120, 320, 240, "wooden crate")
    m = _first_point_tag(tag)
    assert m is not None
    assert float(m.group(1)) == 50.0
    assert float(m.group(2)) == 50.0
    assert m.group(3) == "wooden crate"
    # Non-self-closing form — the "/" form is NOT matched by localize's parser.
    assert tag.endswith('>')


def test_point_tag_empty_alt_falls_back_and_clamps():
    tag = point_to_molmo_tag(10_000, 10_000, 100, 100, "")
    m = _first_point_tag(tag)
    assert m is not None
    assert float(m.group(1)) == 100.0
    assert float(m.group(2)) == 100.0
    assert m.group(3) == "point"  # localize needs non-empty alt ([^"]+)


# ── template substitution interop with vqasynth.prompt_templates ────────

def test_substitute_is_compatible_with_repo_template_tokens():
    """The repo's prompt templates (vqasynth.prompt_templates) use [A]/[B]/[X]
    placeholders. Our converter's substitution must render those same tokens so
    correspondence QA composes with the existing template vocabulary."""
    # Pull real templates from the pre-existing module.
    question = prompt_templates.distance_template_questions[0]   # has [A], [B]
    answer = prompt_templates.distance_template_answers[0]        # is "[X]"
    assert "[A]" in question and "[B]" in question
    assert "[X]" in answer

    q_rendered = _substitute(question, A="chair", B="desk", X="2 meters")
    assert "chair" in q_rendered and "desk" in q_rendered
    assert "[A]" not in q_rendered and "[B]" not in q_rendered

    a_rendered = _substitute(answer, A="chair", B="desk", X="2 meters")
    assert a_rendered == "2 meters"


def test_converter_renders_answer_via_substitution():
    match = CorrespondenceMatch(pt_a=(160, 120), pt_b=(480, 360), caption="crate")
    q, a = build_qa_pair(match, (320, 240), (640, 480))
    # Question carries the source <point> (view A, 0-100 space).
    assert _first_point_tag(q) is not None
    # Answer is a <point> in view B's normalized space (480/640, 360/480 -> 75, 75).
    m = _first_point_tag(a)
    assert m is not None
    assert float(m.group(1)) == 75.0
    assert float(m.group(2)) == 75.0
    assert m.group(3) == "crate"


# ── message schema ──────────────────────────────────────────────────────

def _content_blocks(msg, btype):
    return [b for b in msg["content"] if b.get("type") == btype]


def test_build_qa_message_uses_pipeline_message_schema():
    match = CorrespondenceMatch(pt_a=(10, 10), pt_b=(20, 20), caption="box")
    row = build_qa_message(match, (100, 100), (100, 100))
    assert set(row.keys()) == {"messages"}
    msgs = row["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]

    # Multi-view: two image blocks (index 0 = view A, index 1 = view B).
    user_imgs = _content_blocks(msgs[0], "image")
    assert [b["index"] for b in user_imgs] == [0, 1]
    assert all(b["text"] is None for b in user_imgs)

    # One text block on user turn; assistant answer is a <point>.
    user_text = _content_blocks(msgs[0], "text")
    assert len(user_text) == 1 and user_text[0]["index"] is None
    asst_text = _content_blocks(msgs[1], "text")
    assert _first_point_tag(asst_text[0]["text"]) is not None


def test_correspondences_to_messages_attaches_images_only_to_first_turn():
    result = CorrespondenceResult(
        matches=[CorrespondenceMatch(pt_a=(i, i), pt_b=(i + 1, i + 1)) for i in range(5)],
        view_a_size=(10, 10),
        view_b_size=(10, 10),
        inlier_count=5,
    )
    msgs = correspondences_to_messages(result, max_turns=3)
    # 3 QA turns -> 6 messages, alternating user/assistant.
    assert len(msgs) == 6
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3
    # Only the first user turn carries image blocks (matches PromptGenerator).
    first_imgs = _content_blocks(msgs[0], "image")
    later_user_turns = [msgs[i] for i in range(2, len(msgs), 2)]
    assert len(first_imgs) == 2
    for turn in later_user_turns:
        assert _content_blocks(turn, "image") == []


def test_point_qa_rows_are_one_per_match():
    result = CorrespondenceResult(
        matches=[CorrespondenceMatch(pt_a=(1, 1), pt_b=(2, 2)) for _ in range(4)],
        view_a_size=(10, 10),
        view_b_size=(10, 10),
    )
    rows = correspondences_to_point_qa(result)
    assert len(rows) == 4
    assert all(len(r["messages"]) == 2 for r in rows)


def test_result_repr_is_compact():
    """Large match lists must not blow up logs (cf. DepthResult)."""
    result = CorrespondenceResult(
        matches=[CorrespondenceMatch(pt_a=(1.0, 2.0), pt_b=(3.0, 4.0)) for _ in range(1000)],
        view_a_size=(640, 480),
        view_b_size=(640, 480),
        inlier_count=1000,
        homography=[1.0] * 9,
    )
    text = repr(result)
    assert "matches=1000" in text
    assert "pt_a" not in text  # no per-match dump


# ── guarded round-trip through vqasynth.localize ────────────────────────

def _localize_parser():
    """extract_points_and_descriptions, or skip if localize's deps are absent.

    localize imports sam2 + transformers + accelerate at module top, which may
    not all be present in a minimal test environment."""
    try:
        from vqasynth.localize import extract_points_and_descriptions
    except Exception as e:  # ImportError or a transitive failure
        pytest.skip(f"vqasynth.localize not importable in this env: {e}")
    return extract_points_and_descriptions


def test_point_output_round_trips_through_localize_parser():
    """Our <point> tags must parse back through the existing pipeline's parser
    into the same pixel coordinates we encoded."""
    parse = _localize_parser()
    tag = point_to_molmo_tag(160, 60, 320, 240, "fire extinguisher")
    parsed = parse(tag, image_w=320, image_h=240)
    assert len(parsed) == 1
    entry = parsed[0]
    assert entry["points"] == [160.0, 60.0]
    assert entry["caption"] == "fire extinguisher"


# ── view coercion (apply_transform helper, no cv2 needed) ───────────────

class _FakePIL:
    """Duck-typed stand-in for PIL.Image — has .mode/.size/.convert."""

    def __init__(self, mode="RGB", size=(10, 10)):
        self.mode = mode
        self.size = size

    def convert(self, mode):
        return _FakePIL(mode=mode, size=self.size)


def test_coerce_views_normalizes_single_and_list_and_nests_to_rgb():
    coerce = CorrespondenceExtractor._coerce_views
    # Single image.
    assert len(coerce(_FakePIL("RGBA"))) == 1
    assert coerce(_FakePIL("RGBA"))[0].mode == "RGB"
    # List of images.
    out = coerce([_FakePIL("RGBA"), _FakePIL("RGB")])
    assert [v.mode for v in out] == ["RGB", "RGB"]
    # HF-batched nesting: list-of-lists -> take first view of each.
    nested = coerce([[_FakePIL(), _FakePIL()], [_FakePIL()]])
    assert len(nested) == 2


def test_extractor_rejects_bad_config():
    with pytest.raises(ValueError, match="detector"):
        CorrespondenceExtractor(detector="surf")
    with pytest.raises(ValueError, match="matcher"):
        CorrespondenceExtractor(matcher="annoy")
    with pytest.raises(ValueError, match="ratio"):
        CorrespondenceExtractor(ratio=1.5)


# ── cv2-guarded end-to-end extractor test ───────────────────────────────

def _synthetic_pair():
    """Two mildly-warped, highly-textured images with a known homography."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.RandomState(0)
    # Rich texture: random RGB noise + a coarse grid so SIFT has stable corners.
    base = (rng.rand(256, 256, 3) * 255).astype(np.uint8)
    for x in range(0, 256, 32):
        cv2.line(base, (x, 0), (x, 256), (0, 0, 0), 1)
        cv2.line(base, (0, x), (256, x), (0, 0, 0), 1)
    src = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
    dst = np.float32([[8, 4], [250, 6], [254, 252], [4, 248]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(base, H, (256, 256))
    return base, warped


def test_extractor_recovers_correspondences_on_warped_pair():
    base, warped = _synthetic_pair()
    extractor = CorrespondenceExtractor()
    result = extractor.extract(base, warped)

    assert isinstance(result, CorrespondenceResult)
    assert result.view_a_size == (256, 256)
    assert result.view_b_size == (256, 256)
    # A mildly-warped textured pair must yield geometrically-consistent matches.
    assert result.inlier_count >= 4
    assert result.homography is not None
    assert len(result.homography) == 9
    # Recovered match coords stay in-bounds.
    for m in result.matches:
        assert 0 <= m.pt_a[0] <= 256 and 0 <= m.pt_a[1] <= 256
        assert 0 <= m.pt_b[0] <= 256 and 0 <= m.pt_b[1] <= 256


def test_extractor_returns_empty_on_identical_noise_free_blank():
    """A featureless image yields no descriptors -> empty result, no crash."""
    pytest.importorskip("cv2")
    import numpy as np
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    extractor = CorrespondenceExtractor()
    result = extractor.extract(blank, blank)
    assert result.matches == []
    assert result.homography is None
