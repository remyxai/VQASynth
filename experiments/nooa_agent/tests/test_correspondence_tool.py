"""Structural smoke tests for the NOOA correspondence tool wrapper.

Verifies the :class:`CorrespondenceMatch` / :class:`CorrespondenceResult`
dataclass shapes + compact ``__repr__``, the ``(W,H)`` -> ``(H,W)`` shape swap +
backend relabel + ``n_raw``/confidence derivation in the lift, the tool-boundary
validation guards, and the singleton delegation — all WITHOUT cv2/numpy/PIL
(heavy deps are imported lazily inside the few tests that need them, so the
always-on suite is collectable in a minimal env). The real OpenCV extraction
path is exercised by a cv2-guarded end-to-end test (``pytest.importorskip``)
that runs only when opencv is importable — mirroring the philosophy of
``tests/test_correspondence.py`` / ``tests/test_orientation_tool.py``.

Imports from the pre-existing package module this tool composes —
``vqasynth.correspondence`` (the stage whose ``CorrespondenceExtractor.extract``
the agent delegates to) — so this is a cross-module integration check, not a
self-test of ``correspondence`` alone.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from experiments.nooa_agent.tools.correspondence import (
    CorrespondenceExtractorAgent,
    CorrespondenceMatch,
    CorrespondenceResult,
    find_correspondences,
)
# Pre-existing package module — exercised here so the suite isn't a self-test of
# only the new tool: the lift must read real fields off the upstream
# CorrespondenceResult / CorrespondenceMatch produced by vqasynth.correspondence.
from vqasynth.correspondence import (
    CorrespondenceExtractor,
    CorrespondenceMatch as UpstreamMatch,
    CorrespondenceResult as UpstreamResult,
)


# ── CorrespondenceResult.__repr__ — compact, bounded, no per-match dump ────

def test_result_repr_is_compact_with_500_matches():
    """A NOOA trace event fires per tool call, and a busy scene can yield
    hundreds of matches — the repr must stay one line and bounded regardless of
    match count (mirrors DepthResult / OrientationResult.__repr__)."""
    result = CorrespondenceResult(
        matches=[
            CorrespondenceMatch(
                point_a=(float(i), float(i)),
                point_b=(float(i), float(i)),
                confidence=0.42,
            )
            for i in range(500)
        ],
        view_a_shape=(480, 640),
        view_b_shape=(480, 640),
        n_kept=500,
        n_raw=900,
        backend="opencv_sift_bf",
    )
    text = repr(result)
    assert "\n" not in text
    assert len(text) < 200, f"repr is {len(text)} chars — probably dumping matches"
    assert "opencv_sift_bf" in text
    assert "kept=500/900" in text
    # No per-match coordinate / confidence dump — a busy scene would blow up the trace.
    assert "point_a" not in text
    assert "(0.0, 0.0)" not in text


def test_result_repr_format_matches_brief_example():
    result = CorrespondenceResult(
        matches=[],
        view_a_shape=(256, 256),
        view_b_shape=(256, 256),
        n_kept=142,
        n_raw=210,
        backend="opencv_sift_bf",
    )
    assert repr(result) == (
        "CorrespondenceResult(backend='opencv_sift_bf', kept=142/210, "
        "view_shapes=((256, 256), (256, 256)))"
    )


# ── _lift — composes the pre-existing vqasynth.correspondence result ──────

def _upstream_result(n_matches: int, raw: int | None = None, backend: str = "sift-bf"):
    """Build a REAL vqasynth.correspondence.CorrespondenceResult (no cv2) to
    drive the lift — the agent must read its fields verbatim."""
    matches = [
        UpstreamMatch(
            pt_a=(float(i) * 2.0, float(i) * 3.0),
            pt_b=(float(i), float(i)),
            caption="feature",
            distance=float(i),
        )
        for i in range(n_matches)
    ]
    if raw is None:
        raw = n_matches
    return UpstreamResult(
        matches=matches,
        view_a_size=(640, 480),  # (W, H)
        view_b_size=(320, 240),
        homography=None,
        inlier_count=n_matches,
        raw_match_count=raw,
        backend=backend,
    )


def test_agent_lift_swaps_width_height_size_to_height_width_shape():
    """The tool surfaces ``(H, W)`` shapes — the inverse order of the underlying
    stage's ``(W, H)`` ``view_*_size`` — for downstream row-major coord
    normalization."""
    agent = CorrespondenceExtractorAgent()
    result = agent._lift(_upstream_result(n_matches=4))
    assert result.view_a_shape == (480, 640)  # (H, W) — swapped from (640, 480)
    assert result.view_b_shape == (240, 320)


def test_agent_lift_maps_backend_and_counts_and_confidence():
    agent = CorrespondenceExtractorAgent()
    result = agent._lift(_upstream_result(n_matches=8, raw=20, backend="sift-bf"))
    assert result.backend == "opencv_sift_bf"  # sift-bf -> opencv_sift_bf
    assert result.n_kept == 8
    assert result.n_raw == 20
    # confidence is the RANSAC inlier score (n_kept / n_raw), shared per match.
    expected_conf = 8 / 20
    assert len(result.matches) == 8
    for m in result.matches:
        assert m.confidence == pytest.approx(expected_conf)
        assert isinstance(m.point_a, tuple) and len(m.point_a) == 2
        assert isinstance(m.point_b, tuple) and len(m.point_b) == 2


def test_agent_lift_carries_matched_coordinates_verbatim():
    agent = CorrespondenceExtractorAgent()
    result = agent._lift(_upstream_result(n_matches=1))
    assert result.matches[0].point_a == (0.0, 0.0)
    assert result.matches[0].point_b == (0.0, 0.0)


def test_agent_lift_empty_upstream_result():
    """No matches -> n_kept 0, confidence-derivation guarded (0/0 -> 0.0), repr
    still works (no crash)."""
    agent = CorrespondenceExtractorAgent()
    result = agent._lift(_upstream_result(n_matches=0, raw=0))
    assert result.matches == []
    assert result.n_kept == 0
    assert result.n_raw == 0
    assert "kept=0/0" in repr(result)


# ── backend selection (parsed before the lazy cv2 load) ───────────────────

def test_backend_constructor_defaults_to_bf():
    assert CorrespondenceExtractorAgent().matcher_name == "bf"


def test_backend_constructor_selects_flann():
    assert CorrespondenceExtractorAgent(backend="flann").matcher_name == "flann"


def test_backend_constructor_accepts_full_tool_label():
    assert CorrespondenceExtractorAgent(backend="opencv_sift_flann").matcher_name == "flann"


def test_backend_constructor_rejects_unknown():
    with pytest.raises(ValueError, match="backend"):
        CorrespondenceExtractorAgent(backend="magic")


# ── find_correspondences — delegation + input validation (no cv2) ─────────

class _FakePIL:
    """Duck-typed stand-in for PIL.Image — passes the tool's ``_looks_like_pil``
    check (``convert`` + ``size``) without requiring Pillow, so the delegation
    boundary is testable in a minimal env."""

    def __init__(self, size=(10, 10)):
        self.size = size

    def convert(self, mode):  # noqa: D401 - test stub
        return _FakePIL(size=self.size)


class _FakeArray:
    """Duck-typed stand-in for an (H, W, C) ndarray (``shape`` + ``ndim``)."""

    ndim = 3
    shape = (8, 12, 3)


def test_find_correspondences_delegates_to_default_extractor(monkeypatch):
    """Patch the singleton to a stub agent returning a known result, so the
    tool->agent delegation boundary is tested independent of cv2/SIFT. Mirrors
    test_detect_3d_boxes_delegates_to_default_estimator."""
    canned = CorrespondenceResult(
        matches=[CorrespondenceMatch(point_a=(1.0, 2.0), point_b=(3.0, 4.0), confidence=0.9)],
        view_a_shape=(10, 20),
        view_b_shape=(10, 20),
        n_kept=1,
        n_raw=1,
    )

    class _StubAgent:
        def __init__(self):
            self.received = []

        def extract(self, view_a, view_b):
            self.received.append((view_a, view_b))
            return canned

    stub = _StubAgent()
    monkeypatch.setattr(
        "experiments.nooa_agent.tools.correspondence._get_default_extractor",
        lambda: stub,
    )

    a, b = _FakePIL(), _FakePIL()
    result = find_correspondences(a, b)
    assert result is canned
    assert stub.received == [(a, b)]  # both views forwarded verbatim


def test_find_correspondences_accepts_ndarray_like(monkeypatch):
    """An ndarray (has .shape + .ndim) passes the input guard too — the
    underlying stage accepts (HxWxC) arrays via the same duck type."""
    canned = CorrespondenceResult(
        matches=[], view_a_shape=(8, 12), view_b_shape=(8, 12), n_kept=0, n_raw=0
    )
    monkeypatch.setattr(
        "experiments.nooa_agent.tools.correspondence._get_default_extractor",
        lambda: type("S", (), {"extract": staticmethod(lambda a, b: canned)})(),
    )
    result = find_correspondences(_FakeArray(), _FakeArray())
    assert result is canned


def test_find_correspondences_rejects_non_image_input():
    with pytest.raises(ValueError, match="view_a"):
        find_correspondences("not-an-image", _FakePIL())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="view_b"):
        find_correspondences(_FakePIL(), 42)  # type: ignore[arg-type]


# ── API-drift guards vs upstream vqasynth.correspondence ──────────────────
#
# The wrapper composes CorrespondenceExtractor + lifts its CorrespondenceResult
# via a handful of surface points. Stub tests never touch the real classes;
# these guards catch upstream drift explicitly (cf. test_boxes3d_tool.py).

def test_underlying_result_exposes_fields_the_wrapper_reads():
    """``_lift`` reads ``view_a_size`` / ``view_b_size`` / ``raw_match_count`` /
    ``backend`` off the upstream CorrespondenceResult. Guard those names — a
    rename would silently zero ``n_raw`` (and hence confidence) or mislabel the
    backend."""
    fields = {f.name for f in dataclasses.fields(UpstreamResult)}
    for name in (
        "matches", "view_a_size", "view_b_size",
        "inlier_count", "raw_match_count", "backend",
    ):
        assert name in fields, (
            f"upstream CorrespondenceResult lost {name} (has: {sorted(fields)})"
        )


def test_underlying_match_exposes_pt_a_pt_b():
    """``_lift`` reads ``pt_a`` / ``pt_b`` off each upstream match."""
    fields = {f.name for f in dataclasses.fields(UpstreamMatch)}
    for name in ("pt_a", "pt_b"):
        assert name in fields, (
            f"upstream CorrespondenceMatch lost {name} (has: {sorted(fields)})"
        )


def test_underlying_extractor_extract_signature():
    """The wrapper calls ``CorrespondenceExtractor().extract(view_a, view_b)``.
    Guard the method name + parameter names against an upstream rename."""
    method = getattr(CorrespondenceExtractor, "extract", None)
    assert callable(method), "CorrespondenceExtractor.extract went missing"
    params = list(inspect.signature(method).parameters)
    assert params[:3] == ["self", "image_a", "image_b"], (
        f"extract signature drifted: {params}"
    )


# ── cv2-guarded end-to-end through the real CorrespondenceExtractor ────────

def _warped_pair():
    """A textured image + a known-homography warp of it (deterministic — no
    external image). Mirrors ``examples/correspondence_example.make_synthetic_view``
    and ``tests/test_correspondence._synthetic_pair``."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.RandomState(0)
    base = (rng.rand(256, 256, 3) * 255).astype(np.uint8)
    for x in range(0, 256, 32):
        cv2.line(base, (x, 0), (x, 256), (0, 0, 0), 1)
        cv2.line(base, (0, x), (256, x), (0, 0, 0), 1)
    src = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
    dst = np.float32([[8, 4], [250, 6], [254, 252], [4, 248]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(base, H, (256, 256))
    return base, warped


def test_find_correspondences_returns_matches_on_warped_pair():
    base, warped = _warped_pair()
    result = find_correspondences(base, warped)

    assert isinstance(result, CorrespondenceResult)
    assert result.view_a_shape == (256, 256)  # (H, W)
    assert result.view_b_shape == (256, 256)
    assert result.backend == "opencv_sift_bf"
    # A mildly-warped textured pair must yield geometrically-consistent matches.
    assert len(result.matches) >= 4
    assert result.n_kept == len(result.matches)
    assert result.n_raw >= result.n_kept
    # Confidence is the inlier ratio, in [0, 1]; recovered coords stay in-bounds.
    for m in result.matches:
        assert 0.0 <= m.confidence <= 1.0
        assert 0 <= m.point_a[0] <= 256 and 0 <= m.point_a[1] <= 256
        assert 0 <= m.point_b[0] <= 256 and 0 <= m.point_b[1] <= 256


def test_find_correspondences_empty_on_identical_blank():
    """A featureless image yields no descriptors -> empty result, no crash."""
    pytest.importorskip("cv2")
    import numpy as np

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    result = find_correspondences(blank, blank)
    assert result.matches == []
    assert result.n_kept == 0
    assert result.n_raw == 0
    assert "kept=0/0" in repr(result)


def test_keep_ratio_lower_for_unrelated_than_warped_views():
    """``n_kept`` / ``n_raw`` is the agent's view-similarity signal: related
    (warped) views keep a higher fraction of ratio-test matches than unrelated
    views, which share no geometry for RANSAC to recover."""
    base, warped = _warped_pair()
    related = find_correspondences(base, warped)

    cv2 = pytest.importorskip("cv2")
    import numpy as np

    # Unrelated view: different random texture + a different grid phase — rich
    # enough to produce ratio-test matches, but sharing no coherent geometry.
    rng = np.random.RandomState(123)
    other = (rng.rand(256, 256, 3) * 255).astype(np.uint8)
    for x in range(16, 256, 48):
        cv2.line(other, (x, 0), (x, 256), (0, 0, 0), 2)
        cv2.line(other, (0, x), (256, x), (0, 0, 0), 2)
    unrelated = find_correspondences(base, other)

    def _ratio(r: CorrespondenceResult) -> float:
        return r.n_kept / r.n_raw if r.n_raw > 0 else 0.0

    # Related views must actually correspond (sanity), then dominate unrelated.
    assert related.n_kept >= 4
    if unrelated.n_raw > 0:
        assert _ratio(unrelated) <= _ratio(related)
        # Loose absolute bound — unrelated views keep well under half their
        # ratio-test matches (no exact number; bounded per the brief).
        assert _ratio(unrelated) <= 0.5
