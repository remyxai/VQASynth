"""Multi-view point-correspondence tool wrapper for the SpatialAnnotator agent.

Wraps :class:`vqasynth.correspondence.CorrespondenceExtractor` (PR #127) as a
NOOA agent tool, so the SpatialAnnotator can ask "which pixel in view B
corresponds to the point I clicked in view A?" as a discrete step in a
dynamically-composed pipeline — instead of only through the batch Docker stage
(``docker/correspondence_stage/``). Applicable to Ego4D adjacent-frame
annotation and any multi-view robotics scenario.

This is the FIRST multi-view tool in NOOA's inventory: every other tool in
:mod:`experiments.nooa_agent.tools` (depth, orientation, describe, pose,
boxes3d) takes a single image. The shape here — a tool function that takes two
views and returns a compact result dataclass backed by a lazy-loaded singleton —
extends verbatim to N views: an N-view variant is just a list of views returning
one :class:`CorrespondenceResult` per adjacent pair. Future tool authors should
follow this two-view entry point rather than re-derive a single-image shape.

Single backend for now: OpenCV classical (SIFT + ratio-tested BFMatcher +
RANSAC homography filter), CPU-only, no model weights — exactly what the
underlying stage ships. ``backend="flann"`` opts into the approximate-NN matcher
for very large keypoint sets. The neural multi-view backends cited in PR #127's
brief (StreamVGGT — arXiv:2507.11539; PlanarRecon — arXiv:2104.00681) are NOT
wired up here — they were explicitly deferred there and are left as
future-extension slots on the ``backend`` field.

Heavy deps (cv2, numpy, PIL) stay deferred: this module imports only the stdlib
at load time and constructs :class:`CorrespondenceExtractor` lazily inside
:meth:`CorrespondenceExtractorAgent._ensure_loaded`, mirroring the lazy-load
convention in :mod:`experiments.nooa_agent.tools.depth` /
:mod:`experiments.nooa_agent.tools.orientation`. Importing this module therefore
never requires cv2/numpy/PIL — the always-on structural tests rely on that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _tool_backend(underlying: str) -> str:
    """Re-label the underlying stage's backend for the agent surface.

    The stage uses ``"sift-bf"`` / ``"sift-flann"`` / ``"orb-bf"`` /
    ``"orb-flann"``; we prefix with ``opencv_`` and swap ``-`` for ``_`` so the
    label names the implementation family. A future neural backend would carry
    its own label (e.g. ``"streamvggt"``) without colliding with this scheme.
    """
    return f"opencv_{underlying.replace('-', '_')}"


def _looks_like_pil(view: Any) -> bool:
    """Duck-type a PIL image without importing PIL.

    Matches the underlying stage's own ``_to_gray`` coercion, which tests for
    ``convert`` + ``size`` rather than ``isinstance(view, Image.Image)`` — so a
    view the stage would accept also passes the tool's input guard.
    """
    return hasattr(view, "convert") and hasattr(view, "size")


def _looks_like_ndarray(view: Any) -> bool:
    """Duck-type an ``(H, W, C)`` ndarray without importing numpy."""
    return hasattr(view, "shape") and hasattr(view, "ndim") and not _looks_like_pil(view)


@dataclass
class CorrespondenceMatch:
    """A single matched point pair across two views (agent-facing).

    Fields:
        point_a: ``(x, y)`` pixel coordinate of the feature in view A.
        point_b: matched ``(x, y)`` pixel coordinate in view B.
        confidence: ``0..1`` — the RANSAC inlier score for the result this match
            came from (``n_kept / n_raw``), shared across every match a single
            :func:`find_correspondences` call returns. The underlying stage does
            not surface a per-match Lowe ratio, so the global inlier score is the
            available per-match confidence today.
    """

    point_a: tuple[float, float]
    point_b: tuple[float, float]
    confidence: float


@dataclass
class CorrespondenceResult:
    """Agent-facing output of :func:`find_correspondences`.

    Fields:
        matches: post-RANSAC correspondences — the agent-consumable summary.
            Raw SIFT descriptors are deliberately NOT surfaced.
        view_a_shape: ``(H, W)`` of view A — row-major, for downstream
            coordinate normalization (e.g. rescaling pixel coords into a 0..1
            or Molmo 0..100 frame). NOTE the ``(H, W)`` order: it differs from
            the underlying stage's ``(W, H)`` ``view_a_size``.
        view_b_shape: ``(H, W)`` of view B.
        n_kept: matches surviving the RANSAC geometry filter.
        n_raw: matches surviving the Lowe ratio test, BEFORE RANSAC.
            ``n_kept / n_raw`` is the inlier ratio — a 5% keep rate is a strong
            hint the two views are of different scenes.
        backend: ``"opencv_sift_bf"`` by default (or ``"opencv_sift_flann"``).
    """

    matches: list[CorrespondenceMatch]
    view_a_shape: tuple[int, int]
    view_b_shape: tuple[int, int]
    n_kept: int
    n_raw: int
    backend: str = "opencv_sift_bf"

    def __repr__(self) -> str:
        # Compact one-liner. A NOOA trace event fires per tool call and a busy
        # scene can yield hundreds of matches, so we must NOT enumerate them —
        # mirrors DepthResult / OrientationResult.__repr__'s guard against
        # dumping large per-call payloads into the trace.
        return (
            f"CorrespondenceResult(backend={self.backend!r}, "
            f"kept={self.n_kept}/{self.n_raw}, "
            f"view_shapes=({self.view_a_shape}, {self.view_b_shape}))"
        )


class CorrespondenceExtractorAgent:
    """Backend composing :class:`vqasynth.correspondence.CorrespondenceExtractor`.

    Does NOT reimplement SIFT / BFMatcher / RANSAC — it constructs the
    underlying stage and lifts its result into the agent-facing
    :class:`CorrespondenceResult`, adding the ``(H, W)`` shape convention, the
    ``opencv_*`` backend label, and the ``n_raw`` / per-match ``confidence``
    signals the agent consumes.

    Args:
        backend: ``None`` / ``"bf"`` (default) -> SIFT + BFMatcher.
            ``"flann"`` -> SIFT + FLANN (approximate nearest-neighbour; faster on
            very large keypoint sets). Also accepts the full tool labels
            (``"opencv_sift_bf"`` / ``"opencv_sift_flann"``) for round-tripping.
    """

    BACKEND = "opencv_sift_bf"

    def __init__(self, backend: str | None = None):
        self.matcher_name = self._parse_matcher(backend)  # "bf" | "flann"
        self._extractor = None

    @staticmethod
    def _parse_matcher(backend: str | None) -> str:
        if backend is None or backend in ("bf", "opencv_sift_bf"):
            return "bf"
        if backend in ("flann", "opencv_sift_flann"):
            return "flann"
        raise ValueError(
            "backend must be None, 'bf', 'flann', or a full 'opencv_sift_*' "
            f"label; got {backend!r}"
        )

    def _ensure_loaded(self) -> None:
        if self._extractor is not None:
            return
        # Lazy: keeps cv2/numpy out of module load. The underlying stage's own
        # _ensure_loaded raises a helpful ImportError if opencv is absent, so a
        # missing dependency surfaces as a clear error rather than a cryptic one.
        from vqasynth.correspondence import CorrespondenceExtractor

        self._extractor = CorrespondenceExtractor(
            detector="sift", matcher=self.matcher_name
        )

    def _lift(self, underlying: Any) -> CorrespondenceResult:
        """Lift the underlying stage's result into the agent-facing shape.

        Pure conversion — no cv2/numpy. Swaps ``(W, H)`` -> ``(H, W)``,
        re-prefixes the backend label, and derives ``n_raw`` + the per-match
        ``confidence`` from the stage's inlier accounting.
        """
        # underlying.view_a_size / view_b_size are (W, H); the tool surfaces (H, W).
        aw, ah = underlying.view_a_size
        bw, bh = underlying.view_b_size
        n_kept = len(underlying.matches)
        n_raw = int(getattr(underlying, "raw_match_count", 0))
        # Per-match confidence = the result's RANSAC inlier score. Defensive
        # clamp: the ratio is in [0, 1] by construction today, but a future
        # backend that reports counts differently must not leak > 1.
        confidence = (n_kept / n_raw) if n_raw > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))
        matches = [
            CorrespondenceMatch(
                point_a=(float(m.pt_a[0]), float(m.pt_a[1])),
                point_b=(float(m.pt_b[0]), float(m.pt_b[1])),
                confidence=confidence,
            )
            for m in underlying.matches
        ]
        return CorrespondenceResult(
            matches=matches,
            view_a_shape=(int(ah), int(aw)),
            view_b_shape=(int(bh), int(bw)),
            n_kept=n_kept,
            n_raw=n_raw,
            backend=_tool_backend(underlying.backend),
        )

    def extract(self, view_a: Any, view_b: Any) -> CorrespondenceResult:
        """Extract RANSAC-filtered point correspondences view A -> view B.

        Args:
            view_a / view_b: ``PIL.Image`` or ``np.ndarray`` (``HxWxC``, RGB) —
                the two views of the same scene.

        Returns:
            :class:`CorrespondenceResult` with the RANSAC-filtered matches.
        """
        self._ensure_loaded()
        return self._lift(self._extractor.extract(view_a, view_b))


# Module-level singleton so N tool calls in one agent session share one
# configured extractor, keeping the surface uniform with the other tool
# backends (see ``orientation._get_default_estimator`` /
# ``boxes3d._get_default_estimator``).
_DEFAULT_EXTRACTOR: CorrespondenceExtractorAgent | None = None


def _get_default_extractor() -> CorrespondenceExtractorAgent:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = CorrespondenceExtractorAgent()
    return _DEFAULT_EXTRACTOR


def find_correspondences(view_a: Any, view_b: Any) -> CorrespondenceResult:
    """Find which pixels in ``view_b`` correspond to points in ``view_a``.

    Wraps the batch correspondence stage as a discrete NOOA tool step: the
    SpatialAnnotator can ground a "the same point in the other view is here"
    claim in real geometry instead of dropping into
    ``docker/correspondence_stage/``.

    Args:
        view_a / view_b: ``PIL.Image`` or ``np.ndarray`` (``HxWxC``, RGB) — the
            two views of the same scene.

    Returns:
        A single :class:`CorrespondenceResult` with the RANSAC-filtered matches.
        ``n_kept`` vs ``n_raw`` is the agent's signal for how well the two views
        actually correspond — a low keep rate hints the views are of different
        scenes.
    """
    for name, view in (("view_a", view_a), ("view_b", view_b)):
        if not (_looks_like_pil(view) or _looks_like_ndarray(view)):
            raise ValueError(
                f"{name} must be a PIL.Image or np.ndarray (HxWxC), "
                f"got {type(view).__name__}"
            )
    return _get_default_extractor().extract(view_a, view_b)


__all__ = [
    "CorrespondenceMatch",
    "CorrespondenceResult",
    "CorrespondenceExtractorAgent",
    "find_correspondences",
]
