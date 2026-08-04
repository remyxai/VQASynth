"""Multi-view point-correspondence stage + pointing-VLM (Molmo) QA converter.

Design brief: issue #41 (https://github.com/remyxai/VQASynth/issues/41) — sample
frames from Ego4D-style multi-view sources, extract point-level correspondences
between views, and turn them into training data for a pointing VLM (Molmo).

Method chosen: **OpenCV classical** (SIFT keypoints + ratio-tested BFMatcher +
RANSAC homography filter). CPU-only, no model weights, no GPU — matches the
lightweight data-pipeline stage shape the rest of VQASynth ships
(``docker/*_stage``). Well suited to Ego4D-style adjacent-frame matching where
the viewpoint change is modest. The neural alternatives cited in the brief
(StreamVGGT — arXiv:2507.11539; PlanarRecon — arXiv:2104.00681) trade CPU
lightness for accuracy on large viewpoint changes and are referenced in the PR
body's discovery notes rather than implemented here.

The QA converter emits Molmo ``<point x=".." y=".." alt="..">`` tags in the
**exact format parsed by** ``vqasynth.localize.extract_points_and_descriptions``
— coordinates normalized to Molmo's 0–100 space — so correspondence outputs
drop straight into the existing pointing-VLM training pipeline.

Heavy deps (cv2, numpy, PIL) are imported lazily inside the extractor methods,
mirroring ``experiments/nooa_agent/tools/depth.py``. That keeps the converter
+ data structures importable for the structural tests without those packages
installed.

Scope (per the brief — one method shipped, others referenced, not implemented):
  - IN: OpenCV correspondence stage + Molmo <point> QA converter + docker
    stage wrapper + runnable example + structural tests.
  - OUT: semantic captioning of correspondences (raw SIFT matches carry no
    object label; the converter accepts a per-match ``caption`` so a downstream
    Molmo/Florence localizer can supply one — default is "feature"); wiring
    into the default ``pipelines/spatialvqa.yaml`` chain (it's a standalone
    multi-view stage with different inputs); the StreamVGGT/PlanarRecon neural
    backends (cited above).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Molmo's pointing coordinate space is normalized to [0, 100]; this is what
# ``vqasynth.localize.extract_points_and_descriptions`` multiplies back into
# pixels. Pixel coords passed to the converter must be rescaled into this
# range before rendering a <point> tag.
_MOLMO_NORM_MAX = 100.0

# Tag form intentionally ends with ">" (not "/>"). The regex in
# vqasynth.localize.extract_points_and_descriptions is
#   <point\s+x="..."\s+y="..."\s+alt="([^"]+)">
# which requires the quote to be immediately followed by ">" — the
# self-closing "/>" form is NOT matched by that parser. We emit the
# parser-compatible form so outputs round-trip through the existing pipeline.
_POINT_TAG_TEMPLATE = '<point x="{x}" y="{y}" alt="{alt}">'


# ────────────────────────────────────────────────────────────────────────
# Coordinate / tag helpers (pure Python — no cv2 / numpy / PIL needed)
# ────────────────────────────────────────────────────────────────────────

def to_molmo_coord(value_px: float, extent_px: float) -> float:
    """Pixel coordinate -> Molmo's 0–100 normalized space, clamped + rounded.

    Mirrors the inverse of the pixel conversion in
    ``localize.extract_points_and_descriptions`` (``x_pixel = x_norm/100 * W``).
    """
    extent_px = float(extent_px)
    if extent_px <= 0:
        raise ValueError(f"extent_px must be positive, got {extent_px}")
    norm = (float(value_px) / extent_px) * _MOLMO_NORM_MAX
    # Clamp to [0, 100]; localize rejects tags whose max coord exceeds 100.
    norm = max(0.0, min(_MOLMO_NORM_MAX, norm))
    return round(norm, 2)


def point_to_molmo_tag(
    x_px: float,
    y_px: float,
    width: float,
    height: float,
    alt: str = "point",
) -> str:
    """Render a single pixel point as a Molmo ``<point ...>`` tag.

    ``alt`` must be non-empty — ``localize``'s parser requires ``[^"]+`` and
    silently drops empty-alt tags.
    """
    if not alt:
        alt = "point"
    x_norm = to_molmo_coord(x_px, width)
    y_norm = to_molmo_coord(y_px, height)
    return _POINT_TAG_TEMPLATE.format(x=x_norm, y=y_norm, alt=alt)


def _substitute(template: str, **slots: Any) -> str:
    """Bracket-token substitution in the repo's template convention.

    VQASynth's prompt templates (see ``vqasynth.prompt_templates``) use
    ``[A]`` / ``[B]`` / ``[X]`` placeholders. This helper replaces any
    ``[KEY]`` token whose KEY is passed as a kwarg, so correspondence
    templates compose with the existing template vocabulary.
    """
    out = template
    for key, val in slots.items():
        out = out.replace(f"[{key}]", str(val))
    return out


# Default QA phrasing for a forward correspondence (point in view A → point in
# view B). Tokens: [SRC] = <point> in view A, [DST] = <point> in view B,
# [ALT] = shared caption describing the matched feature.
_DEFAULT_FORWARD_QUESTION_TEMPLATES = [
    "[SRC] marks the [ALT] in the first image. Where is the same [ALT] in the second image?",
    "[SRC] indicates the [ALT] in view 1. Point to it in view 2.",
    "Given [SRC] labels the [ALT] in the first view, identify the matching [ALT] in the second view.",
]

_DEFAULT_FORWARD_ANSWER_TEMPLATES = [
    "[DST]",
    "The same [ALT] is at [DST] in the second image.",
    "[DST]",
]


# ────────────────────────────────────────────────────────────────────────
# Result data structures
# ────────────────────────────────────────────────────────────────────────

@dataclass
class CorrespondenceMatch:
    """A single matched point pair across two views."""

    pt_a: tuple[float, float]        # (x, y) pixel coords in view A
    pt_b: tuple[float, float]        # (x, y) pixel coords in view B
    caption: str = "feature"          # shared description of the matched feature
    distance: float | None = None     # descriptor distance (lower = better)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorrespondenceResult:
    """Output of :meth:`CorrespondenceExtractor.extract`."""

    matches: list[CorrespondenceMatch]
    view_a_size: tuple[int, int]               # (W, H) of view A
    view_b_size: tuple[int, int]               # (W, H) of view B
    homography: list[float] | None = None      # 3x3 H (view A -> B), row-major, if recovered
    inlier_count: int = 0                       # matches surviving RANSAC
    backend: str = "sift-bf"

    def __repr__(self) -> str:
        # Default dataclass repr would dump every match tuple — fine here, but
        # keep it compact for logs / trace events (cf. DepthResult in
        # experiments/nooa_agent/tools/depth.py).
        return (
            f"CorrespondenceResult(backend={self.backend!r}, "
            f"matches={len(self.matches)}, inliers={self.inlier_count}, "
            f"view_a={self.view_a_size}, view_b={self.view_b_size}, "
            f"has_homography={self.homography is not None})"
        )


# ────────────────────────────────────────────────────────────────────────
# QA converter (pure Python — no cv2 / numpy / PIL needed)
# ────────────────────────────────────────────────────────────────────────

def build_qa_pair(
    match: CorrespondenceMatch,
    view_a_size: tuple[int, int],
    view_b_size: tuple[int, int],
    question_template: str | None = None,
    answer_template: str | None = None,
    alt: str | None = None,
) -> tuple[str, str]:
    """Render one correspondence as a (question, answer) string pair.

    The answer is a Molmo ``<point>`` tag in view B's 0–100 normalized space.
    """
    wa, ha = view_a_size
    wb, hb = view_b_size
    caption = alt if alt is not None else match.caption
    src_tag = point_to_molmo_tag(match.pt_a[0], match.pt_a[1], wa, ha, caption)
    dst_tag = point_to_molmo_tag(match.pt_b[0], match.pt_b[1], wb, hb, caption)

    q_tpl = question_template or _DEFAULT_FORWARD_QUESTION_TEMPLATES[0]
    a_tpl = answer_template or _DEFAULT_FORWARD_ANSWER_TEMPLATES[0]
    slots = {"SRC": src_tag, "DST": dst_tag, "ALT": caption}
    return _substitute(q_tpl, **slots), _substitute(a_tpl, **slots)


def build_qa_message(
    match: CorrespondenceMatch,
    view_a_size: tuple[int, int],
    view_b_size: tuple[int, int],
    **kwargs: Any,
) -> dict:
    """One standalone QA row in the dataset message schema.

    Schema matches ``docker/prompt_stage/process_prompts.py`` /
    ``vqasynth.prompts.PromptGenerator.create_messages_from_prompts``:
    ``{"role", "content": [{"index", "text", "type"}]}``. Multi-view — image
    index 0 is view A, index 1 is view B.
    """
    question, answer = build_qa_pair(match, view_a_size, view_b_size, **kwargs)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"index": 0, "text": None, "type": "image"},   # view A
                    {"index": 1, "text": None, "type": "image"},   # view B
                    {"index": None, "text": question, "type": "text"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"index": None, "text": answer, "type": "text"}],
            },
        ]
    }


def correspondences_to_point_qa(
    result: CorrespondenceResult,
    max_turns: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Turn a result's matches into standalone single-turn QA rows."""
    matches = result.matches if max_turns is None else result.matches[:max_turns]
    return [
        build_qa_message(m, result.view_a_size, result.view_b_size, **kwargs)
        for m in matches
    ]


def correspondences_to_messages(
    result: CorrespondenceResult,
    max_turns: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Assemble matches into ONE multi-turn conversation (the dataset row shape).

    Mirrors ``PromptGenerator.create_messages_from_prompts``: the two view
    images are attached only to the first user turn; later turns are text-only.
    """
    matches = result.matches if max_turns is None else result.matches[:max_turns]
    messages: list[dict] = []
    first = True
    for m in matches:
        question, answer = build_qa_pair(m, result.view_a_size, result.view_b_size, **kwargs)
        if first:
            messages.append({
                "role": "user",
                "content": [
                    {"index": 0, "text": None, "type": "image"},
                    {"index": 1, "text": None, "type": "image"},
                    {"index": None, "text": question, "type": "text"},
                ],
            })
            first = False
        else:
            messages.append({
                "role": "user",
                "content": [{"index": None, "text": question, "type": "text"}],
            })
        messages.append({
            "role": "assistant",
            "content": [{"index": None, "text": answer, "type": "text"}],
        })
    return messages


# ────────────────────────────────────────────────────────────────────────
# Lazy dependency loaders
# ────────────────────────────────────────────────────────────────────────

def _require_cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - exercised only without cv2
        raise ImportError(
            "opencv-python is required for CorrespondenceExtractor. It is "
            "already in VQASynth's requirements (opencv-python==4.8.1.78); "
            "install it with `pip install opencv-python`."
        ) from e
    return cv2


def _require_numpy():
    import numpy as np
    return np


# ────────────────────────────────────────────────────────────────────────
# Correspondence extractor (OpenCV classical)
# ────────────────────────────────────────────────────────────────────────

class CorrespondenceExtractor:
    """SIFT + BFMatcher (Lowe ratio test) + RANSAC homography filter.

    CPU-only, no model weights. Picks up the same "lightweight data-pipeline
    stage" shape as the other ``docker/*_stage`` stages.

    Args:
        detector: ``"sift"`` (default, float descriptors) or ``"orb"``
            (binary descriptors, faster, patent-free historically).
        matcher: ``"bf"`` (brute-force, default) or ``"flann"`` (approximate
            nearest-neighbour, faster on very large keypoint sets).
        ratio: Lowe's ratio-test threshold (default 0.75). Lower = stricter.
        min_match_count: minimum raw good matches before attempting RANSAC
            and before emitting any correspondence (default 4).
        max_features: cap on detected keypoints per image (default 2000).
        ransac_thresh: RANSAC reprojection threshold in pixels (default 5.0).
    """

    def __init__(
        self,
        detector: str = "sift",
        matcher: str = "bf",
        ratio: float = 0.75,
        min_match_count: int = 4,
        max_features: int = 2000,
        ransac_thresh: float = 5.0,
    ):
        if detector not in ("sift", "orb"):
            raise ValueError(f"detector must be 'sift' or 'orb', got {detector!r}")
        if matcher not in ("bf", "flann"):
            raise ValueError(f"matcher must be 'bf' or 'flann', got {matcher!r}")
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        self.detector_name = detector
        self.matcher_name = matcher
        self.ratio = float(ratio)
        self.min_match_count = int(min_match_count)
        self.max_features = int(max_features)
        self.ransac_thresh = float(ransac_thresh)
        self.backend = f"{detector}-{matcher}"
        self._detector = None
        self._matcher = None

    # -- lazy loading ----------------------------------------------------
    def _ensure_loaded(self) -> None:
        cv2 = _require_cv2()
        if self._detector is None:
            if self.detector_name == "sift":
                self._detector = cv2.SIFT_create(nfeatures=self.max_features)
            else:
                self._detector = cv2.ORB_create(nfeatures=self.max_features)
        if self._matcher is None:
            if self.matcher_name == "bf":
                norm = cv2.NORM_L2 if self.detector_name == "sift" else cv2.NORM_HAMMING
                self._matcher = cv2.BFMatcher(norm)
            else:
                # FLANN params: L2 (KD-tree) for SIFT, Hamming (LSH) for ORB.
                if self.detector_name == "sift":
                    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
                else:
                    index_params = dict(
                        algorithm=6, table_number=6, key_size=12, multi_probe_level=1
                    )  # FLANN_INDEX_LSH
                search_params = dict(checks=50)
                self._matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # -- image normalization --------------------------------------------
    def _to_gray(self, image: Any) -> tuple[Any, tuple[int, int]]:
        """Accept a PIL image or an (H, W, 3) ndarray; return (gray, (W, H)).

        Treats ndarrays as RGB to match PIL's convention (the rest of VQASynth
        converts via ``image.convert("RGB")`` before handing arrays to cv2).
        """
        cv2 = _require_cv2()
        np = _require_numpy()
        if hasattr(image, "convert") and hasattr(image, "size"):  # PIL.Image
            w, h = image.size
            arr = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            return gray, (int(w), int(h))
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError("expected an RGB image (H, W, 3), got shape {}".format(arr.shape))
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return gray, (int(w), int(h))

    # -- core ------------------------------------------------------------
    def detect(self, image: Any):
        """Return (keypoints, descriptors, (W, H)) for one image."""
        self._ensure_loaded()
        gray, size = self._to_gray(image)
        kp, des = self._detector.detectAndCompute(gray, None)
        return kp, des, size

    def extract(self, image_a: Any, image_b: Any) -> CorrespondenceResult:
        """Extract geometrically-consistent point correspondences A -> B.

        Returns a :class:`CorrespondenceResult` whose ``matches`` survived the
        Lowe ratio test AND the RANSAC inlier mask (when a homography could be
        recovered). ``homography`` is the recovered 3x3 map from view A to
        view B, row-major flattened, or ``None`` if too few matches.
        """
        self._ensure_loaded()
        cv2 = _require_cv2()
        np = _require_numpy()

        kp_a, des_a, size_a = self.detect(image_a)
        kp_b, des_b, size_b = self.detect(image_b)

        empty = CorrespondenceResult(
            matches=[], view_a_size=size_a, view_b_size=size_b,
            homography=None, inlier_count=0, backend=self.backend,
        )
        if (
            des_a is None or des_b is None
            or len(des_a) < self.min_match_count
            or len(des_b) < self.min_match_count
        ):
            return empty

        knn = self._matcher.knnMatch(des_a, des_b, k=2)
        good = []
        for pair in knn:
            if len(pair) >= 2:
                m, n = pair[0], pair[1]
                if m.distance < self.ratio * n.distance:
                    good.append(m)

        if len(good) < self.min_match_count:
            return empty

        # RANSAC filter — keeps only matches consistent with a single global
        # A->B transform. Drops spurious matches that survive the ratio test
        # but don't share a coherent geometry (repeats, occluders).
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, self.ransac_thresh)
        if mask is not None:
            keep = mask.ravel().astype(bool)
            good = [m for m, k in zip(good, keep) if k]

        matches = [
            CorrespondenceMatch(
                pt_a=(float(kp_a[m.queryIdx].pt[0]), float(kp_a[m.queryIdx].pt[1])),
                pt_b=(float(kp_b[m.trainIdx].pt[0]), float(kp_b[m.trainIdx].pt[1])),
                caption="feature",
                distance=float(m.distance),
            )
            for m in good
        ]
        h_flat = None
        if H is not None:
            h_flat = [float(v) for row in H for v in row]
        return CorrespondenceResult(
            matches=matches,
            view_a_size=size_a,
            view_b_size=size_b,
            homography=h_flat,
            inlier_count=len(matches),
            backend=self.backend,
        )

    # -- HF datasets integration ----------------------------------------
    def apply_transform(self, example: dict, images: str) -> dict:
        """``datasets.map`` transform: pair adjacent views, emit QA messages.

        Treats ``example[images]`` as a list of PIL images — the natural shape
        for an Ego4D-style clip (multiple frames per example). Adjacent frame
        pairs ``(v_i, v_{i+1})`` are matched; their correspondences are folded
        into a single multi-turn conversation on ``example["messages"]``.

        A single image (no list) is a no-op that yields no correspondences —
        correspondences require at least two views by definition.
        """
        views = self._coerce_views(example.get(images))
        all_messages: list[dict] = []
        for i in range(len(views) - 1):
            result = self.extract(views[i], views[i + 1])
            all_messages.extend(
                correspondences_to_messages(result, max_turns=8)
            )
        example["messages"] = all_messages
        return example

    @staticmethod
    def _coerce_views(raw: Any) -> list:
        """Normalize the image column into a flat list of RGB PIL images."""
        # Duck-type PIL (avoid importing PIL at module load for the tests).
        def _is_pil(v):
            return hasattr(v, "convert") and hasattr(v, "size")

        if _is_pil(raw):
            views = [raw]
        elif isinstance(raw, list):
            views = []
            for v in raw:
                if isinstance(v, list) and v and _is_pil(v[0]):
                    views.append(v[0])  # HF batched nesting: take first view
                elif _is_pil(v):
                    views.append(v)
        else:
            views = []

        out = []
        for v in views:
            if v.mode != "RGB":
                v = v.convert("RGB")
            out.append(v)
        return out


__all__ = [
    "CorrespondenceMatch",
    "CorrespondenceResult",
    "CorrespondenceExtractor",
    "to_molmo_coord",
    "point_to_molmo_tag",
    "build_qa_pair",
    "build_qa_message",
    "correspondences_to_point_qa",
    "correspondences_to_messages",
]
