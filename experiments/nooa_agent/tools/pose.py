"""Body-keypoint pose tool wrapper for the SpatialAnnotator agent.

Wraps :class:`vqasynth.pose.KeypointExtractor` (PR #132) as a NOOA agent tool,
so the SpatialAnnotator can call "give me this person's keypoints" as a discrete
per-turn step — instead of only through the batch Docker stage
(``docker/pose_stage/``). The detector tier (:mod:`florence`) can already surface
a person box; this tool turns that box (or the whole image) into per-joint pixel
coordinates the agent can reason about ("which body part is pointing where").

Same tool ABI as :mod:`experiments.nooa_agent.tools.orientation` (PR #134) and
:mod:`depth` / :mod:`florence`: compact-``__repr__`` result dataclasses + a
backend class with lazy imports + a module-level singleton + one standalone
tool function. We do NOT reimplement pose extraction — :class:`KeypointExtractor`
already has a pluggable backend ABI; :class:`PoseEstimator` composes it and lifts
its pose dicts into the agent-facing :class:`PoseResult`.

Single tier, MediaPipe default: MediaPipe Pose is CPU-friendly and Apache-2.0,
matching the same-tier decision the orientation tool makes (Orient-Anything also
runs on CPU). The ``backend`` field on :class:`PoseResult` is ``"mediapipe"`` so
a future backend (e.g. a GPU YOLOv8-pose wrapper) can be slotted in without
changing the tool surface.

Licensing note (per PR #132): MediaPipe is Apache-2.0 and is the default.
YOLOv8-pose (``ultralytics``) is AGPL-3.0 and is **opt-in only** — it is never
selected by name and is not added as a runtime dependency here. ``ultralytics``
is not imported; selecting ``"yolov8"`` surfaces the opt-in refusal from
:func:`vqasynth.pose._resolve_backend` so callers who accept that license shape
pass a backend instance explicitly.

Heavy imports stay inside methods — mirror :mod:`orientation` / :mod:`depth` so
importing this module doesn't drag in ``mediapipe`` (or ``torch``) and break the
NOOA test ABI on a host without them. ``vqasynth.pose`` is imported lazily for
the same reason, and constructing the default backend (which is what triggers the
``mediapipe`` import) is deferred to the first ``extract`` call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Reuse the same dtype alias resolver as the other tools so calling conventions
# match across the whole tool surface (see florence.py / depth.py / orientation.py).
from experiments.nooa_agent.tools.florence import _resolve_torch_dtype


@dataclass
class Keypoint:
    """A single detected body joint.

    Fields:
        name: canonical COCO-17 joint name (e.g. ``"nose"``, ``"left_shoulder"``).
        x: pixel x-coordinate in the source image.
        y: pixel y-coordinate in the source image.
        confidence: 0..1 detection confidence. The underlying
            :class:`vqasynth.pose.MediaPipePoseBackend` pre-filters joints below
            its ``min_visibility`` / ``min_detection_confidence`` thresholds and
            only surfaces a binary visible flag, so a joint that reaches this
            layer is reported at full confidence (``1.0``). Fractional scores
            would require extending the backend ABI and are intentionally not
            invented here.
    """
    name: str
    x: float
    y: float
    confidence: float


@dataclass
class PoseResult:
    """Per-person body-keypoint pose estimate.

    Fields:
        keypoints: one :class:`Keypoint` per detected (visible) joint. A scene
            may report anywhere from 0 to 17 COCO joints per person; undetected
            joints are omitted rather than reported at a fabricated location.
        person_bbox: inferred bounding box of the pose — the min/max envelope of
            the visible keypoints — as ``(x1, y1, x2, y2)`` integers in pixel
            coordinates (same shape as :class:`florence.Box`). When
            :func:`pose_keypoints` is called with a ``bbox``, this reflects the
            supplied region rather than the inferred envelope. ``None`` when no
            joints were detected.
        backend: ``"mediapipe"`` for the default backend; leaves room for a
            future backend (e.g. ``"yolov8_pose"``) without changing the surface.
    """
    keypoints: list[Keypoint]
    person_bbox: tuple[int, int, int, int] | None
    backend: str = "mediapipe"

    def __repr__(self) -> str:
        # Compact one-liner. A person carries 17-33 joints; a default-dataclass
        # repr would enumerate every Keypoint and blow up NOOA trace lines (one
        # trace event fires per tool call, and a scene may have several people).
        # The per-joint data remains accessible via ``keypoints``; this repr is
        # the trace-readable summary — mirrors ``DepthResult.__repr__``'s
        # "don't dump the array" guard and ``OrientationResult.__repr__``.
        return (
            f"PoseResult(backend={self.backend!r}, "
            f"n_keypoints={len(self.keypoints)}, "
            f"bbox={self.person_bbox})"
        )


def _backend_name(backend: Any) -> str:
    """Display name for the pose backend used in :attr:`PoseResult.backend`.

    Maps the ``backend`` argument :class:`PoseEstimator` accepts — ``None``, a
    name string, or a :class:`vqasynth.pose.PoseBackend` instance — onto the
    short string surfaced on each :class:`PoseResult`. ``None`` and
    ``"mediapipe"`` both report ``"mediapipe"`` (the default); an instance is
    derived from its class name (``MediaPipePoseBackend`` → ``"mediapipe"``,
    ``StubPoseBackend`` → ``"stub"``).
    """
    if isinstance(backend, str):
        return backend if backend else "mediapipe"
    if backend is None:
        return "mediapipe"
    cls = type(backend).__name__.lower()
    for suffix in ("posebackend", "backend"):
        if cls.endswith(suffix):
            cls = cls[: -len(suffix)]
            break
    return cls or "pose"


def _result_from_pose(pose: dict, *, backend: str) -> PoseResult:
    """Lift a ``vqasynth.pose`` pose dict into an agent-facing :class:`PoseResult`.

    Keeps only the joints the backend reported visible (``xy is not None``) and
    infers ``person_bbox`` as their min/max envelope. Keypoints stay in the
    coordinate space of the image the backend was run on; the crop path in
    :func:`pose_keypoints` remaps them into full-image space afterwards.
    """
    keypoints: list[Keypoint] = []
    for joint in pose.get("keypoints", []):
        xy = joint.get("xy") if isinstance(joint, dict) else None
        if not xy:
            continue
        keypoints.append(Keypoint(
            name=joint["name"],
            x=float(xy[0]),
            y=float(xy[1]),
            confidence=1.0,
        ))

    person_bbox: tuple[int, int, int, int] | None = None
    if keypoints:
        xs = [kp.x for kp in keypoints]
        ys = [kp.y for kp in keypoints]
        person_bbox = (
            int(round(min(xs))), int(round(min(ys))),
            int(round(max(xs))), int(round(max(ys))),
        )
    return PoseResult(keypoints=keypoints, person_bbox=person_bbox, backend=backend)


def _shift_to_full_image(
    result: PoseResult, *, offset: tuple[int, int], person_bbox: tuple[int, int, int, int]
) -> PoseResult:
    """Remap a crop-space :class:`PoseResult` back into full-image coordinates.

    The pose backend reports keypoints in the pixel space of the image it was
    given; when :func:`pose_keypoints` cropped to a region first, those
    coordinates are crop-local. Shifting them by the crop origin lines them up
    with full-image detections (Florence boxes, depth maps). ``person_bbox`` is
    set to the supplied region per the tool contract.
    """
    ox, oy = offset
    shifted = [
        Keypoint(name=kp.name, x=kp.x + ox, y=kp.y + oy, confidence=kp.confidence)
        for kp in result.keypoints
    ]
    return PoseResult(keypoints=shifted, person_bbox=person_bbox, backend=result.backend)


class PoseEstimator:
    """Backend holding the pose keypoint model.

    Composes :class:`vqasynth.pose.KeypointExtractor` internally — it does NOT
    reimplement keypoint extraction or the COCO-17 skeleton (those live in
    :mod:`vqasynth.pose`). This class adds the agent-facing :class:`PoseResult`
    lifting on top of the extractor's pose dicts and provides the module-level
    singleton used by :func:`pose_keypoints`.

    Args:
        backend: pose backend selector, passed straight through to
            :class:`KeypointExtractor`. ``None`` (default) and ``"mediapipe"``
            select the CPU-friendly Apache-2.0 MediaPipe backend; a
            :class:`vqasynth.pose.PoseBackend` instance is used as-is (handy for
            the :class:`vqasynth.pose.StubPoseBackend` in tests, or to wrap a
            custom/YOLOv8 backend). ``"yolov8"`` is refused by
            :func:`vqasynth.pose._resolve_backend` (AGPL-3.0, opt-in only).
        model: advisory injection slot, accepted for ABI symmetry with the other
            tool backends (:class:`~experiments.nooa_agent.tools.orientation.OrientAnythingEstimator`
            takes a ``model=``). MediaPipe constructs its own model, so this is
            unused by the default backend; a future GPU backend could consume it.
        device: advisory accelerator selection (e.g. ``"cuda:1"``). MediaPipe is
            CPU-only, so this is ignored by the default backend; kept for
            symmetry with the other tools and a future GPU backend.
        dtype: torch dtype or ``fp32``/``fp16``/``bf16`` string alias, resolved
            via :func:`_resolve_torch_dtype`. Advisory — MediaPipe ignores
            precision; resolving it here keeps a bad alias loud and leaves a
            future GPU backend the resolved value ready.

    ``model`` / ``device`` / ``dtype`` are deliberately advisory on the default
    MediaPipe path (mirroring how the orientation tool's device/dtype are
    advisory because the underlying estimator picks its own placement); they
    exist so a GPU backend can be slotted in without reshaping the constructor.
    """

    BACKEND = "mediapipe"

    def __init__(self, backend=None, model=None, device=None, dtype: Any = None):
        self.device = device
        self.dtype = dtype
        self._model = model
        self._backend_arg = backend
        self.backend_name = _backend_name(backend)
        # Constructed lazily so importing this module never triggers a mediapipe
        # import or model load.
        self._extractor = None

    def _ensure_loaded(self):
        if self._extractor is not None:
            return
        from vqasynth.pose import KeypointExtractor

        # Resolve the dtype alias once — validates fp32/fp16/bf16 strings the
        # same way the other tools do. Advisory for inference precision today
        # (see class docstring); resolving it here keeps bad aliases loud.
        _resolve_torch_dtype(self.dtype)

        resolved = self._backend_arg if self._backend_arg is not None else "mediapipe"
        self._extractor = KeypointExtractor(backend=resolved)

    def extract(self, image) -> list[PoseResult]:
        """Run pose extraction over ``image``; return one :class:`PoseResult` per person.

        Args:
            image: an RGB ``PIL.Image`` (use :func:`pose_keypoints` to crop to a
                specific person region first).

        Returns:
            One :class:`PoseResult` per detected person, in the coordinate space
            of ``image``. Empty list if no person was detected.
        """
        self._ensure_loaded()
        poses = self._extractor.extract(image)
        return [_result_from_pose(pose, backend=self.backend_name) for pose in poses]


# Module-level singleton so N tool calls in one agent session share the same
# loaded pose backend — constructing MediaPipe's Pose solution is the expensive
# part and must not repeat per call. Mirrors the singleton pattern the other
# tool backends use.
_DEFAULT_ESTIMATOR: PoseEstimator | None = None


def _get_default_estimator() -> PoseEstimator:
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None:
        _DEFAULT_ESTIMATOR = PoseEstimator()
    return _DEFAULT_ESTIMATOR


def pose_keypoints(image, *, bbox=None) -> list[PoseResult]:
    """Extract per-person body keypoints from the image.

    Wraps the batch-stage pose extractor as a discrete NOOA tool step: the
    SpatialAnnotator can ask "where are this person's joints?" inline rather
    than dropping into ``docker/pose_stage/``. Useful after
    :func:`~experiments.nooa_agent.tools.florence.FlorenceDetector.detect_objects`
    surfaces a person box, to ground pose claims ("the left arm is extended to
    the right") in real joint coordinates.

    Args:
        image: RGB ``PIL.Image``.
        bbox: optional ``(x1, y1, x2, y2)`` pixel region (matches Florence's
            ``Box`` shape). When given, detection is restricted to that crop —
            useful when the agent already has a single detected person box and
            wants pose for just that person; the returned keypoints are remapped
            into full-image coordinates and ``person_bbox`` reflects the supplied
            region. When ``None`` (default), pose runs on the whole image and one
            :class:`PoseResult` is returned per detected person.

    Returns:
        One :class:`PoseResult` per detected person (possibly several when
        ``bbox`` is ``None``; typically one when ``bbox`` targets a single
        person). Empty list if no person was detected.

    Raises:
        ValueError: if ``image`` is not a :class:`PIL.Image.Image`, or ``bbox``
            is not a 4-element sequence.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected a PIL image but got {type(image)}")

    estimator = _get_default_estimator()

    if bbox is None:
        return estimator.extract(image)

    bbox = tuple(bbox)
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (x1, y1, x2, y2); got {bbox!r}")
    x1, y1, x2, y2 = bbox
    # Florence's Box uses floats; PIL.crop rounds to integer pixel coordinates.
    ix1, iy1, ix2, iy2 = (
        int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    )
    crop = image.crop((ix1, iy1, ix2, iy2))
    input_bbox = (ix1, iy1, ix2, iy2)

    raw = estimator.extract(crop)
    # Keypoints come back in crop-local coords; shift them into full-image space
    # so they line up with the agent's other full-image detections, and report
    # person_bbox as the supplied region per the tool contract.
    return [
        _shift_to_full_image(r, offset=(ix1, iy1), person_bbox=input_bbox)
        for r in raw
    ]


__all__ = [
    "Keypoint",
    "PoseResult",
    "PoseEstimator",
    "pose_keypoints",
]
