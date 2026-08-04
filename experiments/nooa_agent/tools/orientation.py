"""Object-level orientation tool wrapper for the SpatialAnnotator agent.

Wraps :class:`vqasynth.orientation.OrientationEstimator` (PR #121) as a NOOA
agent tool, so the SpatialAnnotator can call "orient this object" as a discrete
step in a dynamically-composed pipeline — instead of only through the batch
Docker stage (``docker/orientation_stage/``).

Single-tier tool for now: Orient-Anything runs on CPU too (slow), so unlike
:mod:`experiments.nooa_agent.tools.depth` (DepthPro/VGGT backend switch) there
is no per-tier backend split here. The ``backend`` field on
:class:`OrientationResult` is ``"orient_anything_v1"`` so a v2 backend can be
slotted in later without changing the tool surface.

Heavy imports stay inside methods — mirror :mod:`depth` / :mod:`florence` so
importing this module doesn't drag in ``torch`` / ``transformers`` / the
Orient-Anything repo and break the NOOA test ABI on a host without CUDA or
weights. ``vqasynth.orientation`` is imported lazily for the same reason (it
imports torch at its own module top level).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Reuse the same dtype alias resolver as the other tools so calling
# conventions match across the whole tool surface (see depth.py).
from experiments.nooa_agent.tools.florence import _resolve_torch_dtype


# Eight-point compass labels for the azimuth→prose mapping. Orient-Anything's
# azimuth is a 0..359 heading; we bucket it to a familiar compass word so the
# agent gets something quotable ("facing east") rather than a bare degree.
_COMPASS = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
]


def _describe_orientation(
    azimuth: float, polar: float, rotation: float, confidence: float
) -> str:
    """Natural-language summary of an orientation estimate.

    Deterministic given the four scalars — unit-tested separately from the
    estimator so the prose mapping is pinned independent of model behavior.
    Deliberately qualitative (no raw degree numbers): the exact values remain
    on the :class:`OrientationResult` fields; this string exists so the agent
    can quote the result in prose.
    """
    heading = _COMPASS[int(round(azimuth / 45.0)) % 8]

    # Polar is an elevation angle: -90 (straight down) .. 89 (up).
    if polar <= -60:
        tilt = "steeply downward"
    elif polar < -15:
        tilt = "slightly downward"
    elif polar <= 15:
        tilt = "level"
    elif polar < 60:
        tilt = "slightly upward"
    else:
        tilt = "steeply upward"

    # Rotation is in-plane roll. Normalize to [-180, 180) so the bucketing is
    # symmetric; positive ≈ clockwise (roll right).
    rot = ((rotation + 180.0) % 360.0) - 180.0
    if rot <= -60:
        roll = "rolled hard left"
    elif rot < -15:
        roll = "slight roll left"
    elif rot <= 15:
        roll = "upright"
    elif rot < 60:
        roll = "slight roll right"
    else:
        roll = "rolled hard right"

    if confidence >= 0.75:
        conf = "high confidence"
    elif confidence >= 0.4:
        conf = "moderate confidence"
    else:
        conf = "low confidence"

    return f"facing {heading}, {tilt}, {roll}, {conf}"


def _result_from_angles(
    angles: dict, backend: str = "orient_anything_v1"
) -> "OrientationResult":
    """Lift ``vqasynth.orientation``'s dict into an :class:`OrientationResult`.

    Adds the natural-language ``description`` on top of the raw scalars the
    underlying estimator returns.
    """
    azimuth = float(angles["azimuth"])
    polar = float(angles["polar"])
    rotation = float(angles["rotation"])
    confidence = float(angles["confidence"])
    return OrientationResult(
        azimuth_deg=azimuth,
        polar_deg=polar,
        rotation_deg=rotation,
        confidence=confidence,
        description=_describe_orientation(azimuth, polar, rotation, confidence),
        backend=backend,
    )


@dataclass
class OrientationResult:
    """Object-level 3D orientation estimate from Orient-Anything.

    Fields:
        azimuth_deg: 0..359 — compass heading the object faces.
        polar_deg: -90..89 — vertical tilt (-90 = straight down, 0 = level,
            89 = up). The full range is covered (PR #121 fixed an earlier
            -180..-1 truncation in the rotation head).
        rotation_deg: -180..179 — in-plane roll about the camera axis.
        confidence: 0..1 — Orient-Anything's in-distribution probability.
            Low values mean the crop was likely off-distribution (cluttered,
            multi-object, or non-rigid) and the estimate should be distrusted.
        description: one-line natural-language summary the agent can quote
            in prose (e.g. "facing east, slightly upward, slight roll right,
            high confidence").
        backend: ``"orient_anything_v1"`` — leaves room for a v2 backend.
    """
    azimuth_deg: float
    polar_deg: float
    rotation_deg: float
    confidence: float
    description: str
    backend: str = "orient_anything_v1"

    def __repr__(self) -> str:
        # Compact one-liner. The raw floats already live on the named fields
        # and the ``description`` field carries the trace-readable view, so we
        # don't re-emit azimuth/polar/rotation here — mirrors
        # ``DepthResult.__repr__``'s "don't dump the array" guard. A NOOA trace
        # event fires per tool call; a verbose repr scales trace size with the
        # number of orient_object calls in a session.
        return (
            f"OrientationResult(backend={self.backend!r}, "
            f"confidence={self.confidence:.2f}, "
            f"description={self.description!r})"
        )


class OrientAnythingEstimator:
    """Backend holding the Orient-Anything orientation model.

    Composes :class:`vqasynth.orientation.OrientationEstimator` internally —
    it does NOT reimplement :func:`crop_to_object` / :func:`decode_angles`
    or the bin-count constants (those are imported from
    :mod:`vqasynth.orientation`). This class adds the natural-language
    ``description`` synthesis on top of the underlying estimator's dict output
    and provides the module-level singleton used by :func:`orient_object`.

    ``device`` + ``dtype`` mirror the other tool backends so multi-GPU nodes
    can pin this model separately. ``dtype`` accepts a torch dtype or an
    ``fp32``/``fp16``/``bf16`` string alias (resolved via
    :func:`_resolve_torch_dtype`). It is *advisory* on the default-load path:
    :class:`vqasynth.orientation.OrientationEstimator` selects its own precision
    via ``pick_dtype()``, mirroring how :class:`VggtEstimator`'s device/dtype
    are advisory because ``SpatialSceneConstructor`` controls its own placement.
    """

    BACKEND = "orient_anything_v1"

    def __init__(
        self,
        model=None,
        preprocess=None,
        device=None,
        dtype: Any = None,
    ):
        self.device = device
        self.dtype = dtype
        # Stash the injection points; the underlying OrientationEstimator is
        # constructed lazily so importing this module never triggers a weight
        # download or an Orient-Anything repo import.
        self._model = model
        self._preprocess = preprocess
        self._estimator = None

    def _ensure_loaded(self):
        if self._estimator is not None:
            return
        from vqasynth.orientation import OrientationEstimator

        # Resolve the dtype alias once — validates fp32/fp16/bf16 strings the
        # same way the other tools do. Advisory for inference precision today
        # (see class docstring); resolving it here keeps bad aliases loud and
        # leaves a future dtype-aware load path the resolved value ready.
        _resolve_torch_dtype(self.dtype)

        self._estimator = OrientationEstimator(
            model=self._model,
            preprocess=self._preprocess,
            device=self.device,
        )

    def estimate(self, image) -> OrientationResult:
        """Estimate orientation for a single (ideally isolated) object image.

        Args:
            image: a cropped single-object ``PIL.Image`` (use
                :func:`orient_object` to produce that crop from a full scene).

        Returns:
            :class:`OrientationResult` with the three angles, an
            in-distribution confidence, and a natural-language description.
        """
        self._ensure_loaded()
        angles = self._estimator.run(image)
        return _result_from_angles(angles, backend=self.BACKEND)


# Module-level singleton so N tool calls in one agent session share the same
# loaded Orient-Anything weights — loading DINOv2-large + the orientation head
# is the expensive part and must not repeat per call. Mirrors the singleton
# pattern the other tool backends use.
_DEFAULT_ESTIMATOR: OrientAnythingEstimator | None = None


def _get_default_estimator() -> OrientAnythingEstimator:
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None:
        _DEFAULT_ESTIMATOR = OrientAnythingEstimator()
    return _DEFAULT_ESTIMATOR


def orient_object(image, *, bbox=None, mask=None) -> OrientationResult:
    """Estimate the 3D orientation of a single object in the image.

    Wraps the batch-stage orientation estimator as a discrete NOOA tool step:
    the SpatialAnnotator can ask "which way is this object facing?" inline
    rather than dropping into ``docker/orientation_stage/``.

    Orient-Anything is trained on isolated single-object renders and only
    generalizes in-the-wild when each object is cropped out of its scene first
    (the repo's stated "Best Practice"). Exactly ONE localization prompt is
    therefore required to reduce the image to a single-object crop before
    inference.

    Args:
        image: RGB ``PIL.Image`` containing the object.
        bbox: exactly one of ``bbox`` / ``mask``. ``bbox=(x1, y1, x2, y2)`` in
            pixel coordinates (matches Florence's ``Box`` shape) → direct crop
            of that region.
        mask: alternatively, an ``HxW`` ``uint8``/``bool`` array (matches SAM's
            mask output shape) → background whited out + cropped to the mask's
            bounding box via :func:`vqasynth.orientation.crop_to_object`.

    Returns:
        :class:`OrientationResult`.

    Raises:
        ValueError: if neither or both of ``bbox`` / ``mask`` are given.
    """
    from PIL import Image

    if (bbox is None) == (mask is None):
        raise ValueError(
            "orient_object requires exactly one of bbox= or mask= (got "
            + ("both" if bbox is not None else "neither")
            + ")"
        )
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected a PIL image but got {type(image)}")

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Direct crop to the box region. Florence's Box uses floats; PIL.crop
        # rounds to integer pixel coordinates.
        crop = image.crop(
            (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        )
    else:
        # Mask path: reuse crop_to_object so we get the same square + white-out
        # isolation the batch docker stage uses — do NOT reimplement it here.
        from vqasynth.orientation import crop_to_object
        crop = crop_to_object(image, mask)

    return _get_default_estimator().estimate(crop)


def orientation_delta(a: OrientationResult, b: OrientationResult) -> dict:
    """Signed orientation difference between two objects.

    Pure function (no model call). Use after two :func:`orient_object` calls
    to ground relative-orientation claims ("B is rotated 30° clockwise of A")
    in real geometry instead of guessing. Same shape as
    :func:`distance_3d_meters`: signed deltas + backend, plus one
    narrative-language field (``b_is``) the agent can quote.

    Args:
        a: first object's :class:`OrientationResult`.
        b: second object's :class:`OrientationResult`.

    Returns:
        Dict with ``azimuth_delta_deg`` / ``polar_delta_deg`` /
        ``rotation_delta_deg`` (wrapped to the minimal-magnitude rotation),
        a natural-language ``b_is`` summary, and ``backend``.
    """

    def _wrap(deg: float) -> float:
        # Wrap to [-180, 180) so the delta is the minimal rotation.
        return ((deg + 180.0) % 360.0) - 180.0

    d_az = _wrap(b.azimuth_deg - a.azimuth_deg)
    d_polar = b.polar_deg - a.polar_deg
    d_rot = _wrap(b.rotation_deg - a.rotation_deg)

    if d_az > 45:
        az_word = f"turned {d_az:.0f}° to the right of a"
    elif d_az < -45:
        az_word = f"turned {-d_az:.0f}° to the left of a"
    else:
        az_word = "facing roughly the same direction as a"

    return {
        "azimuth_delta_deg": round(d_az, 1),
        "polar_delta_deg": round(d_polar, 1),
        "rotation_delta_deg": round(d_rot, 1),
        "b_is": az_word,
        "backend": a.backend,
    }


__all__ = [
    "OrientationResult",
    "OrientAnythingEstimator",
    "orient_object",
    "orientation_delta",
]
