"""Regional captioning tool wrapper for the SpatialAnnotator agent.

Wraps :class:`vqasynth.describe_anything.DescribeAnything` (PR #130) as a NOOA
agent tool, so the SpatialAnnotator can call "describe this region in detail"
as a discrete step in a dynamically-composed pipeline — instead of only through
the batch Docker stage (``docker/describe_anything_stage/``).

SpatialAnnotator already produces SAM masks via
:class:`experiments.nooa_agent.tools.florence.FlorenceSegmenter`; this tool turns
each mask into a detailed regional caption from NVIDIA DAM — the natural next
step after "found the box, segmented it, now say what it is in more than one
word" (issue #51).

Single-tier tool: DAM is the one regional captioner, so unlike
:mod:`experiments.nooa_agent.tools.depth` (DepthPro/VGGT/FoundationGeo backend
switch) there is no per-tier backend split here. The ``backend`` field on
:class:`RegionCaption` is ``"dam_3b_self_contained"`` so a future variant can be
slotted in without changing the tool surface.

Heavy imports stay inside methods — mirror :mod:`depth` / :mod:`florence` /
:mod:`orientation` so importing this module doesn't drag in ``torch`` /
``transformers`` / DAM weights and break the NOOA test ABI on a host without
CUDA or weights. ``vqasynth.describe_anything`` is imported lazily for the same
reason (it imports torch at its own module top level).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Reuse the same dtype alias resolver as the other tools so calling conventions
# match across the whole tool surface (see florence.py).
from experiments.nooa_agent.tools.florence import _resolve_torch_dtype

# DAM captions can run a few hundred characters; a verbose repr would scale
# trace size with the number of describe_region calls (a NOOA trace event
# fires per tool call). Truncate the caption surfaced in __repr__ to this many
# characters — the full text always remains on RegionCaption.caption.
_REPR_CAPTION_LIMIT = 48


def _bbox_of_mask(mask) -> tuple[int, int, int, int]:
    """Tight ``(x1, y1, x2, y2)`` bounding box of a mask's non-zero pixels.

    Accepts every mask convention the pipeline produces — ``uint8`` ``0``/``255``
    (Localizer), ``bool`` (SAM2), and ``float`` in ``[0, 1]`` — matching the
    foreground rule :meth:`DescribeAnything._normalize_mask` uses, so the bbox
    we report is exactly the region DAM captioned.

    ``x1``/``y1`` are the first foreground col/row; ``x2``/``y2`` are one past
    the last (half-open), matching the (x1, y1, x2, y2) convention
    :class:`experiments.nooa_agent.tools.florence.Box` and ``PIL.crop`` use.
    Returns ``(0, 0, 0, 0)`` for an empty mask.
    """
    arr = np.asarray(mask)
    if arr.dtype == bool:
        foreground = arr
    elif np.issubdtype(arr.dtype, np.floating):
        foreground = arr > 0.5
    else:
        foreground = arr > 0
    cols = np.any(foreground, axis=0)
    rows = np.any(foreground, axis=1)
    if not cols.any():
        return (0, 0, 0, 0)
    x1 = int(np.argmax(cols))
    x2 = int(len(cols) - np.argmax(cols[::-1]))
    y1 = int(np.argmax(rows))
    y2 = int(len(rows) - np.argmax(rows[::-1]))
    return (x1, y1, x2, y2)


def _mask_from_bbox(bbox, width: int, height: int) -> np.ndarray:
    """Build a filled-rectangle ``uint8`` mask for an ``(x1, y1, x2, y2)`` box.

    DAM prefers mask prompts over box prompts for regional captioning (its
    README is explicit on this), so when the agent only has a box — e.g.
    straight off :meth:`florence.Box.to_list` — synthesize the mask DAM wants.
    The synthesized mask's bbox round-trips to the input box (coordinates are
    rounded to integer pixels and clamped to the image, matching how
    :func:`PIL.Image.crop` treats the box).
    """
    x1, y1, x2, y2 = bbox
    xi1 = max(0, min(width, int(round(x1))))
    yi1 = max(0, min(height, int(round(y1))))
    xi2 = max(0, min(width, int(round(x2))))
    yi2 = max(0, min(height, int(round(y2))))
    # Guard against degenerate / inverted boxes — DAM needs a non-empty mask.
    if xi2 <= xi1:
        xi2 = min(width, xi1 + 1)
    if yi2 <= yi1:
        yi2 = min(height, yi1 + 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[yi1:yi2, xi1:xi2] = 255
    return mask


@dataclass
class RegionCaption:
    """Detailed regional caption for one localized object from NVIDIA DAM.

    Fields:
        caption: DAM's detailed description of the masked region (can run a few
            hundred characters — that's the richness over a class label that
            motivates wrapping DAM as a tool).
        mask_bbox: ``(x1, y1, x2, y2)`` pixel bounding box of the captioned
            region, half-open (matches :class:`florence.Box`). Surfaced so
            downstream tools can compose — e.g. crop-around-region for a
            follow-up depth or orientation call. For a ``mask`` input this is
            the bbox of the mask's non-zero pixels; for a ``bbox`` input it
            round-trips to the input box.
        backend: ``"dam_3b_self_contained"`` — leaves room for a future DAM
            variant without changing the tool surface.
    """
    caption: str
    mask_bbox: tuple[int, int, int, int]
    backend: str = "dam_3b_self_contained"

    def __repr__(self) -> str:
        # Compact one-liner. A NOOA trace event fires per tool call; DAM
        # captions run hundreds of chars and a verbose repr would scale trace
        # size with the number of describe_region calls in a session. Truncate
        # the caption (the full text remains on .caption) and keep the bbox as
        # a tight 4-tuple — mirrors DepthResult.__repr__'s "don't dump the
        # array" and OrientationResult.__repr__'s compactness guard.
        caption = self.caption
        if len(caption) > _REPR_CAPTION_LIMIT:
            caption = caption[:_REPR_CAPTION_LIMIT].rstrip() + "…"
        return (
            f"RegionCaption(backend={self.backend!r}, "
            f"mask_bbox={self.mask_bbox}, caption={caption!r})"
        )


class DAMEstimator:
    """Backend holding the NVIDIA DAM regional-captioning model.

    Composes :class:`vqasynth.describe_anything.DescribeAnything` internally —
    it does NOT reimplement DAM weight loading, mask normalization, the
    ``get_description`` invocation, or the ``datasets.map`` batch column-append
    path (those live in :mod:`vqasynth.describe_anything`; this wrapper is a
    per-call adapter). This class adds the :class:`RegionCaption` shape on top
    of the underlying stage's plain-string caption and provides the
    module-level singleton used by :func:`describe_region`.

    ``device`` + ``dtype`` mirror the other tool backends so multi-GPU nodes
    can pin DAM separately. ``dtype`` accepts a torch dtype or an
    ``fp32``/``fp16``/``bf16`` string alias (resolved via
    :func:`_resolve_torch_dtype`); ``None`` lets the underlying stage pick its
    default (fp16, matching how DAM ships/serves).

    Args:
        model: optional pre-built DAM inference object (the thing DAM's
            ``init_dam`` returns). When supplied it short-circuits the lazy
            load — forwarded as the ``dam=`` seam on
            :class:`DescribeAnything`, the same injection hook
            ``tests/test_describe_anything.py`` exercises. This is the
            dependency-light path the tests stub.
        processor: reserved injection point. The self-contained
            ``nvidia/DAM-3B-Self-Contained`` variant bundles its own processor,
            so :class:`DescribeAnything` exposes no separate processor seam
            today; stashed for API symmetry with the other tool backends.
        device: torch device string (e.g. ``"cuda:1"``) or ``None`` for the
            stage default.
        dtype: torch dtype or ``fp32``/``fp16``/``bf16`` string alias.
        model_id: HuggingFace id of the self-contained DAM variant; defaults to
            ``"nvidia/DAM-3B-Self-Contained"`` (matches the underlying stage).
    """

    BACKEND = "dam_3b_self_contained"
    DEFAULT_MODEL_ID = "nvidia/DAM-3B-Self-Contained"

    def __init__(
        self,
        model=None,
        processor=None,
        device=None,
        dtype: Any = None,
        model_id: str | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        # Stash the injection points; the underlying DescribeAnything stage is
        # constructed lazily so importing this module never triggers a weight
        # download or a transformers/DAM import.
        self._model = model
        self._processor = processor
        self._stage = None

    def _ensure_loaded(self):
        if self._stage is not None:
            return
        from vqasynth.describe_anything import DescribeAnything

        # Resolve the dtype alias once — validates fp32/fp16/bf16 strings the
        # same way the other tools do. None → DescribeAnything picks fp16.
        resolved_dtype = _resolve_torch_dtype(self.dtype)

        self._stage = DescribeAnything(
            model_id=self.model_id,
            device=self.device,
            dtype=resolved_dtype,
            # A pre-built DAM short-circuits the stage's load(); None means
            # load the real DAM on first describe().
            dam=self._model,
        )

    def describe(self, image, mask) -> RegionCaption:
        """Caption a single region mask with DAM.

        Args:
            image: RGB ``PIL.Image`` containing the object.
            mask: ``HxW`` ``uint8``/``bool`` array (or any convention
                :meth:`DescribeAnything._normalize_mask` accepts).

        Returns:
            :class:`RegionCaption` with DAM's detailed description and the
            bbox of the captioned region.
        """
        self._ensure_loaded()
        caption = self._stage.describe(image, mask)
        return RegionCaption(
            caption=caption,
            mask_bbox=_bbox_of_mask(mask),
            backend=self.BACKEND,
        )


# Module-level singleton so N tool calls in one agent session share the same
# loaded DAM weights — loading DAM-3B is the expensive part and must not repeat
# per call. Mirrors the singleton pattern the other tool backends use.
_DEFAULT_ESTIMATOR: DAMEstimator | None = None


def _get_default_estimator() -> DAMEstimator:
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None:
        _DEFAULT_ESTIMATOR = DAMEstimator()
    return _DEFAULT_ESTIMATOR


def describe_region(image, *, bbox=None, mask=None) -> RegionCaption:
    """Detailed regional caption for one localized object via NVIDIA DAM.

    Wraps the batch-stage DAM captioner (:mod:`vqasynth.describe_anything`) as a
    discrete NOOA tool step: the SpatialAnnotator can ask "what exactly is in
    this region?" inline after
    :class:`experiments.nooa_agent.tools.florence.FlorenceSegmenter` produces a
    mask, rather than dropping into ``docker/describe_anything_stage/``.

    DAM captions MASKS, not boxes — its README is explicit that regional
    captioning prefers a mask prompt. Exactly ONE localization prompt is
    therefore required. When ``bbox`` is given we synthesize a filled-rectangle
    mask for DAM and report that same box back on the result.

    Args:
        image: RGB ``PIL.Image`` containing the object.
        bbox: exactly one of ``bbox`` / ``mask``. ``bbox=(x1, y1, x2, y2)`` in
            pixel coordinates (matches :meth:`florence.Box.to_list` ordering —
            NOT ``(x, y, w, h)``) → a filled-rectangle mask is synthesized for
            DAM.
        mask: alternatively, an ``HxW`` ``uint8``/``bool`` array (matches
            SAM/SAM2 mask output shape) → passed straight to DAM.

    Returns:
        :class:`RegionCaption`.

    Raises:
        ValueError: if neither or both of ``bbox`` / ``mask`` are given, or the
            image isn't a ``PIL.Image``.
    """
    from PIL import Image

    if (bbox is None) == (mask is None):
        raise ValueError(
            "describe_region requires exactly one of bbox= or mask= (got "
            + ("both" if bbox is not None else "neither")
            + ")"
        )
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected a PIL image but got {type(image)}")

    if bbox is not None:
        # DAM wants a mask, not a box; synthesize the filled rectangle the box
        # describes so DAM gets the mask prompt it captioning path expects.
        mask = _mask_from_bbox(bbox, image.width, image.height)

    return _get_default_estimator().describe(image, mask)


__all__ = [
    "RegionCaption",
    "DAMEstimator",
    "describe_region",
]
