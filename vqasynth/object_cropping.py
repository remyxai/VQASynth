"""Per-object crop construction for object-level grounded training rows.

VQASynth's prompt stage emits one global image per conversation and refers to
objects only by their caption text ("the chair to the left of the table").
Every referent is therefore ambiguous: the reader must re-derive which pixels
each caption names from the whole-scene representation.

This module adds the *object-level alignment* half of that row. From the
per-object masks the localization stage already produces it derives, per
object, a square crop (as an index into the row's image list, plus the pixel
box) and a caption-to-image-index map that downstream consumers use to
interleave the crops into the text in place of the textual entity mentions.

Adapted from *MultiModal Code-Switching: Interleaving Visual Objects into
Language for Explicit Object-Level Alignment* (MMCS, arXiv:2608.11167), which
replaces textual entities with their visual object crops to enforce local
vision-language grounding. Only the data-construction mechanism is ported;
the paper's pretraining run, model and scale study are out of scope here.
"""
from __future__ import annotations

import numpy as np

# Masks/crops are optional inputs throughout — the whole module degrades to an
# empty result when they are absent, so callers can add the interleaved variant
# unconditionally without gating on upstream columns existing.
MIN_CROP_SIDE = 8


def _as_2d_mask(mask) -> np.ndarray | None:
    """Coerce a mask to a 2D bool array, or None if it isn't one."""
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        return None
    return arr > 0


def mask_to_box(mask, padding: float = 0.1) -> tuple[int, int, int, int] | None:
    """Axis-aligned bounding box of a mask as a padded square ``(l, t, r, b)``.

    Same square-with-padding convention as
    :func:`vqasynth.orientation.crop_to_object` (the orientation stage crops
    each object the same way before its head sees it), so every consumer of a
    row's crops sees the same framing. Returns ``None`` for an empty or
    non-2D mask.

    Args:
        mask: 2D array; non-zero pixels mark the object.
        padding: fraction of the longer side to pad around the object.

    Returns:
        ``(left, top, right, bottom)`` in pixel coordinates, or ``None``.
    """
    bools = _as_2d_mask(mask)
    if bools is None or not bools.any():
        return None

    ys, xs = np.nonzero(bools)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    side = int(max(y_max - y_min, x_max - x_min) * (1 + padding))
    side = max(side, MIN_CROP_SIDE)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    return (left, top, left + side, top + side)


def clamp_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Clamp a box to the image and drop it if nothing remains.

    Cropping is done by the dataset consumer, not here, so a box that runs off
    the frame is clipped rather than canvas-grown — the caller decides how to
    pad. Boxes that collapse to less than ``MIN_CROP_SIDE`` on either axis are
    dropped (``None``) so a degenerate mask never yields a 1-pixel "object".
    """
    left = max(0, min(box[0], width))
    top = max(0, min(box[1], height))
    right = max(0, min(box[2], width))
    bottom = max(0, min(box[3], height))
    if right - left < MIN_CROP_SIDE or bottom - top < MIN_CROP_SIDE:
        return None
    return (left, top, right, bottom)


def crop_object_records(
    masks,
    captions=None,
    image_size: tuple[int, int] | None = None,
    padding: float = 0.1,
) -> list[dict]:
    """Build one crop record per localized object.

    Args:
        masks: per-object mask arrays (HxW, any numeric/bool dtype) — the
            ``masks`` column written by the localization stage.
        captions: per-object caption strings — the ``captions`` column. When
            given it must be the same length as ``masks``; extra captions are
            ignored, and captions without a usable mask get no record.
        image_size: ``(width, height)`` of the source image, used to clamp
            boxes into the frame. When omitted boxes are left unclamped.
        padding: passed through to :func:`mask_to_box`.

    Returns:
        A list of ``{"index", "caption", "bbox"}`` records. ``index`` is the
        object's slot (starting at 1, so 0 stays reserved for the full scene
        image — the convention ``create_messages_from_prompts`` already uses
        for its per-content image indices), ``caption`` is ``None`` when no
        captions were supplied.
    """
    if masks is None:
        return []

    usable_captions = list(captions) if captions is not None else None

    records: list[dict] = []
    for slot, mask in enumerate(masks, start=1):
        box = mask_to_box(mask, padding=padding)
        if box is None:
            continue
        if image_size is not None and len(image_size) == 2:
            box = clamp_box(box, int(image_size[0]), int(image_size[1]))
            if box is None:
                continue

        caption = None
        if usable_captions is not None and slot - 1 < len(usable_captions):
            caption = usable_captions[slot - 1]

        records.append(
            {
                "index": slot,
                "caption": caption,
                "bbox": box,
            }
        )
    return records


def build_crop_features(records: list[dict]) -> dict[str, list]:
    """Split crop records into the two columns ``apply_transform`` emits.

    Returns a dict with ``object_crops`` (one ``{"index", "text", "type"}``
    image content entry per record, matching the message content schema) and
    ``object_crop_boxes`` (the parallel pixel boxes, for consumers that crop
    lazily instead of storing expanded images).
    """
    entries: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    for record in records:
        entries.append(
            {"index": record["index"], "text": None, "type": "image"}
        )
        boxes.append(record["bbox"])
    return {"object_crops": entries, "object_crop_boxes": boxes}


def caption_to_crop_index(records: list[dict]) -> dict[str, int]:
    """Map each caption string to the image index of its crop.

    The join key for interleaving: a prompt's ``[A]``/``[B]`` slot resolves to
    a caption, and the caption resolves here to the crop's image index.
    Captions appearing more than once (Florence grounds the same phrase to
    several boxes) collapse to the first crop — the paper's construction
    assumes one crop per entity mention.
    """
    mapping: dict[str, int] = {}
    for record in records:
        caption = record.get("caption")
        if not caption:
            continue
        key = caption.strip().lower()
        if key not in mapping:
            mapping[key] = record["index"]
    return mapping


def interleave_object_references(
    text: str, crop_index: dict[str, int], placeholder: str = "this object"
) -> dict:
    """Replace textual object mentions in ``text`` with their crop indices.

    Captions are matched longest-first, so a caption that contains a shorter
    caption ("the red mug on the table" vs "the red mug") resolves to its own
    crop rather than being partially eaten by the shorter one. Each match is
    swapped for a positionally-numbered stand-in, which is what makes mention
    order recoverable: the returned crop indices follow the order a reader
    meets the referents, not the order the captions happened to be stored in.

    Args:
        text: a prompt question or answer, with object captions embedded.
        crop_index: caption (case-insensitive) -> crop image index.
        placeholder: what to leave in the text where a caption was. A
            deictic phrase rather than the empty string, so the sentence
            still reads grammatically once the visual token is attached.

    Returns:
        ``{"text": the rewritten text, "crop_indices": indices of the crops
        the text now refers to, in first-mention order}``.
    """
    spans: list[tuple[int, int, int]] = []  # (start, end, crop index)

    # ``caption_to_crop_index`` lowercases its keys and the prompt templates
    # lowercase their captions before substitution, so matching in lowercase
    # keeps both sides aligned even when a caption arrives cased.
    haystack = text.lower()
    for caption, index in sorted(
        crop_index.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if not caption:
            continue
        needle = caption.lower()
        start = haystack.find(needle)
        while start != -1:
            spans.append((start, start + len(needle), index))
            start = haystack.find(needle, start + len(needle))

    # Resolve overlaps in reading order: walk the spans by start position and
    # drop any that begins before the previous kept span ends. A tie at the
    # same start is broken by length (longest first) via the pre-sort below, so
    # a caption that contains a shorter caption wins over it.
    ordered = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    kept: list[tuple[int, int, int]] = []
    for start, end, index in ordered:
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, index))

    crop_indices: list[int] = []
    pieces: list[str] = []
    cursor = 0
    for start, end, index in kept:
        pieces.append(text[cursor:start])
        pieces.append(f"{placeholder} ({len(crop_indices) + 1})")
        if index not in crop_indices:
            crop_indices.append(index)
        cursor = end

    pieces.append(text[cursor:])
    return {"text": "".join(pieces), "crop_indices": crop_indices}


def build_grounded_messages(prompts, crop_index: dict[str, int]) -> list[dict]:
    """Rebuild a prompt-stage conversation with object mentions grounded.

    Mirrors :meth:`vqasynth.prompts.PromptGenerator.create_messages_from_prompts`
    turn for turn — one user and one assistant message per prompt, the scene
    image attached only to the first user turn — except that each object
    mention is replaced by the image index of that object's crop, and each
    crop is attached exactly once at its own index (0 stays reserved for the
    full scene).

    Args:
        prompts: prompt strings in the pipeline's ``question + " Answer: " +
            answer`` form.
        crop_index: caption -> crop image index, from
            :func:`caption_to_crop_index`.

    Returns:
        Message dictionaries in the dataset's ``{"role", "content"}`` schema,
        with content items of ``{"index", "text", "type"}``.
    """
    messages: list[dict] = []
    first_prompt = True
    attached: set[int] = set()

    for prompt in prompts:
        if " Answer: " not in prompt:
            continue

        question, answer = prompt.split(" Answer: ", 1)

        content: list[dict] = []
        if first_prompt:
            content.append({"index": 0, "text": None, "type": "image"})

        # Question first, then answer, so the crops attach in the order a
        # reader meets the referents across both halves of the turn.
        for text in (question, answer):
            grounded = interleave_object_references(text, crop_index)
            for index in grounded["crop_indices"]:
                if index not in attached:
                    attached.add(index)
                    content.append({"index": index, "text": None, "type": "image"})
            content.append(
                {"index": None, "text": grounded["text"].strip(), "type": "text"}
            )

        messages.append({"content": content, "role": "user"})
        messages.append(
            {
                "content": [{"index": None, "text": answer.strip(), "type": "text"}],
                "role": "assistant",
            }
        )

        first_prompt = False

    return messages
