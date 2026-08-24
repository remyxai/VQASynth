"""Structural tests for vqasynth.object_cropping and its prompts.py wiring.

Mirrors the philosophy of ``tests/test_correspondence.py``: verify the crop /
interleave mechanics and the message-schema contract without a GPU, model
download, or open3d. ``PromptGenerator.apply_transform`` needs real point
clouds, so the wiring is exercised by stubbing ``run`` (the point it would
call into) and driving the real ``apply_transform`` -> ``create_messages_
from_prompts_interleaved`` path over mask arrays.
"""
from __future__ import annotations

import numpy as np
import pytest

from vqasynth.object_cropping import (
    build_crop_features,
    build_grounded_messages,
    caption_to_crop_index,
    clamp_box,
    crop_object_records,
    interleave_object_references,
    mask_to_box,
)

# The call site under test. ``vqasynth.prompts`` pulls in
# ``vqasynth.scene_fusion``, which imports vggt at module scope — a Docker-only
# dep. Same guard as ``tests/test_describe_anything.py``: the wiring tests run
# in the full env and skip here; the crop/interleave tests above always run.
try:
    from vqasynth.prompts import PromptGenerator
except Exception:  # vggt / open3d / sam2 not installed in this env
    PromptGenerator = None  # type: ignore[assignment]

requires_prompt_generator = pytest.mark.skipif(
    PromptGenerator is None,
    reason="vqasynth.prompts unavailable (Docker-only deps missing)",
)


H, W = 64, 80


def _mask_in(x0, y0, x1, y1):
    """A rectangular object mask occupying the given pixel box."""
    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


# ---------------------------------------------------------------------------
# crop geometry
# ---------------------------------------------------------------------------
def test_mask_to_box_is_square_and_padded():
    box = mask_to_box(_mask_in(10, 10, 30, 20), padding=0.0)
    left, top, right, bottom = box
    # Square framing, centred on the blob. Side arithmetic matches
    # vqasynth.orientation.crop_to_object (side = max span, no +1), so crops
    # here frame objects exactly the way the orientation stage does.
    assert right - left == bottom - top == 19
    assert (left, top) == (10, 5)


def test_mask_to_box_rejects_empty_and_non_2d():
    assert mask_to_box(np.zeros((H, W), dtype=np.uint8)) is None
    assert mask_to_box(np.zeros((2, H, W), dtype=np.uint8)) is None
    assert mask_to_box(None) is None


def test_clamp_box_drops_offscreen_and_degenerate():
    assert clamp_box((-10, -10, 40, 40), W, H) == (0, 0, 40, 40)
    assert clamp_box((0, 0, 4, 4), W, H) is None      # collapses below minimum
    assert clamp_box((W - 2, H - 2, W + 9, H + 9), W, H) is None


def test_crop_object_records_skips_unusable_masks_and_keeps_slots():
    masks = [np.zeros((H, W), dtype=np.uint8), _mask_in(5, 5, 25, 25)]
    records = crop_object_records(masks, ["empty object", "a red chair"])
    assert [r["index"] for r in records] == [2]      # slot, not list position
    assert records[0]["caption"] == "a red chair"
    assert len(records[0]["bbox"]) == 4


def test_crop_object_records_degrades_to_empty():
    assert crop_object_records(None) == []
    assert crop_object_records([None, None]) == []
    assert crop_object_records([], ["caption with no mask"]) == []


def test_crop_records_clamp_to_image_when_size_given():
    records = crop_object_records(
        [_mask_in(60, 40, 79, 63)], image_size=(W, H)
    )
    left, top, right, bottom = records[0]["bbox"]
    assert 0 <= left < right <= W and 0 <= top < bottom <= H


# ---------------------------------------------------------------------------
# the interleave join
# ---------------------------------------------------------------------------
def test_longest_caption_wins_over_containing_caption():
    records = [
        {"index": 1, "caption": "a red mug", "bbox": (0, 0, 10, 10)},
        {"index": 2, "caption": "a red mug on the table", "bbox": (0, 0, 11, 11)},
    ]
    grounded = interleave_object_references(
        "Is a red mug on the table to the left of a red mug?",
        caption_to_crop_index(records),
    )
    assert grounded["crop_indices"] == [2, 1]
    assert "a red mug on the table" not in grounded["text"]
    # Each mention keeps a numbered deictic stand-in, so the text still parses
    # and the crops attach in reading order.
    assert grounded["text"] == (
        "Is this object (1) to the left of this object (2)?"
    )


def test_unmatched_text_is_left_untouched():
    grounded = interleave_object_references(
        "How far apart are they?", {"a red mug": 3}
    )
    assert grounded["text"] == "How far apart are they?"
    assert grounded["crop_indices"] == []


def test_cased_mention_still_matches_lowercased_caption_key():
    # caption_to_crop_index lowercases its keys; a cased mention in the text
    # must still resolve and keep its original casing in the surrounding text.
    grounded = interleave_object_references(
        "Is A Red Mug to the left of it?", {"a red mug": 3}
    )
    assert grounded == {"text": "Is this object (1) to the left of it?",
                        "crop_indices": [3]}


# ---------------------------------------------------------------------------
# the prompts.py call site
# ---------------------------------------------------------------------------
def _example():
    return {
        "captions": ["a red chair", "a wooden table"],
        "masks": [_mask_in(4, 4, 24, 24), _mask_in(40, 30, 70, 55)],
        "pointclouds": [None, None],
        "is_canonicalized": [False, False],
    }


def _generator_with_stubbed_run(prompts):
    gen = PromptGenerator()
    gen.run = lambda *args, **kwargs: list(prompts)
    return gen


@requires_prompt_generator
def test_apply_transform_emits_both_message_variants():
    gen = _generator_with_stubbed_run(
        ["Is a red chair to the left of a wooden table? Answer: yes"]
    )
    out = gen.apply_transform(_example())

    # Plain variant is unchanged: one global image, textual object names.
    assert out["messages"][0]["content"][0] == {
        "index": 0, "text": None, "type": "image"
    }
    assert "a red chair" in out["messages"][0]["content"][1]["text"]

    # Grounded variant: scene at 0, each object's crop at its own index, and
    # neither caption surviving in the text.
    user = out["grounded_messages"][0]
    images = [c for c in user["content"] if c["type"] == "image"]
    assert [c["index"] for c in images] == [0, 1, 2]
    texts = [c["text"] for c in user["content"] if c["type"] == "text"]
    assert "a red chair" not in " ".join(texts)
    assert "a wooden table" not in " ".join(texts)

    # Schema shape matches docker/prompt_stage/process_prompts.py: every
    # content item carries exactly role/index/text/type keys.
    for message in out["grounded_messages"]:
        assert message["role"] in ("user", "assistant")
        for item in message["content"]:
            assert set(item) == {"index", "text", "type"}


@requires_prompt_generator
def test_apply_transform_answers_keep_grounding_too():
    gen = _generator_with_stubbed_run(
        ["Which is wider? Answer: a red chair is wider"]
    )
    out = gen.apply_transform(_example())
    user_content = out["grounded_messages"][0]["content"]
    # Scene image + the crop the answer mentions, both attached in the user
    # turn as input context, then the question text. The answer itself moves to
    # the assistant turn (grounded), it does not stay in the user turn.
    assert [c["type"] for c in user_content] == ["image", "image", "text"]
    assert [c["index"] for c in user_content if c["type"] == "image"] == [0, 1]
    assert user_content[-1]["text"] == "Which is wider?"
    # The assistant turn is the grounded answer, not the verbatim one.
    assert (
        out["grounded_messages"][1]["content"][0]["text"]
        == "this object (1) is wider"
    )


@requires_prompt_generator
def test_apply_transform_without_masks_yields_empty_not_failure():
    example = _example()
    example["masks"] = None
    gen = _generator_with_stubbed_run(
        ["Is a red chair to the left of a wooden table? Answer: yes"]
    )
    out = gen.apply_transform(example)

    assert out is not None
    assert out["grounded_messages"] == []
    assert out["object_crops"] == []
    assert out["object_crop_boxes"] == []
    assert out["messages"]  # the plain variant still gets produced


def test_build_crop_features_matches_records():
    records = crop_object_records(
        [_mask_in(4, 4, 24, 24), _mask_in(40, 30, 70, 55)],
        ["a red chair", "a wooden table"],
    )
    features = build_crop_features(records)
    assert features["object_crops"] == [
        {"index": 1, "text": None, "type": "image"},
        {"index": 2, "text": None, "type": "image"},
    ]
    assert len(features["object_crop_boxes"]) == 2


def test_build_grounded_messages_attaches_each_crop_once_across_turns():
    prompts = [
        "Is a red chair to the left of a wooden table? Answer: yes",
        "Which is wider? Answer: a red chair is wider",
    ]
    messages = build_grounded_messages(
        prompts, {"a red chair": 1, "a wooden table": 2}
    )

    # Two prompts -> two user/assistant turn pairs.
    assert [m["role"] for m in messages] == ["user", "assistant"] * 2

    first_user = messages[0]["content"]
    assert [c["index"] for c in first_user if c["type"] == "image"] == [0, 1, 2]

    # The scene image and both crops appear only in the first user turn; the
    # second turn's question mentions nothing new and its answer's crop is
    # already attached, so the user turn is just the question text.
    second_user = messages[2]["content"]
    assert [c["type"] for c in second_user] == ["text"]
    assert second_user[0]["text"] == "Which is wider?"
    # The grounded answer lives in the assistant turn now, not the user turn.
    assert messages[3]["content"][0]["text"] == "this object (1) is wider"
