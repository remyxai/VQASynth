"""Smoke tests for vqasynth.describe_anything.

Verifies the DAM captioning + QA-emitter mechanics against synthetic masks and
a stub DAM callable — no CUDA, no transformers DAM download, no SAM. The real
end-to-end captioning path runs in the Docker stage on a GPU host.

Dependency note: this module deliberately stays clear of the
``vggt``/``open3d``/``sam2`` chain (Docker-only deps). The QA emitter's message
schema is therefore exercised two ways:

  * directly, against :meth:`DescribeAnything._messages_from_prompts`, and
  * against the *pre-existing* :mod:`vqasynth.prompt_templates` data module to
    prove the emitter interoperates with the prompt strings the rest of the
    pipeline already produces.

When ``vggt`` happens to be importable (full Docker env), an extra test
cross-checks byte-for-byte parity with
``vqasynth.prompts.PromptGenerator.create_messages_from_prompts``.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vqasynth.describe_anything import DescribeAnything
# Pre-existing package module (pure data, no heavy deps) — imported to exercise
# the prompt convention the rest of the pipeline relies on.
from vqasynth.prompt_templates import (
    distance_template_questions,
    distance_template_answers,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _StubDAM:
    """Callable stub: dam(image, mask_pil) -> str. Records what it was called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, image, mask_pil):
        arr = np.asarray(mask_pil)
        foreground_px = int((arr > 0).sum())
        self.calls.append(
            {"image_size": image.size, "mask_shape": arr.shape, "foreground_px": foreground_px}
        )
        return f"caption #{len(self.calls)} ({foreground_px} px)"


class _StubDAMWithGetDescription:
    """Mimics the real DAM surface: exposes get_description, non-streaming."""

    def __init__(self):
        self.calls = 0

    def get_description(self, image, mask_pil, query, streaming=False, **kwargs):
        assert streaming is False, "non-streaming must be requested"
        self.calls += 1
        return f"gd caption #{self.calls}"


def _make_image(size=(16, 12)):
    return Image.new("RGB", size, (123, 200, 50))


def _make_masks():
    """Three masks in different conventions (uint8 0/255, bool, float)."""
    u8 = np.zeros((12, 16), dtype=np.uint8)
    u8[0:4, 0:4] = 255

    boolean = np.zeros((12, 16), dtype=bool)
    boolean[4:8, 4:8] = True

    fl = np.zeros((12, 16), dtype=np.float32)
    fl[8:12, 8:12] = 1.0
    return [u8, boolean, fl]


# ---------------------------------------------------------------------------
# Lifecycle / import seam
# ---------------------------------------------------------------------------
def test_module_imports_and_does_not_load_without_dam():
    """Constructing without a stub must not trigger any model load/download."""
    dam = DescribeAnything()  # no dam= -> would load on first describe(), not now
    assert dam._dam is None
    assert dam.model_id == "nvidia/DAM-3B-Self-Contained"


def test_injected_stub_short_circuits_load():
    stub = _StubDAM()
    dam = DescribeAnything(dam=stub)
    assert dam.load() is stub  # load() returns the injected object unchanged


# ---------------------------------------------------------------------------
# Mask normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("index", range(3))
def test_normalize_mask_canonicalizes_each_convention(index):
    masks = _make_masks()
    arr = np.asarray(DescribeAnything._normalize_mask(masks[index]))
    assert arr.dtype == np.uint8
    assert set(np.unique(arr)).issubset({0, 255})
    assert int((arr > 0).sum()) == 16  # each mask covers a 4x4 foreground block


def test_normalize_mask_accepts_pil_image():
    pil_mask = Image.new("L", (8, 8), 0)
    out = DescribeAnything._normalize_mask(pil_mask)
    assert np.asarray(out).shape == (8, 8)


def test_normalize_mask_foreground_from_float_threshold():
    fl = np.full((6, 6), 0.4, dtype=np.float32)  # below 0.5 -> background
    fl[0, 0] = 0.9
    arr = np.asarray(DescribeAnything._normalize_mask(fl))
    assert arr[0, 0] == 255
    assert arr[1, 1] == 0


# ---------------------------------------------------------------------------
# Per-region captioning
# ---------------------------------------------------------------------------
def test_describe_regions_aligns_and_calls_once_per_mask():
    stub = _StubDAM()
    dam = DescribeAnything(dam=stub)
    image = _make_image()
    masks = _make_masks()

    captions = dam.describe_regions(image, masks)

    assert len(captions) == len(masks)
    assert len(stub.calls) == len(masks)
    # Every mask was normalized to a 0/255 PIL image matching the image's HxW.
    for call in stub.calls:
        assert call["mask_shape"] == (12, 16)
        assert call["foreground_px"] == 16  # each mask has a 4x4 block


def test_describe_uses_get_description_when_present():
    stub = _StubDAMWithGetDescription()
    dam = DescribeAnything(dam=stub)
    caption = dam.describe(_make_image(), _make_masks()[0])
    assert caption == "gd caption #1"
    assert stub.calls == 1


def test_describe_strips_and_joins_generator_return():
    class GenStub:
        def __call__(self, image, mask_pil):
            return iter(["  hello ", "world"])

    dam = DescribeAnything(dam=GenStub())
    assert dam.describe(_make_image(), _make_masks()[0]) == "hello world"


def test_describe_regions_empty_masks():
    dam = DescribeAnything(dam=_StubDAM())
    assert dam.describe_regions(_make_image(), []) == []


# ---------------------------------------------------------------------------
# QA-pair emitter
# ---------------------------------------------------------------------------
def test_generate_qa_pairs_count_scales_with_objects():
    dam = DescribeAnything(dam=_StubDAM(), n_questions_per_object=3)
    masks = _make_masks()
    captions = ["a red mug", "a wooden chair", "a black cat"]
    prompts, messages = dam.generate_qa_pairs(masks, captions)

    assert len(prompts) == 3 * 3
    # Every prompt is the "question Answer: answer" shape the pipeline uses.
    assert all(" Answer: " in p for p in prompts)


def test_generate_qa_pairs_messages_schema():
    dam = DescribeAnything(dam=_StubDAM(), n_questions_per_object=2)
    captions = ["a red mug with a chipped handle", "a wooden chair"]
    prompts, messages = dam.generate_qa_pairs(_make_masks()[:2], captions)

    # Roles alternate user/assistant, beginning with user.
    roles = [m["role"] for m in messages]
    assert roles[0] == "user"
    assert roles == ["user", "assistant"] * (len(roles) // 2)

    # Exactly one image token, on the first user message, at index 0.
    image_parts = [
        part for m in messages for part in m["content"] if part.get("type") == "image"
    ]
    assert len(image_parts) == 1
    first = messages[0]["content"][0]
    assert first["type"] == "image" and first["index"] == 0 and first["text"] is None

    # Subsequent turns carry no image token, only text.
    for m in messages[2:]:
        for part in m["content"]:
            assert part["type"] == "text" and isinstance(part["text"], str)


def test_messages_from_prompts_parses_existing_pipeline_prompt_format():
    """The message builder must accept the exact 'q Answer: a' strings the
    existing pipeline (prompt_templates + PromptGenerator) already produces."""
    q = (
        distance_template_questions[0]
        .replace("[A]", "mug")
        .replace("[B]", "chair")
    )
    a = (
        distance_template_answers[0]
        .replace("[A]", "mug")
        .replace("[B]", "chair")
        .replace("[X]", "30 cm")
    )
    prompt = f"{q} Answer: {a}"

    messages = DescribeAnything._messages_from_prompts([prompt])

    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "mug" in messages[0]["content"][-1]["text"]
    assert "30 cm" in messages[1]["content"][0]["text"]


def test_messages_match_real_prompt_generator_when_importable():
    """When vggt/open3d are installed (full Docker env), the inlined message
    builder must match vqasynth.prompts.PromptGenerator byte-for-byte."""
    try:
        from vqasynth.prompts import PromptGenerator
    except Exception:  # vggt / open3d / sam2 not installed in this env
        pytest.skip("vqasynth.prompts unavailable (Docker-only deps missing)")

    prompts = ["What is this? Answer: a mug.", "Is it red? Answer: yes."]
    assert DescribeAnything._messages_from_prompts(prompts) == (
        PromptGenerator().create_messages_from_prompts(prompts)
    )


def test_generate_qa_pairs_answers_contain_caption():
    dam = DescribeAnything(dam=_StubDAM(), n_questions_per_object=1)
    caption = "a person wearing a blue jacket and round glasses"
    prompts, _ = dam.generate_qa_pairs([np.zeros((4, 4), np.uint8)], [caption])

    assert len(prompts) == 1
    _, answer = prompts[0].split(" Answer: ", 1)
    assert caption in answer


def test_generate_qa_pairs_rejects_misaligned_lengths():
    dam = DescribeAnything(dam=_StubDAM())
    with pytest.raises(ValueError, match="align"):
        dam.generate_qa_pairs([np.zeros((4, 4), np.uint8)] * 2, ["only one"])


def test_generate_qa_pairs_skips_empty_captions():
    dam = DescribeAnything(dam=_StubDAM(), n_questions_per_object=2)
    masks = [np.zeros((4, 4), np.uint8)] * 2
    prompts, messages = dam.generate_qa_pairs(masks, ["a red mug", ""])
    assert len(prompts) == 2
    assert messages  # still produced for the non-empty caption


# ---------------------------------------------------------------------------
# apply_transform — mirrors Localizer.apply_transform batched/single handling
# ---------------------------------------------------------------------------
def test_apply_transform_single_example():
    stub = _StubDAM()
    dam = DescribeAnything(dam=stub)
    image = _make_image()
    example = {"image": image, "masks": _make_masks()}

    out = dam.apply_transform(example, images="image")

    assert len(out["dam_captions"]) == 3
    assert len(stub.calls) == 3
    assert isinstance(out["dam_messages"], list)
    assert out["dam_messages"]  # non-empty QA pairs


def test_apply_transform_batched():
    stub = _StubDAM()
    dam = DescribeAnything(dam=stub)
    img_a, img_b = _make_image(), _make_image((20, 10))
    example = {
        "image": [img_a, img_b],
        "masks": [_make_masks(), _make_masks()[:2]],
    }

    out = dam.apply_transform(example, images="image")

    assert len(out["dam_captions"]) == 2          # one list per image
    assert len(out["dam_captions"][0]) == 3
    assert len(out["dam_captions"][1]) == 2
    assert len(stub.calls) == 5                       # 3 + 2 masks captioned
    assert len(out["dam_messages"]) == 2


def test_apply_transform_converts_non_rgb_image():
    dam = DescribeAnything(dam=_StubDAM())
    rgba = Image.new("RGBA", (16, 12), (1, 2, 3, 255))
    example = {"image": rgba, "masks": _make_masks()[:1]}
    out = dam.apply_transform(example, images="image")
    assert len(out["dam_captions"]) == 1
