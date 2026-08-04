"""Region-level captioning with NVIDIA's Describe Anything Model (DAM).

This stage sits downstream of :mod:`vqasynth.localize`. ``Localizer.run`` already
produces one SAM mask per detected object (uint8, ``0``/``255``); DAM turns each
of those masks into a detailed region caption, which is much richer than the
class-label-only captions the localizer emits. Those detailed captions then feed
a QA-pair emitter so each ``(image, mask, caption)`` triple becomes
spatial-VQA training samples.

DAM is loaded lazily through ``transformers`` (the self-contained
``nvidia/DAM-3B-Self-Contained`` variant, matching the reference example in the
DAM repo: ``examples/dam_with_sam_self_contained.py``). Nothing here imports DAM
or downloads weights at module import time, and the captioning path accepts an
injected ``dam`` callable, so the module is fully testable without a GPU or DAM
weights installed — see ``tests/test_describe_anything.py``.

Refs:
  - Paper:  https://arxiv.org/abs/2504.16072 (Describe Anything, ICCV 2025)
  - Code:   https://github.com/NVlabs/describe-anything
  - Weights: https://huggingface.co/collections/nvidia/describe-anything-680825bb8f5e41ff0785834c
"""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

# NVIDIA-published self-contained DAM variant. Overridable via the
# ``model_id`` ctor arg (and the ``--model_id`` flag on the Docker stage).
DEFAULT_MODEL_ID = "nvidia/DAM-3B-Self-Contained"

# Default query the DAM repo uses for region captioning. The ``<image>`` token
# is required by DAM's conversation template.
DEFAULT_QUERY = "<image>\nDescribe the masked region in detail."

# DAM prompt modes exposed by the self-contained model. ``full+focal_crop``
# (aliased as ``focal_prompt`` upstream) gives the richest region detail, which
# is what we want for distinguishing subjects by small visual features.
DEFAULT_PROMPT_MODE = "full+focal_crop"
DEFAULT_CONV_MODE = "v1"


# ---------------------------------------------------------------------------
# QA-pair templates for the detailed-caption emitter.
#
# These intentionally target the fine-grained per-region detail DAM produces
# (the maintainer feedback on issue #51 calls out "distinguishing people in a
# scene based on small details"). They mirror the question/answer template
# style used in :mod:`vqasynth.prompt_templates`.
# ---------------------------------------------------------------------------
_DETAILED_QUESTION_TEMPLATES = [
    "Describe the highlighted object in detail.",
    "What does the highlighted object look like?",
    "Describe the masked region in the image in detail.",
    "What distinguishing visual features does the highlighted object have?",
    "How would you describe the appearance of the highlighted object?",
    "What fine-grained details can you see on the highlighted object?",
    "Describe the highlighted subject, focusing on the small visual details "
    "that set it apart from other objects in the scene.",
]

# Answer frames wrap the DAM caption. ``{caption}`` is the detailed region
# description; frames are chosen so they read naturally whether the caption
# begins capitalized or not.
_DETAILED_ANSWER_FRAMES = [
    "{caption}",
    "The highlighted object: {caption}",
    "Detailed description: {caption}",
    "Here is what I see in the highlighted region: {caption}",
]


class DescribeAnything:
    """Caption each SAM mask from :class:`vqasynth.localize.Localizer` with DAM.

    Parameters mirror the conventions of the other pipeline stages
    (:class:`Localizer`, :class:`DepthEstimator`, ...): a ``device`` default, a
    configurable ``model_id`` that points at a NVIDIA-published DAM variant, and
    an ``apply_transform`` method shaped for ``datasets.map``.

    Args:
        model_id: HuggingFace id of the self-contained DAM variant.
        device: Torch device; defaults to CUDA when available.
        dtype: Model dtype; DAM ships/serves in fp16.
        conv_mode / prompt_mode: Forwarded to ``model.init_dam(...)``.
        query: Text prompt handed to ``get_description`` (needs ``<image>``).
        temperature / top_p / max_new_tokens: Generation kwargs for DAM.
        dam: Optional pre-built DAM object or callable
            ``dam(image, mask_pil) -> str``. When supplied, :meth:`load` is a
            no-op — this is the dependency-light seam the tests exercise.
        n_questions_per_object: QA pairs emitted per ``(mask, caption)`` triple.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        conv_mode: str = DEFAULT_CONV_MODE,
        prompt_mode: str = DEFAULT_PROMPT_MODE,
        query: str = DEFAULT_QUERY,
        temperature: float = 0.2,
        top_p: float = 0.5,
        max_new_tokens: int = 512,
        dam=None,
        n_questions_per_object: int = 2,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else torch.float16
        self.conv_mode = conv_mode
        self.prompt_mode = prompt_mode
        self.query = query
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.n_questions_per_object = n_questions_per_object
        # When None, :meth:`load` builds the real DAM on first use.
        self._dam = dam

    # ------------------------------------------------------------------
    # DAM lifecycle
    # ------------------------------------------------------------------
    def load(self):
        """Lazily build the DAM model via ``transformers`` (self-contained path).

        Returns the injected ``dam`` unchanged when one was supplied at
        construction, so tests never touch transformers or download weights.
        """
        if self._dam is not None:
            return self._dam

        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        # init_dam returns the DAM inference object exposing get_description().
        self._dam = model.init_dam(
            conv_mode=self.conv_mode, prompt_mode=self.prompt_mode
        )
        return self._dam

    # ------------------------------------------------------------------
    # Mask handling
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_mask(mask) -> Image.Image:
        """Coerce any mask convention VQASynth/SAM produces into a DAM mask.

        DAM expects a PIL image whose foreground pixels are ``255``. Accepts:
          * ``PIL.Image`` (converted to single-channel "L"),
          * ``bool`` / ``uint8`` / ``int`` arrays (``Localizer`` emits uint8
            ``0``/``255``; SAM2 emits bool),
          * floating arrays in ``[0, 1]``.

        Returns a binary ``0``/``255`` PIL image sized to the mask.
        """
        if isinstance(mask, Image.Image):
            arr = np.asarray(mask.convert("L"))
        else:
            arr = np.asarray(mask)

        if arr.dtype == bool:
            foreground = arr
        elif np.issubdtype(arr.dtype, np.floating):
            foreground = arr > 0.5
        else:
            # uint8 0/255 (Localizer) and uint8 0/1 both foreground on nonzero.
            foreground = arr > 0

        mask_uint8 = (foreground.astype(np.uint8)) * 255
        return Image.fromarray(mask_uint8)

    def _call_dam(self, dam, image: Image.Image, mask_pil: Image.Image) -> str:
        """Run captioning through the real DAM or an injected callable stub.

        The real DAM object exposes ``get_description`` (non-streaming returns a
        ``str``). A test stub may instead be a plain callable
        ``dam(image, mask_pil) -> str``. Streaming generators are joined, so a
        stub that yields tokens also works.
        """
        if hasattr(dam, "get_description"):
            out = dam.get_description(
                image,
                mask_pil,
                self.query,
                streaming=False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
            )
        else:
            out = dam(image, mask_pil)

        if out is None:
            return ""
        if isinstance(out, str):
            return out.strip()
        # Streaming / generator fallback — join yielded tokens.
        return "".join(str(tok) for tok in out).strip()

    # ------------------------------------------------------------------
    # Per-region captioning
    # ------------------------------------------------------------------
    def describe(self, image: Image.Image, mask) -> str:
        """Return DAM's detailed caption for a single ``image`` + ``mask``."""
        dam = self.load()
        mask_pil = self._normalize_mask(mask)
        return self._call_dam(dam, image, mask_pil)

    def describe_regions(self, image: Image.Image, masks) -> list[str]:
        """Caption every mask; returns a list aligned 1:1 with ``masks``."""
        return [self.describe(image, mask) for mask in masks]

    # ------------------------------------------------------------------
    # QA-pair emitter
    # ------------------------------------------------------------------
    @staticmethod
    def _messages_from_prompts(prompts):
        """Build the pipeline's ``messages`` schema from QA prompt strings.

        This is a faithful, dependency-light replica of
        :meth:`vqasynth.prompts.PromptGenerator.create_messages_from_prompts` —
        byte-for-byte the same output (image token on the first user turn only,
        alternating user/assistant, ``role``/``content``/``index``/``text``/
        ``type`` keys). It is inlined here rather than imported because
        ``vqasynth.prompts`` pulls in ``vqasynth.scene_fusion`` (and therefore
        the Docker-only ``vggt``), which would make this module un-importable
        under ``requirements.txt`` alone. ``tests/test_describe_anything.py``
        cross-checks parity against the real generator when ``vggt`` is present.
        """
        messages = []
        first_prompt = True
        for prompt in prompts:
            if "Answer: " not in prompt:
                continue
            question, answer = prompt.split("Answer: ", 1)
            content = [{"index": None, "text": question.strip(), "type": "text"}]
            if first_prompt:
                content.insert(0, {"index": 0, "text": None, "type": "image"})
            messages.append({"content": content, "role": "user"})
            messages.append(
                {
                    "content": [{"index": None, "text": answer.strip(), "type": "text"}],
                    "role": "assistant",
                }
            )
            first_prompt = False
        return messages

    def generate_qa_pairs(self, masks, captions, n_questions_per_object=None):
        """Turn ``(mask, detailed_caption)`` triples into spatial-VQA samples.

        Builds ``"question Answer: answer"`` strings (the exact shape the rest
        of the pipeline emits — see :mod:`vqasynth.prompt_templates` and
        :mod:`vqasynth.prompts`) and structures them into the ``messages``
        schema via :meth:`_messages_from_prompts`.

        Args:
            masks: List of masks (only used for length alignment here).
            captions: Detailed DAM captions, one per mask.
            n_questions_per_object: Overrides the ctor default.

        Returns:
            ``(prompts, messages)`` — ``prompts`` is a shuffled list of
            ``"question Answer: answer"`` strings; ``messages`` is the
            structured user/assistant message list.
        """
        if len(masks) != len(captions):
            raise ValueError(
                f"masks and captions must align: got {len(masks)} masks and "
                f"{len(captions)} captions."
            )
        n = (
            self.n_questions_per_object
            if n_questions_per_object is None
            else n_questions_per_object
        )
        k = max(1, min(n, len(_DETAILED_QUESTION_TEMPLATES)))

        prompts = []
        for caption in captions:
            caption = (caption or "").strip()
            if not caption:
                continue
            questions = random.sample(_DETAILED_QUESTION_TEMPLATES, k)
            for question in questions:
                frame = random.choice(_DETAILED_ANSWER_FRAMES)
                answer = frame.format(caption=caption)
                prompts.append(f"{question} Answer: {answer}")

        random.shuffle(prompts)
        messages = self._messages_from_prompts(prompts)
        return prompts, messages

    # ------------------------------------------------------------------
    # Dataset.map transform (mirrors Localizer.apply_transform)
    # ------------------------------------------------------------------
    def apply_transform(self, example, images):
        """``datasets.map`` transform: read ``example["masks"]``, write DAM output.

        Adds two columns:
          * ``dam_captions`` — detailed DAM caption per mask (aligned with
            ``example["masks"]``), a drop-in richer replacement for the
            class-label ``captions`` column.
          * ``dam_messages`` — spatial-VQA messages built from those captions
            via :meth:`generate_qa_pairs`.

        Expects ``example["masks"]`` to have been populated by the
        location-refinement stage (:class:`Localizer`).
        """
        is_batched = (
            isinstance(example[images], list)
            and isinstance(example[images][0], (list, Image.Image))
        )

        if is_batched:
            all_captions, all_messages = [], []
            for i, img_list in enumerate(example[images]):
                image = img_list[0] if isinstance(img_list, list) else img_list
                if not isinstance(image, Image.Image):
                    raise ValueError("Expected a PIL Image.")
                if image.mode != "RGB":
                    image = image.convert("RGB")

                masks = example["masks"][i] if "masks" in example else []
                captions = self.describe_regions(image, masks)
                _, messages = self.generate_qa_pairs(masks, captions)
                all_captions.append(captions)
                all_messages.append(messages)

            example["dam_captions"] = all_captions
            example["dam_messages"] = all_messages
        else:
            image = (
                example[images][0]
                if isinstance(example[images], list)
                else example[images]
            )
            if not isinstance(image, Image.Image):
                raise ValueError("Expected a PIL Image.")
            if image.mode != "RGB":
                image = image.convert("RGB")

            masks = example["masks"] if "masks" in example else []
            captions = self.describe_regions(image, masks)
            _, messages = self.generate_qa_pairs(masks, captions)
            example["dam_captions"] = captions
            example["dam_messages"] = messages

        return example
