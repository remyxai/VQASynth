"""Example: VQASynth SAM masks -> detailed DAM captions -> spatial-VQA pairs.

This mirrors the shape of the DAM repo's
``examples/dam_with_sam_self_contained.py`` (SAM-mask -> DAM-caption), but
sources the masks from :class:`vqasynth.localize.Localizer` instead of a
standalone SAM call, and turns each ``(image, mask, caption)`` triple into
spatial-VQA training samples via :class:`vqasynth.describe_anything.DescribeAnything`.

NOTE: this is a GPU-host example — both ``Localizer`` (Molmo + SAM2) and DAM
load real weights. The module itself and ``tests/test_describe_anything.py``
run without a GPU using an injected stub DAM callable.

Refs:
  - DAM:      https://github.com/NVlabs/describe-anything
  - Issue #51: https://github.com/remyxai/VQASynth/issues/51
"""
import argparse

from PIL import Image

from vqasynth.localize import Localizer
from vqasynth.describe_anything import DescribeAnything, DEFAULT_MODEL_ID


def run(image_path, model_id, n_questions_per_object):
    image = Image.open(image_path).convert("RGB")

    # 1) Localization + segmentation -> one SAM mask per object.
    localizer = Localizer(
        captioner_type="molmo",
        segmenter_model="facebook/sam2-hiera-small",
    )
    masks, prompts, captions = localizer.run(image)

    # 2) DAM -> detailed per-region caption for each mask.
    dam = DescribeAnything(
        model_id=model_id,
        n_questions_per_object=n_questions_per_object,
    )
    detailed_captions = dam.describe_regions(image, masks)

    # 3) QA-pair emitter -> spatial-VQA training samples.
    qa_prompts, messages = dam.generate_qa_pairs(masks, detailed_captions)

    print(f"Found {len(masks)} region(s):\n")
    for caption, detailed in zip(captions, detailed_captions):
        print(f"  - {caption}")
        print(f"      DAM: {detailed}\n")

    print(f"Generated {len(qa_prompts)} QA pair(s); first conversation:\n")
    for message in messages[:2]:
        role = message["role"]
        text = " ".join(
            part["text"] for part in message["content"] if part.get("type") == "text"
        )
        print(f"  [{role}] {text}")

    return detailed_captions, messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Caption Localizer SAM masks with DAM and emit VQA pairs",
        add_help=True,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to an input image",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Self-contained DAM variant (default: nvidia/DAM-3B-Self-Contained)",
    )
    parser.add_argument(
        "--n_questions_per_object",
        type=int,
        default=2,
        help="QA pairs emitted per region",
    )
    args = parser.parse_args()

    run(args.image, args.model_id, args.n_questions_per_object)
