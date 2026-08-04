"""Docker entrypoint process: region-level DAM captioning over SAM masks.

Consumes the ``masks`` column produced by the location-refinement stage
(``vqasynth.localize.Localizer``) and writes a richer per-region caption column
(``dam_captions``) plus spatial-VQA messages (``dam_messages``).

Mirrors the shape of ``docker/scene_fusion_stage/process_scene_fusion.py`` /
``docker/location_refinement_stage/process_location_refinement.py``.
"""
import os
import argparse

from vqasynth.datasets import Dataloader
from vqasynth.describe_anything import DescribeAnything, DEFAULT_MODEL_ID
from vqasynth.utils import filter_null


def main(output_dir, source_repo_id, images, model_id, batch_size):
    dam = DescribeAnything(model_id=model_id)
    dataloader = Dataloader(output_dir)

    # Load dataset
    dataset = dataloader.load_dataset(source_repo_id)

    # Drop stale outputs so re-runs don't collide with the inferred schema.
    for col in ["dam_captions", "dam_messages"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    # Caption every mask produced by the localization stage.
    dataset = dataset.map(
        dam.apply_transform,
        fn_kwargs={"images": images},
        batched=True,
        batch_size=batch_size,
    )

    # Filter out rows where captioning produced nothing usable.
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # Save the processed dataset to disk
    dataloader.save_to_disk(dataset)

    print("Describe Anything captioning complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Caption SAM masks with NVIDIA's Describe Anything Model",
        add_help=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to local dataset cache",
    )
    parser.add_argument(
        "--source_repo_id",
        type=str,
        required=True,
        help="Source huggingface dataset repo id",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Column containing PIL.Image images",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=os.environ.get("DAM_MODEL_ID", DEFAULT_MODEL_ID),
        help="Self-contained DAM variant to load "
        "(env: DAM_MODEL_ID; default: nvidia/DAM-3B-Self-Contained)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="datasets.map batch size",
    )
    args = parser.parse_args()

    main(
        args.output_dir,
        args.source_repo_id,
        args.images,
        args.model_id,
        args.batch_size,
    )
