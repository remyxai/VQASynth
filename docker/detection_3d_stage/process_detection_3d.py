import os
import argparse
import pandas as pd
from vqasynth.datasets import Dataloader
from vqasynth.detection_3d import Detection3DGenerator
from vqasynth.utils import filter_null


def main(output_dir, source_repo_id, images):
    generator = Detection3DGenerator()
    dataloader = Dataloader(output_dir)

    # Load dataset (carries `captions` from localization + `pointclouds` from
    # scene fusion).
    dataset = dataloader.load_dataset(source_repo_id)

    # Drop any prior outputs so re-runs don't accumulate stale columns.
    for col in ["detection_3d_boxes", "detection_3d_prompts", "detection_3d_messages"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    # Compute one 3D box per detected object and emit Molmo <point3d> QA pairs.
    dataset = dataset.map(
        generator.apply_transform,
        batched=False,
    )

    # Filter out nulls / rows where no boxes survived (same pattern as the other
    # stages, then a non-empty-messages filter like prompt_stage).
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)
    dataset = dataset.filter(
        lambda batch: [len(msg_list) > 0 for msg_list in batch["detection_3d_messages"]],
        batched=True,
        batch_size=32,
    )

    # Save the processed dataset to disk.
    dataloader.save_to_disk(dataset)

    print("3D object-detection synthesis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize 3D object-detection QA pairs from per-object point clouds",
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
    args = parser.parse_args()

    main(args.output_dir, args.source_repo_id, args.images)
