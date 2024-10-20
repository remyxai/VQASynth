import os
import pickle
import argparse
import pandas as pd
from functools import partial
from vqasynth.datasets import Dataloader
from vqasynth.scene_fusion import SpatialSceneConstructor
from vqasynth.utils import filter_null

def main(output_dir, source_repo_id, images):
    spatial_scene_constructor = SpatialSceneConstructor()
    dataloader = Dataloader(output_dir)

    # Load dataset
    dataset = dataloader.load_dataset(source_repo_id)

    # Create the directory for saving point clouds
    point_cloud_dir = os.path.join(output_dir, "pointclouds")
    if not os.path.exists(point_cloud_dir):
        os.makedirs(point_cloud_dir)

    # Apply the spatial scene construction transformation with batching
    dataset = dataset.map(
        partial(spatial_scene_constructor.apply_transform, output_dir=output_dir, images=images),
        with_indices=True,  # Pass index for each example
        batched=True,
        batch_size=32
    )

    # Filter out nulls with the updated filter_null function
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # Save the processed dataset to disk
    dataloader.save_to_disk(dataset)

    print("Scene fusion complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
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
