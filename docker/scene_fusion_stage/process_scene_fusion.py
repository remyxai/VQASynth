import os
import pickle
import argparse
import pandas as pd
from vqasynth.datasets import Dataloader
from vqasynth.scene_fusion import SpatialSceneConstructor


def main(output_dir, source_repo_id, images):
    spatial_scene_constructor = SpatialSceneConstructor()
    dataloader = Dataloader(output_dir)

    dataset = dataloader.load_dataset(source_repo_id)

    point_cloud_dir = os.path.join(output_dir, "pointclouds")
    if not os.path.exists(point_cloud_dir):
        os.makedirs(point_cloud_dir)

    dataset = dataset.map(lambda example, idx: spatial_scene_constructor.apply_transform(example, idx, output_dir, images), with_indices=True)

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
