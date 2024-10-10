import os
import pickle
import argparse
import pandas as pd
from vqasynth.datasets import Dataloader
from vqasynth.scene_fusion import SpatialSceneConstructor


def main(output_dir, source_repo_id, image_col):
    spatial_scene_constructor = SpatialSceneConstructor()
    dataloader = Dataloader(output_dir)

    dataset = dataloader.load_dataset(source_repo_id)

    point_cloud_dir = os.path.join(output_dir, "pointclouds")
    if not os.path.exists(point_cloud_dir):
        os.makedirs(point_cloud_dir)

    def process_row(example, idx):
        # Run spatial scene constructor and get point cloud data and canonicalization flag
        pcd_data, canonicalized = spatial_scene_constructor.run(
            str(idx), 
            example[image_col],
            example["depth_map"],
            example["focallength"],
            example["masks"],
            output_dir
        )
        # Add point cloud data and canonicalization flag to the example
        example["pointclouds"] = pcd_data
        example["is_canonicalized"] = canonicalized
        return example

    dataset = dataset.map(process_row, with_indices=True)
    dataloader.save_to_disk(dataset)
    print("Scene fusion complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
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
        "--image_col",
        type=str,
        required=True,
        help="Column containing PIL.Image images",
    )
    args = parser.parse_args()

    main(args.output_dir, args.source_repo_id, args.image_col)
