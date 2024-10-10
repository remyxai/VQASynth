import os
import pickle
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from vqasynth.datasets import Dataloader
from vqasynth.depth import DepthEstimator


def main(output_dir, source_repo_id, image_col):
    dataloader = Dataloader(output_dir)
    depth = DepthEstimator()

    dataset = dataloader.load_dataset(source_repo_id)

    def process_row(example):
        depth_map, focallength = depth.run(example[image_col])
        example['depth_map'] = depth_map
        example['focallength'] = focallength
        return example

    dataset = dataset.map(process_row)
    dataloader.save_to_disk(dataset)
    print("Depth extraction complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract depth from images in dataset", add_help=True
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
        "--image_col",
        type=str,
        required=True,
        help="Column containing PIL.Image images",
    )
    args = parser.parse_args()

    main(args.output_dir, args.source_repo_id, args.image_col)
