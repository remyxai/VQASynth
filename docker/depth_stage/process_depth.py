import os
import pickle
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from vqasynth.datasets import Dataloader
from vqasynth.depth import DepthEstimator
from vqasynth.utils import filter_null


def main(output_dir, source_repo_id, images):
    dataloader = Dataloader(output_dir)
    depth = DepthEstimator()

    # Load dataset
    dataset = dataloader.load_dataset(source_repo_id)

    # Apply the depth estimator transformation with batching
    dataset = dataset.map(
        depth.apply_transform,
        fn_kwargs={'images': images},
        batched=True,
        batch_size=32
    )

    # Filter out nulls with the updated filter_null function
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # Save the processed dataset to disk
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
        "--images",
        type=str,
        required=True,
        help="Column containing PIL.Image images",
    )
    args = parser.parse_args()

    main(args.output_dir, args.source_repo_id, args.images)
