import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd

from vqasynth.datasets import Dataloader
from vqasynth.localize import Localizer
from vqasynth.utils import filter_null

def main(output_dir, source_repo_id, images):
    # 1) Instantiate the Dataloader
    dataloader = Dataloader(output_dir)
    
    # 2) Create the Localizer with Molmo + SAM2 (points)
    #    You can choose whichever SAM2 model variant:
    #    e.g. "facebook/sam2-hiera-small", "facebook/sam2-hiera-large", etc.
    localizer = Localizer(
        captioner_type="florence",
        segmenter_model="facebook/sam2-hiera-small"
    )

    # 3) Load the dataset
    dataset = dataloader.load_dataset(source_repo_id)

    # 4) Apply the localizer transformation with batching and pass use_points=True
    dataset = dataset.map(
        localizer.apply_transform,
        fn_kwargs={'images': images},
        batched=True,
        batch_size=1,
    )

    # 5) Filter out nulls
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # 6) Save the processed dataset
    dataloader.save_to_disk(dataset)

    print("Localization complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Localize and describe objects in images", 
        add_help=True
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

