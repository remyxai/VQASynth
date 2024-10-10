import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from vqasynth.datasets import Dataloader
from vqasynth.localize import Localizer


def main(output_dir, source_repo_id, images):
    dataloader = Dataloader(output_dir)
    localizer = Localizer()

    dataset = dataloader.load_dataset(source_repo_id)

    def process_row(example):
        masks, bboxes, captions = localizer.run(example[images])
        example['masks'] = masks
        example['bboxes'] = bboxes
        example['captions'] = captions
        return example

    dataset = dataset.map(process_row)
    dataloader.save_to_disk(dataset)
    print("Localization complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Localize and describe objects in images", add_help=True
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
