import os
import cv2
import json
import pickle
import random
import itertools
import argparse
import numpy as np
import pandas as pd
from vqasynth.datasets import Dataloader
from vqasynth.prompts import PromptGenerator
from vqasynth.utils import filter_null

def save_and_push_datasets(dataset, output_dir, target_repo_name, images, dataloader):
    """
    Save the full dataset and a dataset with selected columns, then push to the hub.

    Args:
        dataset: The full dataset after processing.
        output_dir: Directory to save the dataset.
        target_repo_name: The name of the target repository.
        images: The column name for images.
        dataloader: Dataloader instance to handle saving and pushing datasets.
    """
    dataloader.save_to_disk(dataset)
    dataloader.push_to_hub(dataset, f"{target_repo_name}_full")

    final_dataset = dataset.select_columns([images, "messages"])
    dataloader.push_to_hub(final_dataset, target_repo_name)

def main(output_dir, source_repo_id, target_repo_name, images):
    prompt_generator = PromptGenerator()
    dataloader = Dataloader(output_dir)

    dataset = dataloader.load_dataset(source_repo_id)
    dataset = dataset.map(
        prompt_generator.apply_transform,
        batched=True,
        batch_size=32
    )

    dataset = dataset.filter(lambda x: [len(msg) > 0 for msg in x['messages']], batched=True, batch_size=32)
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    save_and_push_datasets(dataset, output_dir, target_repo_name, images, dataloader)

    print(f"Processed and updated dataset with formatted messages.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prompts from metadata", add_help=True)
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
        "--target_repo_name",
        type=str,
        required=True,
        help="Target huggingface dataset repo id",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Column containing PIL.Image images",
    )
    args = parser.parse_args()

    main(args.output_dir, args.source_repo_id, args.target_repo_name, args.images)
