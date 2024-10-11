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

def main(output_dir, source_repo_id, target_repo_name, images):
    prompt_generator = PromptGenerator()
    dataloader = Dataloader(output_dir)

    dataset = dataloader.load_dataset(source_repo_id)

    # Process each row in the Hugging Face dataset by applying the prompt generator logic
    dataset = dataset.map(lambda example: prompt_generator.apply_transform(example))
    final_dataset = dataset.select_columns([images, "messages"])

    dataloader.save_to_disk(final_dataset)
    dataloader.push_to_hub(final_dataset, target_repo_name)

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
