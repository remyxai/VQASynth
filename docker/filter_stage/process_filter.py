import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.datasets import Dataloader
from vqasynth.embeddings import TagFilter


def main(output_dir, source_repo_id, include_tags, exclude_tags, confidence_threshold=0.7):
    tag_filter = TagFilter()
    dataloader = Dataloader(output_dir)

    dataset = dataloader.load_dataset(source_repo_id)

    include_tags = include_tags.strip().split(",")
    exclude_tags = exclude_tags.strip().split(",")

    def process_row(example):
        example['tag'] = tag_filter.get_best_matching_tag(
            example['embedding'], include_tags + exclude_tags
        )
        return example

    dataset = dataset.map(process_row)

    dataset_filtered = dataset.filter(
        lambda example: tag_filter.filter_by_tag(
            example['tag'], include_tags, exclude_tags
        )
    )

    dataloader.save_to_disk(dataset_filtered)

    print(f"Processed and updated {filename}, filtered out {len(df) - len(df_filtered)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter dataset by tags", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to output dataset directory")
    parser.add_argument("--source_repo_id", type=str, required=True, help="Source huggingface dataset repo id")
    parser.add_argument("--include_tags", type=str, required=False, default=None, help="Comma-separated list of tags to include (optional)")
    parser.add_argument("--exclude_tags", type=str, required=False, default=None, help="Comma-separated list of tags to exclude (optional)")
    args = parser.parse_args()
    main(args.output_dir, args.source_repo_id, args.include_tags, args.exclude_tags)
