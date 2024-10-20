import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.datasets import Dataloader
from vqasynth.embeddings import TagFilter
from vqasynth.utils import filter_null


def main(output_dir, source_repo_id, include_tags, exclude_tags, confidence_threshold=0.7):
    tag_filter = TagFilter()
    dataloader = Dataloader(output_dir)

    dataset = dataloader.load_dataset(source_repo_id)

    include_tags = [tag.strip() for tag in include_tags.split(",")]
    exclude_tags = [tag.strip() for tag in exclude_tags.split(",")]

    def process_and_filter(example):
        """
        Apply the tag filtering and ensure no null values in the example.
        Works for both single and batched examples.
        """
        if isinstance(next(iter(example.values())), list):
            # Batched input: process each row in the batch
            return [
                tag_filter.filter_by_tag(tag, include_tags, exclude_tags) and
                all(value is not None for value in row)
                for tag, row in zip(example['tag'], zip(*example.values()))
            ]
        else:
            # Single example: process directly
            return tag_filter.filter_by_tag(example['tag'], include_tags, exclude_tags) and \
                   all(value is not None for value in example.values())

    dataset = dataset.map(tag_filter.apply_transform, fn_kwargs={'tags': include_tags + exclude_tags}, batched=True, batch_size=32)

    # Filter out examples in a single pass (nulls and tag filtering combined)
    dataset_filtered = dataset.filter(process_and_filter, batched=True, batch_size=32)

    dataloader.save_to_disk(dataset_filtered)

    print(f"Dataset filtering complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter dataset by tags", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to output dataset directory")
    parser.add_argument("--source_repo_id", type=str, required=True, help="Source huggingface dataset repo id")
    parser.add_argument("--include_tags", type=str, required=False, default=None, help="Comma-separated list of tags to include (optional)")
    parser.add_argument("--exclude_tags", type=str, required=False, default=None, help="Comma-separated list of tags to exclude (optional)")
    args = parser.parse_args()
    main(args.output_dir, args.source_repo_id, args.include_tags, args.exclude_tags)
