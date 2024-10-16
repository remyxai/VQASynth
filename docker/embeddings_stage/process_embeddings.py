import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.datasets import Dataloader
from vqasynth.embeddings import EmbeddingGenerator

def main(output_dir, source_repo_id, images):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataloader = Dataloader(output_dir)
    embedding_generator = EmbeddingGenerator()

    dataset = dataloader.load_dataset(source_repo_id)
    dataset = dataset.map(lambda example: embedding_generator.apply_transform(example, images))
    # filter nulls
    dataset = dataset.filter(lambda example: all(value is not None for value in example.values()))

    dataloader.save_to_disk(dataset)
    print("Embedding extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Embedding extraction", add_help=True)
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
