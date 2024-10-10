import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.datasets import Dataloader
from vqasynth.embeddings import EmbeddingGenerator

def main(output_dir, source_repo_id, image_col):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataloader = Dataloader(output_dir)
    embedding_generator = EmbeddingGenerator()

    dataset = dataloader.load_dataset(source_repo_id)

    def process_row(example):
        embedding = embedding_generator.run(example[image_col])
        example['embedding'] = embedding
        return example

    dataset = dataset.map(process_row)
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
        "--image_col",
        type=str,
        required=True,
        help="Column containing PIL.Image images",
    )
    args = parser.parse_args()
    main(args.output_dir, args.source_repo_id, args.image_col)
