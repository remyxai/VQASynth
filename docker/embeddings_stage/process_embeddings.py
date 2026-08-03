import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.datasets import Dataloader
from vqasynth.embeddings import EmbeddingGenerator, list_embedding_backends
from vqasynth.utils import filter_null

def main(output_dir, source_repo_id, images, backend="clip", model_name=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataloader = Dataloader(output_dir)
    # Only forward model_name when explicitly set so each backend keeps its own default.
    kwargs = {"backend": backend}
    if model_name:
        kwargs["model_name"] = model_name
    embedding_generator = EmbeddingGenerator(**kwargs)

    # Load dataset
    dataset = dataloader.load_dataset(source_repo_id)

    # Apply the embedding generation transformation with batching
    dataset = dataset.map(
        embedding_generator.apply_transform,
        fn_kwargs={'images': images},
        batched=True,
        batch_size=32
    )

    # Filter out nulls using the updated function
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # Save the processed dataset to disk
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
    parser.add_argument(
        "--backend",
        type=str,
        default="clip",
        choices=list_embedding_backends(),
        help="Multimodal embedding backend (clip | transformers). "
        "Use 'transformers' to load CLIP/SigLIP/LLM2CLIP checkpoints from HuggingFace.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name/checkpoint for the backend (defaults to backend default, "
        "e.g. microsoft/LLM2CLIP-OpenAI-B-16 with --backend transformers).",
    )
    args = parser.parse_args()
    main(args.output_dir, args.source_repo_id, args.images, args.backend, args.model_name)
