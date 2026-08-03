import os
import argparse

from vqasynth.datasets import Dataloader
from vqasynth.curate import Curator, write_manifest


def main(output_dir, source_repo_id, target_repo_name, strategy, fraction, count,
         seed, metric, images, split, push_to_hub):
    dataloader = Dataloader(output_dir)
    dataset = dataloader.load_dataset(source_repo_id)

    # embedding_source=None -> read the precomputed `embedding` column written by
    # the embeddings stage; fall back to CLIP ViT-B/32 only if absent.
    curator = Curator(embedding_source=None, seed=seed, metric=metric, images=images)
    subset, manifest = curator.curate(
        dataset,
        fraction=fraction,
        count=count,
        strategy=strategy,
        split=split,
        source_repo_id=source_repo_id,
    )

    # Emit the curated subset as its own on-disk variant + manifest.
    dataset_short = source_repo_id.split("/")[-1]
    variant_dir = os.path.join(output_dir, f"{dataset_short}_{strategy}")
    subset.save_to_disk(variant_dir)
    write_manifest(manifest, os.path.join(variant_dir, "curate_manifest.json"))

    if push_to_hub:
        dataloader.push_to_hub(subset, f"{target_repo_name}_{strategy}")

    print(
        f"Curation complete: {manifest['selected_count']}/{manifest['source_count']} "
        f"samples selected via {strategy} (metric={metric}, seed={seed})."
    )
    print(f"Curated subset written to {variant_dir}")
    print(f"Manifest written to {os.path.join(variant_dir, 'curate_manifest.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate a uniform (or random) subset of a HuggingFace dataset", add_help=True
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Path to local dataset cache")
    parser.add_argument("--source_repo_id", type=str, required=True, help="Source HuggingFace dataset repo id")
    parser.add_argument("--target_repo_name", type=str, required=False, default="vqasynth_sample_curated",
                        help="Target repo name suffix when pushing to the Hub")
    parser.add_argument("--strategy", type=str, required=False, default="farthest",
                        choices=["farthest", "random"], help="Selection strategy")
    parser.add_argument("--fraction", type=float, required=False, default=None,
                        help="Fraction of the dataset to keep (e.g. 0.25)")
    parser.add_argument("--count", type=str, required=False, default="",
                        help="Absolute number of samples to keep (overrides fraction)")
    parser.add_argument("--seed", type=int, required=False, default=0, help="RNG seed")
    parser.add_argument("--metric", type=str, required=False, default="euclidean",
                        choices=["euclidean", "cosine"], help="Distance metric for farthest-point selection")
    parser.add_argument("--images", type=str, required=False, default="image",
                        help="Image column name (used only when computing embeddings on the fly)")
    parser.add_argument("--split", type=str, required=False, default="train",
                        help="Split to curate when the dataset has multiple splits")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the curated subset to the HuggingFace Hub")
    args = parser.parse_args()

    count = int(args.count) if args.count.strip() else None
    # count takes precedence over fraction (the entrypoint always passes both,
    # defaulting fraction to 0.25); null fraction when count is set so the
    # curator doesn't see them as conflicting.
    fraction = None if count is not None else args.fraction
    main(
        output_dir=args.output_dir,
        source_repo_id=args.source_repo_id,
        target_repo_name=args.target_repo_name,
        strategy=args.strategy,
        fraction=fraction,
        count=count,
        seed=args.seed,
        metric=args.metric,
        images=args.images,
        split=args.split,
        push_to_hub=args.push_to_hub,
    )
