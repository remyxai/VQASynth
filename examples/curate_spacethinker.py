"""Curate a uniform subset of SpaceThinker via max-min pairwise-distance selection.

Produces two subsets of the same size from ``remyxai/SpaceThinker``:

  * uniform -- farthest-point (max-min pairwise distance) selection over CLIP
               image embeddings (SafeRL-Lab/data-uniformity, arXiv:2506.24120).
  * random  -- random sampling baseline.

This is the direct comparison the maintainer described in issue #28: two
same-sized subsets, one uniform and one random, ready to feed identical LoRA
training runs and compare convergence speed + final accuracy.

CLIP embeddings are computed ONCE (via ``vqasynth.embeddings.EmbeddingGenerator``)
and reused for both strategies, then the curator reads the resulting
``embedding`` column without any model reload.

Requirements: OpenAI CLIP (installed by the vqasynth embeddings stage) and
network access to pull ``remyxai/SpaceThinker``. No GPU needed -- CLIP ViT-B/32
runs on CPU.

Usage:
    OUTPUT_DIR=./curated python examples/curate_spacethinker.py
"""
import os

from datasets import load_dataset

from vqasynth.embeddings import EmbeddingGenerator
from vqasynth.curate import Curator, write_manifest

SOURCE_REPO_ID = "remyxai/SpaceThinker"
FRACTION = 0.25  # 25% subset, as suggested in issue #28
IMAGES = "image"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./curated")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {SOURCE_REPO_ID} ...")
    dataset = load_dataset(SOURCE_REPO_ID)

    # Compute CLIP ViT-B/32 image embeddings once; apply_transform writes the
    # `embedding` column that Curator reads directly (no per-strategy recompute).
    print("Computing CLIP ViT-B/32 image embeddings (CPU) ...")
    generator = EmbeddingGenerator()  # ViT-B/32, auto CPU/CUDA
    dataset = dataset.map(
        generator.apply_transform,
        fn_kwargs={"images": IMAGES},
        batched=True,
        batch_size=32,
    )

    for strategy in ("farthest", "random"):
        curator = Curator(embedding_source=None, seed=0, images=IMAGES)
        # Same dataset object for both strategies -> same-sized subsets for a
        # fair ablation; Curator.curate returns a fresh slice, never mutates.
        subset, manifest = curator.curate(
            dataset,
            fraction=FRACTION,
            strategy=strategy,
            source_repo_id=SOURCE_REPO_ID,
        )
        variant_dir = os.path.join(OUTPUT_DIR, f"spacethinker_{strategy}")
        subset.save_to_disk(variant_dir)
        write_manifest(manifest, os.path.join(variant_dir, "curate_manifest.json"))
        print(
            f"[{strategy:>8}] selected {manifest['selected_count']}/{manifest['source_count']} "
            f"samples -> {variant_dir}"
        )

    print(f"\nDone. Curated variants written under {OUTPUT_DIR}/")
    print("Point your LoRA training config's dataset.repo_id at either variant to compare.")


if __name__ == "__main__":
    main()
