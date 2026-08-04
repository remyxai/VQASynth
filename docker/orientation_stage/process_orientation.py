import argparse
from vqasynth.datasets import Dataloader
from vqasynth.orientation import OrientationEstimator
from vqasynth.utils import filter_null


def main(output_dir, source_repo_id, images, masks):
    dataloader = Dataloader(output_dir)
    orientation = OrientationEstimator()

    # Load dataset
    dataset = dataloader.load_dataset(source_repo_id)

    # Apply the orientation estimator transformation with batching. Each image's
    # per-object masks are isolated and oriented independently.
    dataset = dataset.map(
        orientation.apply_transform,
        fn_kwargs={'images': images, 'masks': masks},
        batched=True,
        batch_size=32
    )

    # Filter out rows where orientation failed
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # Save the processed dataset to disk
    dataloader.save_to_disk(dataset)

    print("Orientation extraction complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract object-level orientation from images in dataset",
        add_help=True,
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
    parser.add_argument(
        "--masks",
        type=str,
        default="masks",
        help="Column containing per-object segmentation masks",
    )
    args = parser.parse_args()

    main(args.output_dir, args.source_repo_id, args.images, args.masks)
