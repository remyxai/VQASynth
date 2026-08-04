import os
import argparse
from vqasynth.datasets import Dataloader
from vqasynth.pose import PoseAnnotator
from vqasynth.utils import filter_null


def main(output_dir, source_repo_id, images, backend, max_questions, seed):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataloader = Dataloader(output_dir)
    annotator = PoseAnnotator(backend=backend, max_questions=max_questions, seed=seed)

    # Load dataset
    dataset = dataloader.load_dataset(source_repo_id)

    # Run the pose backend + QA emitter over the image column, appending a
    # ``pose_messages`` column (one list of SFT chat samples per image) — the
    # same column-append convention the other stages use.
    dataset = dataset.map(
        annotator.apply_transform,
        fn_kwargs={"images": images},
        batched=True,
        batch_size=4,
    )

    # Drop rows where pose detection produced no samples.
    dataset = dataset.filter(filter_null, batched=True, batch_size=32)

    # Save the processed dataset to disk
    dataloader.save_to_disk(dataset)
    print("Pose keypoint annotation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose keypoint annotation", add_help=True)
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
        default="mediapipe",
        help="Keypoint backend name (default: mediapipe, CPU-friendly, Apache-2.0)",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=12,
        help="Cap on SFT samples emitted per detected person",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic QA generation",
    )
    args = parser.parse_args()
    main(
        args.output_dir,
        args.source_repo_id,
        args.images,
        args.backend,
        args.max_questions,
        args.seed,
    )
