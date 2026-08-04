"""Docker entrypoint for the multi-view correspondence stage.

Loads a Hugging Face dataset whose ``--images`` column holds a list of frames
per example (Ego4D-style clips), pairs adjacent frames, extracts OpenCV point
correspondences, and writes Molmo ``<point>`` pointing-VLM training messages.

Mirrors ``docker/prompt_stage/process_prompts.py``: same argparse surface, same
``Dataloader`` save/push pattern, same ``messages`` schema so the output is
shape-compatible with the rest of the VQASynth training pipeline.
"""
import argparse

from datasets import Features, Sequence, Value

from vqasynth.correspondence import CorrespondenceExtractor
from vqasynth.datasets import Dataloader
from vqasynth.utils import filter_null

message_schema = {
    "role": Value("string"),
    "index": Value("int64"),   # int64 is nullable so None is allowed.
    "text": Value("string"),
    "type": Value("string"),
}


def build_new_features(dataset):
    new_features = dataset["train"].features.copy()
    new_features["messages"] = Sequence(message_schema)
    return new_features


def save_and_push_datasets(dataset, output_dir, target_repo_name, images, dataloader):
    """Save the full dataset and a messages-only view, then push to the hub."""
    dataloader.save_to_disk(dataset)
    dataloader.push_to_hub(dataset, f"{target_repo_name}_full")

    final_dataset = dataset.select_columns([images, "messages"])
    dataloader.push_to_hub(final_dataset, target_repo_name)


def main(output_dir, source_repo_id, target_repo_name, images):
    extractor = CorrespondenceExtractor()
    dataloader = Dataloader(output_dir)
    dataset = dataloader.load_dataset(source_repo_id)

    for col in ["messages"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    dataset = dataset.cast(build_new_features(dataset))
    dataset = dataset.map(extractor.apply_transform, fn_kwargs={"images": images}, batched=False)

    dataset = dataset.filter(
        lambda batch: [len(msg_list) > 0 for msg_list in batch["messages"]],
        batched=True,
        batch_size=32,
    )

    save_and_push_datasets(dataset, output_dir, target_repo_name, images, dataloader)

    print(f"Processed correspondence messages for {target_repo_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract multi-view point correspondences -> pointing-VLM messages",
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
        help="Source huggingface dataset repo id (multi-view / clip images)",
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
        help="Column containing a list of PIL.Image frames per example",
    )
    args = parser.parse_args()

    main(args.output_dir, args.source_repo_id, args.target_repo_name, args.images)
