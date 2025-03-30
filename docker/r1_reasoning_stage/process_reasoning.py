import os
import argparse
from vqasynth.datasets import Dataloader
from vqasynth.utils import filter_null
from vqasynth.r1_reasoning import R1Reasoner

def save_and_push_datasets(dataset, output_dir, target_repo_name, images, dataloader):
    """
    Save the full dataset and a dataset with selected columns, then push to the hub.

    Args:
        dataset: The full dataset after processing.
        output_dir: Directory to save the dataset.
        target_repo_name: The name of the target repository.
        images: The column name for images.
        dataloader: Dataloader instance to handle saving and pushing datasets.
    """
    dataloader.save_to_disk(dataset)
    dataloader.push_to_hub(dataset, f"{target_repo_name}_full_reasoning")

    final_dataset = dataset.select_columns([images, "messages", "input", "output", "reasoning"])
    dataloader.push_to_hub(final_dataset, target_repo_name)

def main(
    output_dir,
    source_repo_id,
    target_repo_name,
    images_column,
    text_column,
    api_key,
    model,
    delay
):
    dataloader = Dataloader(output_dir)

    reasoner = R1Reasoner(
        api_key=api_key,
        model=model,
        image_column=images_column,
        text_column=text_column,
        delay=delay
    )

    dataset = dataloader.load_dataset(source_repo_id)

    dataset = dataset.map(
        reasoner.apply_transform,
        fn_kwargs={
            "images": images_column,
            "text": text_column
        },
        batched=True,
        batch_size=2
    )

    dataset = dataset.filter(filter_null, batched=True, batch_size=32)
    print(dataset["train"][0])

    save_and_push_datasets(dataset, output_dir, target_repo_name, images_column, dataloader)

    print("R1 reasoning step complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply R1 Reasoner to dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to local dataset cache")
    parser.add_argument("--source_repo_id", type=str, required=True, help="Source HuggingFace dataset repo id")
    parser.add_argument("--target_repo_name", type=str, required=True, help="Target huggingface dataset repo id")
    parser.add_argument("--images_column", type=str, required=True, help="Name of the image column")
    parser.add_argument("--text_column", type=str, default="messages", help="Name of the text column")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--delay", type=int, default=1, help="Delay in seconds per call")
    args = parser.parse_args()

    main(
        args.output_dir,
        args.source_repo_id,
        args.target_repo_name,
        args.images_column,
        args.text_column,
        args.api_key,
        args.model,
        args.delay
    )
