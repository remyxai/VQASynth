import os
import shutil
from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import HfApi, DatasetCard


class Dataloader:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.api = HfApi()
        self.dataset_name = None
        self.dataset_path = None

    def load_dataset(self, repo_id, trust_remote_code=True):
        """Load dataset from cache or fetch it from Hugging Face Hub."""
        self.dataset_name = repo_id.split("/")[-1]
        self.dataset_path = os.path.join(self.cache_dir, self.dataset_name)

        if os.path.exists(self.dataset_path):
            dataset = load_from_disk(self.dataset_path, trust_remote_code=trust_remote_code)
        else:
            dataset = load_dataset(repo_id, cache_dir=self.cache_dir, trust_remote_code=trust_remote_code)
            dataset.save_to_disk(self.dataset_path)

        return dataset

    def save_to_disk(self, dataset):
        """Save a Hugging Face dataset to the cache directory with overwrite handling."""
        try:
            # Temporary save path
            temp_path = os.path.join(self.cache_dir, f"{self.dataset_name}_temp")
            final_path = os.path.join(self.cache_dir, self.dataset_name)

            # Save dataset to a temporary location
            dataset.save_to_disk(temp_path)

            # Replace the original dataset by moving the temp version
            shutil.rmtree(final_path)  # Remove the old dataset
            shutil.move(temp_path, final_path)  # Move the new dataset to the original path

            print(f"Dataset successfully saved to {final_path}")
        except Exception as e:
            print(f"An error occurred while saving dataset: {str(e)}")

    def _tag_dataset(self, repo_id):
        card = DatasetCard.load(repo_id)
        default_tags = ["vqasynth", "remyx"]

        if "tags" not in card.data:
            card.data["tags"] = default_tags
        elif not any(tag in card.data["tags"] for tag in default_tags):
            card.data["tags"].extend(default_tags)

        card.push_to_hub(repo_id)

    def push_to_hub(self, dataset, repo_name):
        """Push the final dataset to Hugging Face Hub."""
        try:
            repo_id = f"{self.api.whoami()['name']}/{repo_name}"
            dataset.push_to_hub(repo_id)
            self._tag_dataset(repo_id)
        except Exception as e:
            print(f"An error occurred while pushing to Hugging Face Hub: {str(e)}")
