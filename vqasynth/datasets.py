import os
import logging
from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import HfApi, DatasetCard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Dataloader:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.api = HfApi()
        self.dataset_name = None
        self.dataset_path = None

    def load_dataset(self, repo_id):
        """Load dataset from cache or fetch it from Hugging Face Hub."""
        self.dataset_name = repo_id.split("/")[-1]
        self.dataset_path = os.path.join(self.cache_dir, self.dataset_name)

        if os.path.exists(self.dataset_path):
            dataset = load_from_disk(self.dataset_path)
        else:
            dataset = load_dataset(repo_id, cache_dir=self.cache_dir)
            dataset.save_to_disk(self.dataset_path)

        return dataset

    def save_to_disk(self, dataset):
        """Save a Hugging Face dataset to the cache directory."""
        try:
            dataset_path = os.path.join(self.cache_dir, self.dataset_name)
            dataset.save_to_disk(dataset_path)
        except Exception as e:
	    logger.error(f"An error occurred while saving dataset: {str(e)}")

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
            self._tag_dateset(repo_id)
        except Exception as e:
            logger.error(f"An error occurred while pushing to Hugging Face Hub: {str(e)}")
