import clip
import torch
import numpy as np
from PIL import Image

class MultiModalEmbeddingModel:
    def __init__(self, model_name='ViT-B/32', device=None):
        """Initialize the CLIP model and its configuration."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, self.device)

class EmbeddingGenerator(MultiModalEmbeddingModel):
    def run(self, image: Image.Image):
        """
        Generate CLIP embeddings for an image.

        Args:
            image (PIL.Image.Image): The input image for which embeddings are generated.

        Returns:
            torch.Tensor: Normalized CLIP embeddings for the image.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

class TagFilter(MultiModalEmbeddingModel):
    def get_best_matching_tag(self, image_embeddings: np.ndarray, tags: list):
        """
        Get the tag with the highest confidence match for the given image embeddings.

        Args:
            image_embeddings (np.ndarray): Precomputed embeddings for the image as a NumPy array.
            tags (list): List of tags to compare with the image embeddings.

        Returns:
            str: The tag with the highest confidence score.
        """
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {tag}") for tag in tags]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_embeddings_tensor = torch.from_numpy(image_embeddings).to(self.device)
        similarity = (100.0 * image_embeddings_tensor @ text_features.T).softmax(dim=-1)

        best_index = similarity.argmax().item()
        best_tag = tags[best_index]

        return best_tag

    def filter_by_tag(self, best_tag: str, include_tags: list, exclude_tags: list):
        """
        Filter the image based on the best-matching tag by comparing against the include/exclude lists.

        Args:
            best_tag (str): The tag with the highest confidence match.
            include_tags (list): Tags to include if present (optional).
            exclude_tags (list): Tags to exclude if present (optional).

        Returns:
            bool: True if the image passes filtering, False otherwise.
        """
        if exclude_tags and best_tag in exclude_tags:
            return False

        if include_tags and best_tag not in include_tags:
            return False

        return True
