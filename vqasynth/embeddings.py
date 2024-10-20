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
            np.ndarray: Normalized CLIP embeddings for the image.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def apply_transform(self, example, images):
        """
        Process one or more rows in the dataset, adding embeddings from images.

        Args:
            example: A single example or a batch of examples from the dataset.
            images: Column name for image column.

        Returns:
            Updated example(s) with image embeddings.
        """
        is_batched = isinstance(example[images], list) and isinstance(example[images][0], (list, Image.Image))

        try:
            if is_batched:
                embeddings = []
                for img_list in example[images]:
                    # Handle the case where img_list could be a list of images
                    image = img_list[0] if isinstance(img_list, list) else img_list

                    # Ensure that image is a valid PIL image
                    if not isinstance(image, Image.Image):
                        raise ValueError(f"Expected a PIL image but got {type(image)}")

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Run embedding generation on the valid image
                    embedding = self.run(image)
                    embeddings.append(embedding)

                example['embedding'] = embeddings

            else:
                image = example[images][0] if isinstance(example[images], list) else example[images]

                # Ensure that image is a valid PIL image
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected a PIL image but got {type(image)}")

                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Run embedding generation on the valid image
                embedding = self.run(image)
                example['embedding'] = embedding

        except Exception as e:
            print(f"Error processing image, skipping: {e}")
            if is_batched:
                example['embedding'] = [None] * len(example[images])
            else:
                example['embedding'] = None

        return example


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

        image_embeddings_tensor = torch.from_numpy(np.array(image_embeddings)).to(self.device)
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

    def apply_transform(self, example, tags=[]):
        """
        Process one or more rows in the dataset, adding best matching tags from embeddings.

        Args:
            example: A single example or a batch of examples from the dataset.
            tags: A list of tags to match the embeddings with.

        Returns:
            Updated example(s) with best matching tag(s).
        """
        # Detect if input is batched by checking if the first element of 'embedding' is a list
        is_batched = isinstance(example['embedding'], list) and isinstance(example['embedding'][0], list)

        try:
            if is_batched:
                best_tags = []
                for embedding in example['embedding']:
                    # Ensure the embedding is valid
                    if embedding is None:
                        best_tag = None
                    else:
                        best_tag = self.get_best_matching_tag(embedding, tags)
                    best_tags.append(best_tag)
                example['tag'] = best_tags

            else:
                embedding = example['embedding']
                # Ensure the embedding is valid
                if embedding is None:
                    example['tag'] = None
                else:
                    example['tag'] = self.get_best_matching_tag(embedding, tags)

        except Exception as e:
            print(f"Error processing embedding, skipping: {e}")
            if is_batched:
                example['tag'] = [None] * len(example['embedding'])
            else:
                example['tag'] = None

        return example

