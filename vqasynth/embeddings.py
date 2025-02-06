import clip
import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod


class MultiModalEmbeddingModel:
    def __init__(self, model_name="ViT-B/32", device=None):
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
        is_batched = isinstance(example[images], list) and isinstance(
            example[images][0], (list, Image.Image)
        )

        try:
            if is_batched:
                embeddings = []
                for img_list in example[images]:
                    image = img_list[0] if isinstance(img_list, list) else img_list

                    if not isinstance(image, Image.Image):
                        raise ValueError(f"Expected a PIL image but got {type(image)}")

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    embedding = self.run(image)
                    embeddings.append(embedding)

                example["embedding"] = embeddings

            else:
                image = (
                    example[images][0]
                    if isinstance(example[images], list)
                    else example[images]
                )

                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected a PIL image but got {type(image)}")

                if image.mode != "RGB":
                    image = image.convert("RGB")

                embedding = self.run(image)
                example["embedding"] = embedding

        except Exception as e:
            print(f"Error processing image, skipping: {e}")
            if is_batched:
                example["embedding"] = [None] * len(example[images])
            else:
                example["embedding"] = None

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
        text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {tag}") for tag in tags]
        ).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_embeddings_tensor = torch.from_numpy(np.array(image_embeddings)).to(
            self.device
        )
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
        is_batched = isinstance(example["embedding"], list) and isinstance(
            example["embedding"][0], list
        )

        try:
            if is_batched:
                best_tags = []
                for embedding in example["embedding"]:
                    if embedding is None:
                        best_tag = None
                    else:
                        best_tag = self.get_best_matching_tag(embedding, tags)
                    best_tags.append(best_tag)
                example["tag"] = best_tags

            else:
                embedding = example["embedding"]
                if embedding is None:
                    example["tag"] = None
                else:
                    example["tag"] = self.get_best_matching_tag(embedding, tags)

        except Exception as e:
            print(f"Error processing embedding, skipping: {e}")
            if is_batched:
                example["tag"] = [None] * len(example["embedding"])
            else:
                example["tag"] = None

        return example


class MultiModalEmbeddingModel(ABC):
    """
    Abstract base class for multimodal embedding models.
    Supports image-only, text-only, and joint image-text embeddings.
    """

    @abstractmethod
    def preprocess(self, image):
        """Preprocess the input image for the embedding model."""
        pass

    @abstractmethod
    def encode_image(self, image):
        """
        Generate embeddings for an image.

        Args:
            image: Preprocessed image.

        Returns:
            np.ndarray: Normalized image embedding.
        """
        pass

    @abstractmethod
    def encode_text(self, text):
        """
        Generate embeddings for a text input.

        Args:
            text: Input text string.

        Returns:
            np.ndarray: Normalized text embedding.
        """
        pass

    def encode_multimodal(self, image, text):
        """
        Generate a joint embedding using both image and text inputs.
        Models that do not support multimodal embeddings directly can combine
        image and text embeddings via concatenation or other operations.

        Args:
            image: Preprocessed image.
            text: Input text string.

        Returns:
            np.ndarray: Normalized multimodal embedding.
        """
        image_embedding = self.encode_image(image)
        text_embedding = self.encode_text(text)

        # Example: Concatenate image and text embeddings (can be overridden)
        multimodal_embedding = np.concatenate(
            [image_embedding, text_embedding], axis=-1
        )
        return self.normalize(multimodal_embedding)

    def normalize(self, embeddings):
        """
        Normalize embeddings to unit length.

        Args:
            embeddings: Input embeddings.

        Returns:
            np.ndarray: Normalized embeddings.
        """
        return embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)


class CLIPEmbeddingModel(MultiModalEmbeddingModel):
    def __init__(self, model_name="ViT-B/32", device=None):
        import clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess_fn = clip.load(model_name, self.device)

    def preprocess(self, image):
        return self.preprocess_fn(image).unsqueeze(0).to(self.device)

    def encode_image(self, image):
        image_tensor = self.preprocess(image)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        return self.normalize(image_features.cpu().numpy())

    def encode_text(self, text):
        import clip

        text_tensor = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tensor)
        return self.normalize(text_features.cpu().numpy())


class MagicLensEmbeddingModel(MultiModalEmbeddingModel):
    def __init__(self, model_path, model_size="base"):
        from flax import serialization
        from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
        from model import MagicLens

        self.tokenizer = clip_tokenizer.build_tokenizer()
        self.model = MagicLens(model_size)
        with open(model_path, "rb") as f:
            model_bytes = pickle.load(f)
        self.model_params = serialization.from_bytes(
            self.model.init(jax.random.PRNGKey(0), {}), model_bytes
        )

    def preprocess(self, image):
        image = image.resize((224, 224))
        image = np.array(image).astype(np.float32) / 255.0
        if image.shape[-1] == 4:  # Handle alpha channel
            image = image[..., :3]
        return jnp.array(image).reshape(1, 224, 224, 3)

    def encode_image(self, image):
        raise NotImplementedError(
            "MagicLens requires both image and text inputs. Use encode_multimodal instead."
        )

    def encode_text(self, text):
        raise NotImplementedError(
            "MagicLens requires both image and text inputs. Use encode_multimodal instead."
        )

    def encode_multimodal(self, image, text):
        # Preprocess inputs
        image_tensor = self.preprocess(image)
        tokenized_text = self.tokenizer([text])
        tokenized_text = jnp.array(tokenized_text).reshape(1, 1, 77)

        # Pass through the model
        inputs = {"ids": tokenized_text, "image": image_tensor}
        outputs = self.model.apply(self.model_params, inputs)

        # Extract and normalize the multimodal embedding
        embedding = outputs["multimodal_embed_norm"]
        return self.normalize(np.array(embedding).flatten())


class LLM2CLIPEmbeddingModel(MultiModalEmbeddingModel):
    def __init__(self, model_path, llm_model_name, device=None):
        from transformers import AutoModel, AutoConfig, AutoTokenizer
        from eva_clip import create_model_and_transforms
        from llm2vec import LLM2Vec

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _, self.preprocess_fn = create_model_and_transforms(
            "EVA02-CLIP-L-14-336", force_custom_clip=True
        )
        self.clip_model.load_state_dict(torch.load(model_path))
        self.clip_model = self.clip_model.to(self.device).eval()

        config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
        llm_model = AutoModel.from_pretrained(
            llm_model_name, config=config, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm2vec = LLM2Vec(llm_model, tokenizer, pooling_mode="mean")

    def preprocess(self, image):
        return self.preprocess_fn(image).unsqueeze(0).to(self.device)

    def encode_image(self, image):
        image_tensor = self.preprocess(image)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
        return self.normalize(image_features.cpu().numpy())

    def encode_text(self, text):
        text_features = self.llm2vec.encode([text], convert_to_tensor=True).to(
            self.device
        )
        return self.normalize(text_features.cpu().numpy())

    def encode_multimodal(self, image, text):
        # Compute separate embeddings and combine them
        image_embedding = self.encode_image(image)
        text_embedding = self.encode_text(text)
        return self.normalize(
            np.concatenate([image_embedding, text_embedding], axis=-1)
        )
