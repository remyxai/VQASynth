"""Pluggable multimodal embedding backends.

VQASynth embeds images for content-based filtering (``TagFilter``) and dataset
enrichment (``EmbeddingGenerator``). Both used to be hardwired to the OpenAI
``clip`` package. This module exposes a small backend interface so other
CLIP-family models can be dropped in without touching the dataset-transform
logic (https://github.com/remyxai/VQASynth/issues/33).

Built-in backends:

* ``clip`` (default) — OpenAI CLIP, unchanged behaviour. The ``clip`` import is
  deferred to construction time so this module imports cleanly even when the
  CLIP package is not installed.
* ``transformers`` — any HuggingFace model exposing ``get_image_features`` /
  ``get_text_features``. Covers SigLIP (``google/siglip-base-patch16-224``) and
  LLM2CLIP converted checkpoints (``microsoft/LLM2CLIP-OpenAI-B-16``).
  ``transformers`` is already a VQASynth dependency, so the common case needs no
  extra install. (LLM2CLIP checkpoints need ``transformers>=4.52``.)

New backends can be added via :func:`register_embedding_backend`. MagicLens
(https://github.com/google-deepmind/magiclens) is intentionally not provided
here: it encodes a composed (image, text-instruction) query rather than a shared
image/text embedding space, so it does not satisfy the ``TagFilter`` similarity
contract. The registry is the extension point if a future use case needs it.
"""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image

def _to_same_dtype_tensor(x, ref_tensor, device):
    """
    Convert numpy/torch input `x` to a torch tensor on `device` with the same dtype as `ref_tensor`.
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.as_tensor(x)
    return t.to(device=device, dtype=ref_tensor.dtype)


class EmbeddingBackend:
    """Common surface for a multimodal image + text embedding backend.

    ``EmbeddingGenerator`` and ``TagFilter`` talk to backends only through these
    four primitives, so a new backend can be added without touching the
    dataset-transform logic.
    """

    name = "base"

    def __init__(self, device=None):
        self.device = device

    def preprocess(self, image):
        raise NotImplementedError

    def encode_image(self, image_input):
        raise NotImplementedError

    def encode_text(self, text_input):
        raise NotImplementedError

    def tokenize(self, texts):
        raise NotImplementedError


class CLIPBackend(EmbeddingBackend):
    """OpenAI CLIP backend (https://github.com/openai/CLIP). The default.

    Preserves the original load/encode behaviour; only the ``clip`` import is
    deferred so this module is importable without the package installed.
    """

    name = "clip"

    def __init__(self, model_name="ViT-B/32", device=None):
        import clip  # lazy: not required merely to import vqasynth.embeddings

        super().__init__(device=device)
        self.model, self._preprocess = clip.load(model_name, self.device)
        self._clip = clip

    def preprocess(self, image):
        return self._preprocess(image)

    def encode_image(self, image_input):
        return self.model.encode_image(image_input.to(self.device))

    def encode_text(self, text_input):
        return self.model.encode_text(text_input.to(self.device))

    def tokenize(self, texts):
        return self._clip.tokenize(texts)


class TransformersBackend(EmbeddingBackend):
    """HuggingFace Transformers backend for CLIP-family models.

    Loads any model exposing ``get_image_features`` / ``get_text_features``:
    CLIP, SigLIP, or LLM2CLIP converted checkpoints. ``transformers`` is already
    a VQASynth dependency, so the common case needs no extra install.
    """

    name = "transformers"

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        from transformers import AutoModel, AutoProcessor

        super().__init__(device=device)
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device).eval()
        self._processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

    def preprocess(self, image):
        # (C, H, W) for a single image; EmbeddingGenerator.run unsqueezes batch.
        return self._processor(images=image, return_tensors="pt")["pixel_values"][0]

    def encode_image(self, image_input):
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=image_input.to(self.device))

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )

    def encode_text(self, text_input):
        # text_input is the BatchEncoding returned by tokenize().
        moved = {k: v.to(self.device) for k, v in text_input.items()}
        with torch.no_grad():
            return self.model.get_text_features(**moved)


_EMBEDDING_BACKENDS: dict[str, type[EmbeddingBackend]] = {}


def register_embedding_backend(name, cls):
    """Register an :class:`EmbeddingBackend` subclass under ``name``."""
    if not (isinstance(cls, type) and issubclass(cls, EmbeddingBackend)):
        raise TypeError(f"backend class must subclass EmbeddingBackend; got {cls!r}")
    _EMBEDDING_BACKENDS[name] = cls


def build_embedding_backend(backend, model_name="ViT-B/32", device=None):
    """Resolve ``backend`` (name / class / instance) into an EmbeddingBackend."""
    if isinstance(backend, EmbeddingBackend):
        return backend
    if isinstance(backend, type) and issubclass(backend, EmbeddingBackend):
        return backend(model_name=model_name, device=device)
    if not isinstance(backend, str):
        raise TypeError(
            f"backend must be a name, class, or instance; got {backend!r}"
        )
    try:
        cls = _EMBEDDING_BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"Unknown embedding backend {backend!r}. "
            f"Registered: {sorted(_EMBEDDING_BACKENDS)}."
        ) from None
    return cls(model_name=model_name, device=device)


register_embedding_backend("clip", CLIPBackend)
register_embedding_backend("transformers", TransformersBackend)


class MultiModalEmbeddingModel:
    def __init__(self, backend="clip", model_name="ViT-B/32", device=None):
        """Initialize a multimodal embedding backend and its configuration.

        Args:
            backend: backend name (``"clip"`` / ``"transformers"``), an
                ``EmbeddingBackend`` subclass, or an instance. Defaults to CLIP.
            model_name: model/checkpoint name for the selected backend.
            device: torch device; defaults to CUDA when available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._backend = build_embedding_backend(
            backend, model_name=model_name, device=self.device
        )
        self.backend_name = (
            backend
            if isinstance(backend, str)
            else getattr(backend, "name", backend.__class__.__name__)
        )

        # Uniform interface used by EmbeddingGenerator / TagFilter.
        self.preprocess = self._backend.preprocess
        self.encode_image = self._backend.encode_image
        self.encode_text = self._backend.encode_text
        self.tokenize = self._backend.tokenize
        # Backward-compat: expose the underlying model object.
        self.model = getattr(self._backend, "model", self._backend)

class EmbeddingGenerator(MultiModalEmbeddingModel):
    def run(self, image: Image.Image):
        """
        Generate embeddings for an image using the configured backend.

        Args:
            image (PIL.Image.Image): The input image for which embeddings are generated.

        Returns:
            np.ndarray: Normalized embeddings for the image.
        """
        image_input = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Force float32 so numpy doesn't upcast to float64 later
        return image_features.cpu().to(torch.float32).numpy().astype(np.float32)

    def apply_transform(self, example, images):
        """
        Process one or more rows in the dataset, adding embeddings from images.

        Args:
            example: A single example or a batch of examples from the dataset.
            images: Column name for image column.

        Returns:
            Updated example(s) with image embeddings.
        """
        is_batched = isinstance(example[images], list)

        try:
            if is_batched:
                embeddings = []
                for img_item in example[images]:
                    image = img_item[0] if isinstance(img_item, list) else img_item

                    if not isinstance(image, Image.Image):
                        raise ValueError(f"Expected a PIL image but got {type(image)}")

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    embedding = self.run(image)
                    embeddings.append(embedding)

                example['embedding'] = embeddings

            else:
                image = example[images][0] if isinstance(example[images], list) else example[images]

                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected a PIL image but got {type(image)}")

                if image.mode != "RGB":
                    image = image.convert("RGB")

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
        text_inputs = self.tokenize([f"a photo of a {tag}" for tag in tags])
        with torch.no_grad():
            text_features = self.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # image_embeddings may be list/array of shape (1, D); squeeze and match dtype
        img_emb_np = np.asarray(image_embeddings)
        img_emb_np = np.squeeze(img_emb_np, axis=0) if img_emb_np.ndim == 2 and img_emb_np.shape[0] == 1 else img_emb_np
        image_embeddings_tensor = _to_same_dtype_tensor(img_emb_np, text_features, self.device)

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
        is_batched = isinstance(example['embedding'], list)

        try:
            if is_batched:
                best_tags = []
                for emb in example['embedding']:
                    if emb is None:
                        best_tags.append(None)
                        continue
                    best_tags.append(self.get_best_matching_tag(emb, tags))
                example['tag'] = best_tags

            else:
                emb = example['embedding']
                example['tag'] = None if emb is None else self.get_best_matching_tag(emb, tags)

        except Exception as e:
            print(f"Error processing embedding, skipping: {e}")
            if is_batched:
                example['tag'] = [None] * len(example['embedding'])
            else:
                example['tag'] = None

        return example
