"""Multimodal embedding backends for VQASynth.

The pipeline consumes image embeddings in two places:

* ``EmbeddingGenerator`` stores per-image embeddings on the dataset.
* ``TagFilter`` matches those embeddings against textual tags to keep/drop rows.

Originally both were hardwired to OpenAI CLIP. Issue #33 asks for "other
multimodal embedding options" (LLM2CLIP, MagicLens, ...). This module now
exposes a small backend registry so the embedding model can be chosen at
construction time without touching the rest of the pipeline::

    EmbeddingGenerator()                                  # OpenAI CLIP (default)
    EmbeddingGenerator(backend="transformers",
                       model_name="microsoft/LLM2CLIP-OpenAI-B-16",
                       trust_remote_code=True)            # LLM2CLIP text encoder
    TagFilter(backend="transformers",
              model_name="google/siglip-base-patch16-224")  # SigLIP space

Each backend implements ``encode_image`` / ``encode_text`` returning
*unnormalized* feature tensors, so the TagFilter ranking logic stays
backend-agnostic.

Note on MagicLens: it is a representation-adjustment retrieval model that
encodes a (query image, free-form text instruction) pair rather than a shared
image/text embedding space. That does not map onto the TagFilter contract
(image vs. tag similarity), so it is intentionally out of scope here; the
registry below is the extension point for adding it behind a dedicated stage.
"""
import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------- #
# Backend registry
# --------------------------------------------------------------------------- #

_EMBEDDING_BACKENDS = {}


def register_embedding_backend(name):
    """Class decorator: register an ``EmbeddingBackend`` subclass under ``name``."""

    def _decorator(cls):
        _EMBEDDING_BACKENDS[name] = cls
        return cls

    return _decorator


def list_embedding_backends():
    """Return the sorted names of registered embedding backends."""
    return sorted(_EMBEDDING_BACKENDS)


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
    """Interface for a multimodal embedding model.

    Subclasses populate ``self.device`` (and, if useful, ``self.model`` /
    ``self.preprocess`` for backward compatibility) and implement
    ``encode_image`` / ``encode_text``. Both must return *unnormalized*
    ``torch.Tensor`` features so callers control normalization.
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None

    def encode_image(self, image):
        """Return an unnormalized image feature tensor of shape (1, D)."""
        raise NotImplementedError

    def encode_text(self, tags):
        """Return unnormalized text feature tensors of shape (len(tags), D)."""
        raise NotImplementedError


@register_embedding_backend("clip")
class CLIPBackend(EmbeddingBackend):
    """OpenAI CLIP backend (the original, default backend).

    ``clip`` is imported lazily so this module imports cleanly even when the
    OpenAI CLIP package is not installed — selecting a different backend then
    needs no CLIP install at all.
    """

    def __init__(self, model_name="ViT-B/32", device=None, **_):
        super().__init__(model_name=model_name, device=device)
        import clip  # lazy: only required when this backend is actually used

        self.model, self.preprocess = clip.load(model_name, self.device)
        self._tokenize = clip.tokenize

    def encode_image(self, image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(image_input)

    def encode_text(self, tags):
        text_inputs = torch.cat(
            [self._tokenize(f"a photo of a {tag}") for tag in tags]
        ).to(self.device)
        return self.model.encode_text(text_inputs)


@register_embedding_backend("transformers")
class TransformersBackend(EmbeddingBackend):
    """HuggingFace Transformers embedding backend.

    Loads any CLIP/SigLIP-family or LLM2CLIP checkpoint via ``transformers``
    (already a VQASynth dependency, so no new install for the common case).
    This is the primary "other multimodal embedding option":

    * **LLM2CLIP** — pass an LLM2CLIP checkpoint, e.g.
      ``microsoft/LLM2CLIP-OpenAI-B-16`` (requires ``trust_remote_code=True``).
    * **SigLIP** — e.g. ``google/siglip-base-patch16-224`` for a non-CLIP
      multimodal embedding space.

    Any model exposing ``get_image_features`` / ``get_text_features`` works,
    so new HF multimodal encoders slot in without code changes.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None,
                 trust_remote_code=False, **_):
        super().__init__(model_name=model_name, device=device)
        from transformers import AutoModel, AutoProcessor

        self.model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        return self.model.get_image_features(pixel_values=pixel_values)

    def encode_text(self, tags):
        prompts = [f"a photo of a {tag}" for tag in tags]
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs)


# --------------------------------------------------------------------------- #
# Public model classes (unchanged surface, pluggable backend)
# --------------------------------------------------------------------------- #


class MultiModalEmbeddingModel:
    """Owns an embedding backend selected by ``backend``.

    Defaults preserve the original CLIP behaviour, so ``EmbeddingGenerator()``
    and ``TagFilter()`` keep working exactly as before. Any keyword arguments
    beyond ``model_name`` / ``device`` / ``backend`` are forwarded to the
    backend constructor.
    """

    def __init__(self, model_name="ViT-B/32", device=None, backend="clip", **backend_kwargs):
        if backend not in _EMBEDDING_BACKENDS:
            raise ValueError(
                f"Unknown embedding backend {backend!r}. "
                f"Available: {list_embedding_backends()}"
            )
        self.backend_name = backend
        self._backend = _EMBEDDING_BACKENDS[backend](
            model_name=model_name, device=device, **backend_kwargs
        )
        self.device = self._backend.device
        # Backward compatibility: expose model/preprocess when the backend sets them.
        self.model = getattr(self._backend, "model", None)
        self.preprocess = getattr(self._backend, "preprocess", None)


class EmbeddingGenerator(MultiModalEmbeddingModel):
    def run(self, image: Image.Image):
        """
        Generate normalized image embeddings via the selected backend.

        Args:
            image (PIL.Image.Image): The input image for which embeddings are generated.

        Returns:
            np.ndarray: Normalized embeddings for the image (float32).
        """
        with torch.no_grad():
            image_features = self._backend.encode_image(image)
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
        with torch.no_grad():
            text_features = self._backend.encode_text(tags)
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
