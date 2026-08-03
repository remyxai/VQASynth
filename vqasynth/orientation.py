"""Object-level orientation estimation with Orient-Anything.

Adds per-object 3D orientation (azimuth / polar / rotation + a confidence
score) to a dataset, following the same wrapper shape as :mod:`vqasynth.depth`
(``__init__`` -> ``run`` -> ``apply_transform``). The orientation head itself is
the DINOv2_MLP model from `Orient-Anything <https://github.com/SpatialVision/Orient-Anything>`_
(ICML 2025). Its decoding logic is reproduced here in :func:`decode_angles` so
the module is self-contained; only the model class + weights are pulled from
that repo.

Orient-Anything is trained on rendered single-object images and only
generalizes to in-the-wild photos when each object is isolated first (the
repo's stated "Best Practice"). This pipeline already produces one SAM2 mask
per object (see :mod:`vqasynth.localize`), so :func:`crop_to_object` crops and
background-isolates each mask before estimating its orientation, giving one
orientation estimate per object rather than one per image.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from vqasynth.utils import pick_dtype


# Orient-Anything head layout (see DINOv2_MLP construction in its README:
# out_dim = 360 + 180 + 180 + 2). decode_angles splits the model output into
# these contiguous heads. (The repo's inference.get_3angle slices rotation as
# [540:900], which is internally inconsistent with this out_dim; we use the
# documented, self-consistent layout instead.)
_AZIMUTH_BINS = 360   # 0..359 degrees
_POLAR_BINS = 180     # argmax - 90  ->  -90..89 degrees
_ROTATION_BINS = 180  # argmax - 180 ->  -180..-1 degrees
_CONFIDENCE_BINS = 2  # in-/out-of-distribution head, softmax -> P(in-dist)


def decode_angles(logits):
    """Decode Orient-Anything head logits into an orientation dict.

    Faithful port of ``get_3angle`` from the Orient-Anything repo: argmax over
    the azimuth / polar / rotation bins (with the repo's ``-90`` / ``-180``
    shifts) and a softmax over the 2-class in/out-of-distribution head.

    Args:
        logits: raw model output of shape ``(B, 720)`` (or wider). Works on
            torch tensors or numpy arrays; the first batch row is used.

    Returns:
        dict with float ``azimuth`` (0..359), ``polar`` (-90..89),
        ``rotation`` (-180..-1), and ``confidence`` (0..1).
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    logits = logits.float()
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    az_start = 0
    polar_start = _AZIMUTH_BINS
    rot_start = _AZIMUTH_BINS + _POLAR_BINS
    conf_start = _AZIMUTH_BINS + _POLAR_BINS + _ROTATION_BINS

    row = logits[0]
    azimuth = int(torch.argmax(row[az_start:polar_start]).item())
    polar = int(torch.argmax(row[polar_start:rot_start]).item()) - 90
    rotation = int(torch.argmax(row[rot_start:conf_start]).item()) - 180
    confidence = float(
        F.softmax(row[conf_start:conf_start + _CONFIDENCE_BINS], dim=-1)[0].item()
    )

    return {
        "azimuth": float(azimuth),
        "polar": float(polar),
        "rotation": float(rotation),
        "confidence": confidence,
    }


def crop_to_object(image, mask, padding=0.1):
    """Isolate a single object from an image using its segmentation mask.

    Crops to the mask's bounding box, pads it to a square (the orientation head
    expects square-ish input), and whites out everything outside the mask so
    only the object remains. Returns the original image unchanged when the mask
    is empty.

    Args:
        image: ``PIL.Image`` (any mode; converted to RGB).
        mask: 2D ``uint8``/``bool`` array with the same HxW as ``image``.
            Non-zero pixels mark the object.
        padding: fraction of the object's longer side to pad around it.

    Returns:
        ``PIL.Image`` (RGB) square crop containing only the masked object.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask_arr.shape}")
    if mask_arr.shape[:2] != image.size[::-1]:
        raise ValueError(
            f"mask {mask_arr.shape[:2]} and image {image.size[::-1]} "
            "spatial dimensions must match"
        )

    nonzero = np.argwhere(mask_arr > 0)
    if len(nonzero) == 0:
        return image

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)
    side = int(max(y_max - y_min, x_max - x_min) * (1 + padding))
    side = max(side, 1)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    right = left + side
    bottom = top + side

    # Grow the canvas so the square crop fits even when the object hugs an edge,
    # paste the image onto white, then crop.
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image.width)
    pad_bottom = max(0, bottom - image.height)

    canvas = Image.new(
        "RGB",
        (image.width + pad_left + pad_right, image.height + pad_top + pad_bottom),
        (255, 255, 255),
    )
    canvas.paste(image, (pad_left, pad_top))
    crop_box = (
        left + pad_left,
        top + pad_top,
        right + pad_left,
        bottom + pad_top,
    )
    crop = canvas.crop(crop_box)

    # Shift the mask into canvas space and white out everything outside it.
    mask_full = np.zeros((canvas.height, canvas.width), dtype=bool)
    mask_full[pad_top:pad_top + image.height, pad_left:pad_left + image.width] = (
        mask_arr > 0
    )
    crop_mask = mask_full[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    # np.array() (not asarray) — we need a writable copy to whiten the bg.
    crop_arr = np.array(crop)
    crop_arr[~crop_mask] = 255
    return Image.fromarray(crop_arr)


class OrientationEstimator:
    """Estimate per-object 3D orientation with Orient-Anything.

    Mirrors :class:`vqasynth.depth.DepthEstimator`: ``run(image)`` for a single
    inference and ``apply_transform(example, images, masks)`` for the
    ``datasets.map`` integration.
    """

    _DEFAULT_BACKBONE = "facebook/dinov2-large"
    _DEFAULT_WEIGHTS_REPO = "Viglong/Orient-Anything"
    _DEFAULT_WEIGHTS_FILE = "croplargeEX2/dino_weight.pt"

    def __init__(
        self,
        model=None,
        preprocess=None,
        device=None,
        backbone=_DEFAULT_BACKBONE,
        weights_repo=_DEFAULT_WEIGHTS_REPO,
        weights_file=_DEFAULT_WEIGHTS_FILE,
    ):
        """Load the Orient-Anything head + DINOv2 image processor.

        Args:
            model: an already-constructed Orient-Anything model. When ``None``
                (default) the model is loaded lazily from the Orient-Anything
                repo, which must be importable (e.g. on PYTHONPATH). Exposed as
                an injection point for tests and custom backbones.
            preprocess: a ``transformers`` image processor for the DINOv2
                backbone. Injected alongside ``model`` for tests.
            device: torch device. Auto-detected when ``None``.
            backbone, weights_repo, weights_file: pointers used only when the
                model is loaded here.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = pick_dtype()

        if model is not None:
            self.model = model.to(self.device) if hasattr(model, "to") else model
            self.preprocess = preprocess
        else:
            self.model, self.preprocess = self._load_model(
                backbone, weights_repo, weights_file
            )

        if hasattr(self.model, "eval"):
            self.model.eval()

    def _load_model(self, backbone, weights_repo, weights_file):
        """Lazy-load the Orient-Anything model + processor.

        ``DINOv2_MLP`` lives in the Orient-Anything repo (not on PyPI), so it is
        imported here to avoid forcing the dependency on every importer of
        ``vqasynth``. Clone https://github.com/SpatialVision/Orient-Anything and
        put it on your PYTHONPATH to use the real weights.
        """
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor

        try:
            from vision_tower import DINOv2_MLP
        except ImportError as exc:
            raise ImportError(
                "Orient-Anything model code not found. Clone "
                "https://github.com/SpatialVision/Orient-Anything and add it to "
                "your PYTHONPATH, or inject a `model=` into OrientationEstimator."
            ) from exc

        model = DINOv2_MLP(
            dino_mode="large",
            in_dim=1024,
            out_dim=_AZIMUTH_BINS + _POLAR_BINS + _ROTATION_BINS + _CONFIDENCE_BINS,
            evaluate=True,
            mask_dino=False,
            frozen_back=False,
        )
        ckpt_path = hf_hub_download(
            repo_id=weights_repo, filename=weights_file, repo_type="model"
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model = model.to(device=self.device)
        preprocess = AutoImageProcessor.from_pretrained(backbone)
        return model, preprocess

    def run(self, image):
        """Estimate orientation for a single (ideally single-object) image.

        Args:
            image: ``PIL.Image``.

        Returns:
            dict with ``azimuth``, ``polar``, ``rotation``, ``confidence``.
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected a PIL image but got {type(image)}")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Orient-Anything's DINOv2_MLP.forward takes the preprocessor dict and
        # unpacks it into the backbone (self.dinov2(**img_inputs)); pass the
        # whole dict, not just pixel_values — see get_3angle in the repo.
        inputs = self.preprocess(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        inputs["pixel_values"] = pixel_values.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return decode_angles(logits)

    def run_objects(self, image, masks):
        """Estimate orientation for every object mask in an image.

        Args:
            image: ``PIL.Image``.
            masks: iterable of 2D ``uint8``/``bool`` masks (one per object).

        Returns:
            list of orientation dicts aligned with ``masks``; a mask that fails
            to crop/infer yields ``None`` so indices stay aligned.
        """
        orientations = []
        for mask in masks or []:
            try:
                crop = crop_to_object(image, mask)
                orientations.append(self.run(crop))
            except Exception as exc:  # one bad mask shouldn't sink the row
                print(f"Error estimating orientation for an object, skipping: {exc}")
                orientations.append(None)
        return orientations

    def apply_transform(self, example, images, masks="masks"):
        """Add an ``orientation`` column to one or more dataset rows.

        Each image's per-object masks are cropped to single-object sub-images
        (Orient-Anything's Best Practice) and each gets its own estimate.
        Handles batched and unbatched examples and degrades to ``None`` on
        failure, matching :meth:`DepthEstimator.apply_transform`.

        Args:
            example: a single example or a batch from ``datasets.map``.
            images: column name holding the PIL images.
            masks: column name holding the per-object mask lists.

        Returns:
            The example with an added ``orientation`` key — a list of per-object
            orientation dicts (or ``None`` on failure).
        """
        is_batched = isinstance(example[images], list) and isinstance(
            example[images][0], (list, Image.Image)
        )

        try:
            if is_batched:
                all_orientations = []
                for i, img_item in enumerate(example[images]):
                    image = img_item[0] if isinstance(img_item, list) else img_item
                    if not isinstance(image, Image.Image):
                        raise ValueError(
                            f"Expected a PIL image but got {type(image)}"
                        )
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    obj_masks = example[masks][i] if i < len(example[masks]) else []
                    all_orientations.append(self.run_objects(image, obj_masks))
                example["orientation"] = all_orientations
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

                obj_masks = example.get(masks, [])
                example["orientation"] = self.run_objects(image, obj_masks)
        except Exception as exc:
            print(f"Error processing image, skipping: {exc}")
            if is_batched:
                example["orientation"] = [None] * len(example[images])
            else:
                example["orientation"] = None

        return example
