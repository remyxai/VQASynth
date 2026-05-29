"""
Synthetic visual-degradation operators for degradation-aware evaluation.

Adapted from:
- SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation
  (arXiv:2605.22536). SpaceDG shows that spatial reasoning in MLLMs degrades
  sharply under real-world visual corruptions (motion blur, low light, adverse
  weather, lens distortion, compression artifacts, ...). Its full pipeline
  embeds a degradation-formation process into 3D Gaussian Splatting rendering;
  that neural renderer is NOT reproduced here. What IS reproduced is the
  image-space degradation formation model used to perturb already-rendered
  images, which is what makes "how robust is this benchmark accuracy under
  imperfect inputs?" answerable on the repo's existing evaluation stage.

These operators apply a chosen corruption at a 1-5 severity scale (mirroring
the ImageNet-C convention) to a PIL image. They are consumed by
``vqasynth.benchmarks.BenchmarkRunner`` to produce degraded benchmark items,
so the existing multi-benchmark eval can be re-run on degraded inputs and the
clean-vs-degraded accuracy gap measured.

Dependencies are limited to PIL + numpy, which the rest of the package already
uses; no neural checkpoints or 3D infrastructure are required.
"""

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# Severity runs 1 (mild) .. 5 (severe), matching the ImageNet-C / SpaceDG scale.
MIN_SEVERITY = 1
MAX_SEVERITY = 5


def _check_severity(severity):
    if not isinstance(severity, int) or not (MIN_SEVERITY <= severity <= MAX_SEVERITY):
        raise ValueError(
            f"severity must be an int in [{MIN_SEVERITY}, {MAX_SEVERITY}], got {severity!r}"
        )


def _as_rgb(image):
    """Coerce to an RGB PIL image, raising if it is not image-like."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL.Image, got {type(image).__name__}")
    return image.convert("RGB") if image.mode != "RGB" else image


# ---------------------------------------------------------------------------
# Degradation operators
# ---------------------------------------------------------------------------

def motion_blur(image, severity=3):
    """Directional (horizontal) motion blur via averaging shifted copies."""
    img = _as_rgb(image)
    length = [3, 5, 9, 15, 21][severity - 1]
    arr = np.asarray(img, dtype=np.float32)
    acc = np.zeros_like(arr)
    for shift in range(length):
        acc += np.roll(arr, shift - length // 2, axis=1)
    acc /= length
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))


def defocus_blur(image, severity=3):
    """Lens defocus modeled as a Gaussian blur with growing radius."""
    img = _as_rgb(image)
    radius = [1.0, 2.0, 3.0, 4.5, 6.0][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def low_light(image, severity=3):
    """Underexposure: gamma darkening plus a brightness pulldown."""
    img = _as_rgb(image)
    gamma = [1.3, 1.6, 2.0, 2.6, 3.3][severity - 1]
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.power(arr, gamma)
    out = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return ImageEnhance.Brightness(out).enhance(0.85)


def gaussian_noise(image, severity=3):
    """Additive sensor noise (zero-mean Gaussian in pixel space)."""
    img = _as_rgb(image)
    sigma = [8, 16, 26, 38, 52][severity - 1]
    arr = np.asarray(img, dtype=np.float32)
    # Deterministic, image-derived noise so degraded output is reproducible
    # without depending on a global RNG (Math.random-style nondeterminism).
    rng = np.random.default_rng(seed=int(arr.sum()) % (2**32))
    noise = rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def jpeg_compression(image, severity=3):
    """Block/compression artifacts via a low-quality JPEG round-trip."""
    img = _as_rgb(image)
    quality = [40, 28, 18, 11, 6][severity - 1]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fog(image, severity=3):
    """Atmospheric haze: blend the image toward a bright gray veil."""
    img = _as_rgb(image)
    strength = [0.15, 0.3, 0.45, 0.6, 0.75][severity - 1]
    arr = np.asarray(img, dtype=np.float32)
    veil = np.full_like(arr, 235.0)
    out = (1.0 - strength) * arr + strength * veil
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def contrast_loss(image, severity=3):
    """Overcast / washed-out look from reduced global contrast."""
    img = _as_rgb(image)
    factor = [0.75, 0.6, 0.45, 0.3, 0.2][severity - 1]
    return ImageEnhance.Contrast(img).enhance(factor)


def pixelate(image, severity=3):
    """Resolution loss via downscale-then-upscale (nearest neighbor)."""
    img = _as_rgb(image)
    w, h = img.size
    scale = [0.6, 0.45, 0.3, 0.2, 0.12][severity - 1]
    dw = max(1, int(w * scale))
    dh = max(1, int(h * scale))
    small = img.resize((dw, dh), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


def color_shift(image, severity=3):
    """Lens/white-balance color cast: per-channel gain imbalance."""
    img = _as_rgb(image)
    amount = [0.06, 0.12, 0.2, 0.3, 0.42][severity - 1]
    arr = np.asarray(img, dtype=np.float32)
    gains = np.array([1.0 + amount, 1.0, 1.0 - amount], dtype=np.float32)
    out = arr * gains[None, None, :]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# Registry of SpaceDG-style degradation families reproducible in image space.
DEGRADATIONS = {
    "motion_blur": motion_blur,
    "defocus_blur": defocus_blur,
    "low_light": low_light,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
    "fog": fog,
    "contrast_loss": contrast_loss,
    "pixelate": pixelate,
    "color_shift": color_shift,
}

DEGRADATION_TYPES = list(DEGRADATIONS.keys())


def apply_degradation(image, name, severity=3):
    """
    Apply a named degradation at the given severity to a single PIL image.

    Args:
        image: PIL.Image to degrade.
        name: One of ``DEGRADATION_TYPES``.
        severity: Integer in [1, 5].

    Returns a new degraded PIL.Image (RGB).
    """
    _check_severity(severity)
    if name not in DEGRADATIONS:
        valid = ", ".join(DEGRADATION_TYPES)
        raise ValueError(f"Unknown degradation '{name}'. Valid: {valid}")
    return DEGRADATIONS[name](image, severity)


def _coerce_to_pil(img):
    """Best-effort coercion of a benchmark item's image to PIL.Image, or None."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, bytes):
        try:
            return Image.open(io.BytesIO(img)).convert("RGB")
        except Exception:
            return None
    if isinstance(img, str):
        try:
            return Image.open(img).convert("RGB")
        except Exception:
            return None
    if isinstance(img, dict) and img.get("bytes") is not None:
        try:
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        except Exception:
            return None
    return None


def degrade_images(images, name, severity=3):
    """
    Degrade a list of benchmark images.

    Accepts the heterogeneous image entries the benchmark loaders emit (PIL
    images, file paths, raw bytes, or HF ``{"bytes": ...}`` dicts). Each entry
    that can be coerced to a PIL image is degraded and returned as a PIL image;
    entries that cannot be coerced are passed through unchanged so the caller's
    downstream handling is preserved.

    Returns a new list the same length as ``images``.
    """
    out = []
    for img in images:
        pil = _coerce_to_pil(img)
        if pil is None:
            out.append(img)
        else:
            out.append(apply_degradation(pil, name, severity))
    return out
