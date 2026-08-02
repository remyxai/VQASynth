"""Metric3D v2 backend — zero-shot metric depth + surface normals.

Metric3D v2 (Yin et al., arXiv:2404.15506) is a monocular geometric foundation
model that emits both *metric* depth (real-world meters, no per-image scale
calibration) and a dense *surface-normal* map from a single RGB image. Two
ideas make it a natural drop-in for VQASynth's spatial-reasoning depth surface
(``DepthProEstimator``/``VggtEstimator`` slot):

1. **Metric depth without calibration.** The model is trained against a fixed
   set of *canonical* camera intrinsics. At inference you resize the input so
   its effective focal length matches the nearest canonical intrinsic — this
   decouples the recovered metric scale from the (usually unknown) real focal
   length, which is precisely the failure mode the team moved off relative-depth
   models to avoid (see ``tools/depth.py``). The canonicalization itself is
   pure geometry; we implement and unit-test it here, separately from the heavy
   network forward.

2. **Surface normals.** Depth and normals are geometrically complementary (the
   paper's central thesis). Normals carry orientation cues depth alone can't —
   "which way does this surface face", "is the object upright" — and ride the
   same ``DepthResult.normal_map`` field so the existing depth-tool surface
   carries them at no extra plumbing cost.

Adapted-port note (Mode 2): the depth+normal *networks* are invoked through the
upstream ``metric3d`` package (lazy import, optional dep — same convention as
the DepthPro / VGGT / FoundationGeo backends), following that repo's
``run_depth.py`` inference flow. The canonicalization trick — the part that
makes the output metric rather than up-to-scale — is implemented directly as
parameter-free geometry below and is unit-tested without a model. Cut from the
paper: Metric3D's separate benchmark/eval harness (evaluation belongs
downstream).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

# Reuse the same dtype alias resolver as the other tools so calling
# conventions match across the whole tool surface.
from experiments.nooa_agent.tools.florence import _resolve_torch_dtype

if TYPE_CHECKING:
    # Imported lazily at runtime inside ``metric_depth`` (see below) to avoid a
    # circular import — ``tools/depth.py`` re-exports ``Metric3DEstimator`` from
    # this module, so importing depth at module top would deadlock. This block
    # only satisfies the return annotation for static type checkers.
    from experiments.nooa_agent.tools.depth import DepthResult


# Canonical intrinsics Metric3D v2 was trained on — the 7-bin set used by
# upstream ``run_depth.py`` (each entry fixes H, W and the focal length the
# model assumes). At inference we pick the entry whose horizontal FoV is
# nearest the input image's and resize to that resolution; the model then
# returns real-meter depth because the canonical intrinsic pins the scale.
# Stored as (W, H, fx) — fy == fx (square pixels) for every canonical bin.
CANONICAL_INTRINSICS: tuple[tuple[int, int, float], ...] = (
    (1064, 616, 920.4824),
    (1312, 760, 920.0),
    (1076, 622, 937.7376),
    (868, 504, 756.4286),
    (984, 570, 690.0),
    (804, 466, 700.0),
    (896, 518, 815.5040),
)


def horizontal_fov_deg(width_px: float, focal_px: float) -> float:
    """Horizontal field of view (degrees) of a pinhole camera.

    ``fovx = 2 * atan(W / (2 * fx))``. Pure geometry — used to pick the nearest
    canonical intrinsic and to map an image's focal length onto Metric3D's
    canonical bins.
    """
    return float(np.degrees(2.0 * np.arctan(width_px / (2.0 * focal_px))))


def select_canonical_intrinsic(
    fov_x_deg: float,
    canonicals: tuple[tuple[int, int, float], ...] = CANONICAL_INTRINSICS,
) -> tuple[int, int, float]:
    """Pick the canonical (W, H, fx) whose horizontal FoV is nearest the input.

    This is the selection step of Metric3D's canonicalization: the chosen bin's
    resolution is what the input image gets resized to before the forward pass,
    which is what locks the output to metric scale.
    """
    return min(
        canonicals,
        key=lambda c: abs(horizontal_fov_deg(c[0], c[2]) - fov_x_deg),
    )


class Metric3DEstimator:
    """Metric depth + surface normals via Metric3D v2 (arXiv:2404.15506).

    Implements the paper's canonicalization-based metric depth (resize the
    input to the nearest canonical intrinsic so the model's scale assumption
    holds) and additionally surfaces the normal head's output on
    ``DepthResult.normal_map`` — the geometrically-complementary signal the
    repo's depth surface didn't carry before.

    Model: the ViT-Large Metric3D v2 checkpoint from the upstream repo
    (https://github.com/YvanYin/Metric3D). License: MIT (upstream code).

    The depth+normal networks are the substituted auxiliary (Mode 2): they are
    invoked through the upstream ``metric3d`` package following its
    ``run_depth.py`` flow rather than reimplemented here. The exact loader symbol
    varies by installed ``metric3d`` version, so ``_load`` should be adapted to
    the version on the target GPU host — only the canonicalization geometry (the
    paper's core mechanism, what makes the output metric) is implemented and
    unit-tested directly here.

    Requires: ``pip install -e .`` inside the cloned Metric3D repo (provides the
    ``metric3d`` package). The network forward only runs on a GPU host; the
    canonicalization geometry is exercised by the unit tests without it.

    Args:
        device: torch device string (e.g. ``"cuda:0"``, ``"cpu"``).
        dtype: torch dtype or alias resolved by ``_resolve_torch_dtype``
            (``"fp16"``/``"bf16"`` halve VRAM at negligible accuracy loss).
        fov_x_deg: optional horizontal FOV in degrees. When known (e.g. from
            EXIF), pass the true value to pin the canonical bin exactly — this
            is the focal-OOD case canonicalization exists to neutralize.
            ``None`` (default) lets Metric3D estimate FOV internally.
    """

    MODEL_ID = "metric3d_vit_large_800k.pth"

    def __init__(
        self,
        device: str = "cuda",
        dtype: Any = None,
        fov_x_deg: float | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.fov_x_deg = fov_x_deg
        self._model = None

    def _load(self):
        # Lazy import — metric3d isn't a mandatory VQASynth dep. The loader
        # below mirrors upstream run_depth.py; adapt the symbol to the
        # installed metric3d version on the target host (see class docstring).
        try:
            from metric3d.api_inference import Metric3D
        except ImportError as e:
            raise ImportError(
                "metric3d is required for the Metric3Dv2 backend — clone "
                "https://github.com/YvanYin/Metric3D and `pip install -e .` "
                "to get the `metric3d` package. Original error: "
                f"{e}"
            )
        import torch

        model = Metric3D.from_pretrained(self.MODEL_ID).to(
            torch.device(self.device)
        ).eval()
        precision = _resolve_torch_dtype(self.dtype)
        if precision is not None:
            model = model.to(precision)
        self._model = model

    def metric_depth(self, image) -> DepthResult:
        """Predict metric depth (meters) + focal length + surface normals.

        Runs Metric3D v2's canonicalization→forward→resize-back pipeline. The
        depth is metric without scale calibration (the canonical intrinsic pins
        it), and the normal head's (H, W, 3) map is returned on
        ``DepthResult.normal_map`` for downstream orientation reasoning.
        """
        # Imported here (not at module top) to break the depth↔metric3d cycle.
        from experiments.nooa_agent.tools.depth import DepthResult, _unproject
        import torch

        if self._model is None:
            self._load()

        # PIL Image (or ndarray) → RGB ndarray for canonicalization + the model.
        if hasattr(image, "convert"):  # PIL.Image
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
        src_h, src_w = arr.shape[:2]

        # ── Canonicalization: resize the input so its focal length matches the
        # nearest canonical intrinsic. This is the step that makes the depth
        # metric — the model assumes the canonical focal, so we warp the image
        # to agree. If the caller didn't supply an FOV, defer to Metric3D's own
        # estimate (model.infer recomputes canonicalization internally).
        if self.fov_x_deg is not None:
            Wc, Hc, _ = select_canonical_intrinsic(self.fov_x_deg)
            resized = _resize_image(arr, Wc, Hc)
        else:
            Wc, Hc = src_w, src_h
            resized = arr

        img_tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_tensor = img_tensor.to(next(self._model.parameters()).device)

        with torch.no_grad():
            output = self._model.infer(
                img_tensor,
                fov_x=self.fov_x_deg,  # None → Metric3D estimates FOV
            )

        # Metric3D returns depth in meters and, for the v2 head, a normal map.
        depth_canonical = output["depth"].cpu().numpy().astype(np.float32)
        normal_canonical = (
            output["normal"].cpu().numpy().astype(np.float32)
            if "normal" in output
            else None
        )
        # DepthPro/VGGT return pixel-space intrinsics; Metric3D's canonical
        # intrinsic is what locked the scale, so report the canonical focal in
        # pixel space at the *output* resolution for a uniform DepthResult.
        focal_px = float(output.get("focallength_px", _canonical_focal_at(Wc, Hc)))
        H, W = depth_canonical.shape[-2:]
        cx, cy = W / 2, H / 2
        K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1]], dtype=np.float32)

        # Resize depth (and normals, if present) back to the input resolution so
        # downstream bbox-sampled tools index against the original image grid.
        depth_m = _resize_depth(depth_canonical, src_w, src_h) if (W, H) != (src_w, src_h) else depth_canonical
        normal_map = (
            _resize_normal(normal_canonical, src_w, src_h)
            if normal_canonical is not None and (normal_canonical.shape[-2], normal_canonical.shape[-1]) != (src_h, src_w)
            else normal_canonical
        )

        return DepthResult(
            depth_m=depth_m,
            focal_px=focal_px,
            intrinsics_3x3=K,
            point_cloud_xyz=_unproject(depth_m, K),
            normal_map=normal_map,
            backend="metric3d",
        )


# ────────────────────────────────────────────────────────────────
# Canonicalization helpers (image + depth resampling)
# ────────────────────────────────────────────────────────────────

def _canonical_focal_at(width: int, height: int) -> float:
    """Focal length (px) of the canonical bin matching an output resolution.

    Falls back to the most common bin's focal when no exact match — only used
    to populate DepthResult.focal_px when the model didn't return one.
    """
    for w, h, fx in CANONICAL_INTRINSICS:
        if (w, h) == (width, height):
            return fx
    return CANONICAL_INTRINSICS[0][2]


def _resize_image(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an (H, W, 3) uint8 image to (height, width, 3) via PIL bilinear."""
    from PIL import Image

    return np.asarray(Image.fromarray(arr).resize((width, height), Image.BILINEAR))


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a metric depth map back to the input resolution.

    Depth is resampled with bilinear interpolation — metric scale is preserved
    (no renormalization), only the spatial grid changes.
    """
    from PIL import Image

    return np.asarray(
        Image.fromarray(depth.astype(np.float32)).resize((width, height), Image.BILINEAR)
    ).astype(np.float32)


def _resize_normal(normal: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an (H, W, 3) normal map, re-normalizing each vector afterwards.

    Bilinear resampling shrinks vectors unevenly; normalize per-pixel so the
    output stays unit-length (downstream dot-product orientation checks assume
    unit normals).
    """
    from PIL import Image

    out = np.stack(
        [
            np.asarray(
                Image.fromarray(normal[..., c].astype(np.float32)).resize(
                    (width, height), Image.BILINEAR
                )
            )
            for c in range(3)
        ],
        axis=-1,
    ).astype(np.float32)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    return (out / norms).astype(np.float32)
