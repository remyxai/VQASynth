"""Metric depth + intrinsics + 3D point cloud tools, resource-tier aware.

All backends produce the same interface: ``DepthResult(depth_m, focal_px,
intrinsics_3x3, point_cloud_xyz, backend)``. Downstream tools (distance-in-meters,
height, "on top of") consume the interface uniformly regardless of which
model produced it.

- CPU tier: **Apple DepthPro** (~330M) — metric depth + predicted focal length.
  ~1-3 s per image on modern CPUs. Continuity with VQASynth's pre-VGGT default
  depth model.
- GPU tier: **VGGT-1B** (via ``vqasynth.scene_fusion.SpatialSceneConstructor``).
  Matches the current VQASynth production path; also produces multi-view fusion
  if the annotator ever passes more than one image per call.
- GPU tier (alternative): **FoundationGeo v1.1** (~314M) — learned pixel-wise
  scale + ray-direction correction fields. ECCV 2026 (arXiv:2607.11588). Metric
  depth trained with wide focal-length coverage; paper's headline result is
  metric robustness across camera-intrinsic OOD.

Notes on the CPU-tier choice: Depth Anything V2 produces RELATIVE depth and would
require a scale-calibration step we saw hallucinate 5cm→pattern in the
2026-07-19 estimate-then-verify probe. DepthPro's metric output eliminates that
failure mode entirely — the tool returns real meters, no scale invention needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Reuse the same dtype alias resolver as the Florence tools so calling
# conventions match across the whole tool surface.
from experiments.nooa_agent.tools.florence import _resolve_torch_dtype


@dataclass
class DepthResult:
    """Uniform depth-tool output regardless of backend."""
    depth_m: np.ndarray                  # (H, W) metric depth in meters
    focal_px: float                      # focal length in pixels
    intrinsics_3x3: np.ndarray           # (3, 3) camera intrinsic matrix
    point_cloud_xyz: np.ndarray | None   # (H, W, 3) or None if not computed
    backend: str                         # e.g. "depthpro" or "vggt"

    def __repr__(self) -> str:
        # Default dataclass repr dumps the full point cloud + depth array as
        # text — for a 768×768 image that's ~7 MB per line in any log or
        # NOOA trace event. Return a compact summary instead. Full arrays
        # remain accessible via the named fields.
        H, W = self.depth_m.shape
        return (
            f"DepthResult(backend={self.backend!r}, focal_px={self.focal_px:.1f}, "
            f"shape=({H}, {W}), has_pointcloud={self.point_cloud_xyz is not None})"
        )


# ────────────────────────────────────────────────────────────────
# CPU tier — DepthPro
# ────────────────────────────────────────────────────────────────

class DepthProEstimator:
    """Metric depth via Apple DepthPro (works on CPU or GPU).

    Requires: ``pip install depth_pro`` (or install from
    ``https://github.com/apple/ml-depth-pro``).

    ``device`` chooses the accelerator (e.g. ``"cuda:1"``). ``dtype`` sets
    precision — fp16 on GPU roughly halves VRAM for the ~330M-param model
    at negligible accuracy loss for metric depth. On CPU, dtype only affects
    the load footprint; inference is fp32-equivalent regardless.
    """
    MODEL_ID = "apple/DepthPro"

    def __init__(self, device: str = "cpu", dtype: Any = None):
        self.device = device
        self.dtype = dtype
        self._model = None
        self._transform = None

    def _load(self):
        # Lazy import — depth_pro isn't a mandatory VQASynth dep
        import depth_pro
        import torch
        precision = _resolve_torch_dtype(self.dtype) or torch.float32
        self._model, self._transform = depth_pro.create_model_and_transforms(
            device=torch.device(self.device),
            precision=precision,
        )
        self._model.eval()

    def metric_depth(self, image) -> DepthResult:
        """Predict metric depth (meters) and focal length for an RGB image.

        No calibration/scale-guessing needed — DepthPro predicts absolute
        metric depth and focal length natively.
        """
        import torch

        if self._model is None:
            self._load()

        # DepthPro expects a specific transform
        img_tensor = self._transform(image)
        with torch.no_grad():
            prediction = self._model.infer(img_tensor)

        depth_m = prediction["depth"].cpu().numpy().astype(np.float32)  # (H, W)
        focal_px = float(prediction["focallength_px"].cpu().item())

        H, W = depth_m.shape
        cx, cy = W / 2, H / 2
        K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1]], dtype=np.float32)
        return DepthResult(
            depth_m=depth_m,
            focal_px=focal_px,
            intrinsics_3x3=K,
            point_cloud_xyz=_unproject(depth_m, K),
            backend="depthpro",
        )


# ────────────────────────────────────────────────────────────────
# GPU tier — VGGT via VQASynth's SpatialSceneConstructor
# ────────────────────────────────────────────────────────────────

class VggtEstimator:
    """GPU-tier metric depth + point cloud via VGGT-1B.

    Reuses ``vqasynth.scene_fusion.SpatialSceneConstructor`` so the tool
    inherits any perf work (dtype cast, compile, AVGGT step 1) that lands
    on that surface.

    Device + dtype: ``SpatialSceneConstructor`` handles its own device
    placement + fp16 cast internally (via PR #115). ``device`` and ``dtype``
    kwargs here are accepted for symmetry with the other tools but may not
    take effect — pin VGGT to a specific GPU via ``CUDA_VISIBLE_DEVICES``
    at process start if the multi-GPU case matters.
    """

    def __init__(self, device: str | None = None, dtype: Any = None):
        self.device = device      # advisory; SpatialSceneConstructor picks its own
        self.dtype = dtype        # advisory; same
        self._constructor = None

    def _load(self):
        from vqasynth.scene_fusion import SpatialSceneConstructor
        self._constructor = SpatialSceneConstructor()

    def metric_depth(self, image) -> DepthResult:
        """Metric depth + 3D point cloud via VGGT-1B (single-image mode)."""
        if self._constructor is None:
            self._load()

        pcd, depth_map_np, focal_val = self._constructor.create_point_cloud_from_model(image)

        # depth_map_np comes out (H, W) already squeezed
        depth_m = np.asarray(depth_map_np, dtype=np.float32)
        H, W = depth_m.shape[:2]
        focal_px = float(focal_val)
        cx, cy = W / 2, H / 2
        K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1]], dtype=np.float32)

        # VGGT gives us a proper open3d point cloud — extract xyz to a (H*W, 3) array
        # then reshape. Some VGGT outputs may not preserve the (H, W, 3) grid layout;
        # fall back to unprojection if that's the case.
        try:
            xyz = np.asarray(pcd.points).reshape(H, W, 3)
        except (ValueError, AttributeError):
            xyz = _unproject(depth_m, K)

        return DepthResult(
            depth_m=depth_m,
            focal_px=focal_px,
            intrinsics_3x3=K,
            point_cloud_xyz=xyz,
            backend="vggt",
        )


# ────────────────────────────────────────────────────────────────
# GPU tier (alternative) — FoundationGeo v1.1
# ────────────────────────────────────────────────────────────────

class FoundationGeoEstimator:
    """Metric depth via FoundationGeo (arXiv:2607.11588, ECCV 2026).

    Stage-2 model produces metric depth + a *learned* pixel-wise scale field
    + ray-direction correction field. The paper's headline finding is that
    focal-length OOD is the dominant driver of zero-shot metric error;
    FoundationGeo trains on wide focal coverage and learns per-pixel
    corrections rather than relying on a global scale.

    Model: ``mxliu-hku/FoundationGeo-1.1`` (~314M params on HF Hub).
    License: composite MIT (Microsoft upstream) + Apache-2.0 (their code).

    Requires: ``pip install 'foundationgeo @ git+https://github.com/mx-liu6/FoundationGeo.git'``
    plus ``pip install 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git'``.

    Args:
        device: torch device string (e.g. ``"cuda:0"``, ``"cpu"``).
        dtype: torch dtype or alias resolved by ``_resolve_torch_dtype``.
            fp16 halves VRAM at negligible accuracy loss.
        fov_x_deg: optional horizontal FOV in degrees. When known (e.g. from
            EXIF), skip FoundationGeo's FOV estimation and pass the true value
            — this is exactly the focal-OOD case the paper's Stage-2 fields
            target. ``None`` (default) lets the model estimate FOV.
        resolution_level: integer 0-9; higher = finer detail, slower. 9 is the
            paper default and matches ``foundationgeo infer``'s CLI default.
    """
    MODEL_ID = "mxliu-hku/FoundationGeo-1.1"
    # HF repo hosts the checkpoint as `FoundationGeo.pt`, but the upstream
    # ``FoundationGeo.from_pretrained`` hardcodes ``filename="model.pt"`` in
    # its ``hf_hub_download`` call — that fails silently as "model not found".
    # Download the actual filename ourselves and hand ``from_pretrained`` a
    # local path so its ``Path(...).exists()`` branch takes over.
    MODEL_FILENAME = "FoundationGeo.pt"

    def __init__(
        self,
        device: str = "cuda",
        dtype: Any = None,
        fov_x_deg: float | None = None,
        resolution_level: int = 9,
        model_path: str | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.fov_x_deg = fov_x_deg
        self.resolution_level = resolution_level
        self.model_path = model_path
        self._model = None
        self._use_fp16 = False

    def _load(self):
        # Lazy imports — FoundationGeo isn't a mandatory VQASynth dep.
        try:
            from foundationgeo.model.v1 import FoundationGeo  # Stage-2 metric
        except ImportError as e:
            raise ImportError(
                "foundationgeo is required — install via `pip install "
                "'foundationgeo @ git+https://github.com/mx-liu6/FoundationGeo.git'` "
                "(plus utils3d from git+https://github.com/EasternJournalist/utils3d.git). "
                f"Original error: {e}"
            )
        import torch

        # Resolve the checkpoint path: use caller-supplied local path if given,
        # else fetch the correct filename from HF Hub. Works around the
        # filename mismatch in the upstream from_pretrained.
        model_path = self.model_path
        if model_path is None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id=self.MODEL_ID,
                filename=self.MODEL_FILENAME,
            )

        precision = _resolve_torch_dtype(self.dtype)
        model = FoundationGeo.from_pretrained(model_path).to(
            torch.device(self.device)
        ).eval()
        # bf16 has fp32's dynamic range with fp16's VRAM footprint. fp16 alone
        # produces NaN through FG on some scenes (ops overflow/underflow at
        # fp16 precision). Prefer bf16 as the default fast path; fall back to
        # fp32 if bf16 also NaNs.
        if precision in (torch.float16, torch.bfloat16):
            model = model.to(precision)
        # ``use_fp16`` inside FG's infer() only enables autocast, and only when
        # model params are fp32. Once we've cast to fp16/bf16 the autocast is
        # a no-op — pass False so infer doesn't try to open a mixed-precision
        # context on already-cast weights.
        self._use_fp16 = False
        self._model = model

    def metric_depth(self, image) -> DepthResult:
        """Predict metric depth + intrinsics + point cloud via FoundationGeo v1.

        Returns the Stage-2 ``depth_metric`` and ``points_metric`` outputs
        directly (already in meters, no manual unprojection needed). The
        learned scale + ray-direction correction fields are applied inside
        ``model.infer`` — they show up in the model's output dict as
        ``scalefield`` and ``delta`` and could be exposed on the DepthResult
        if downstream tools need them.
        """
        import torch

        if self._model is None:
            self._load()

        # PIL Image (or ndarray) → (C, H, W) float32 tensor in [0, 1],
        # matching what ``foundationgeo/scripts/infer.py`` expects.
        if hasattr(image, "convert"):  # PIL.Image
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
        img_tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_tensor = img_tensor.to(next(self._model.parameters()).device)

        with torch.no_grad():
            output = self._model.infer(
                img_tensor,
                fov_x=self.fov_x_deg,          # None → FG estimates FOV
                resolution_level=self.resolution_level,
                num_tokens=None,
                use_fp16=self._use_fp16,
                # apply_mask=True (FG default) fills masked pixels with
                # torch.inf, which propagates as zeros through our
                # invalid-pixel fill. Get raw metric depth here and let
                # downstream tools decide how to use the ``mask`` output.
                apply_mask=False,
            )

        # Prefer Stage-2 metric outputs; fall back to Stage-1 if the checkpoint
        # is a base (non-metric) build.
        depth_key = "depth_metric" if "depth_metric" in output else "depth"
        points_key = "points_metric" if "points_metric" in output else "points"

        depth_m = output[depth_key].cpu().numpy().astype(np.float32)
        intrinsics = output["intrinsics"].cpu().numpy().astype(np.float32)
        # Intrinsics can arrive as (3, 3) or (1, 3, 3) — drop any leading batch dim.
        while intrinsics.ndim > 2:
            intrinsics = intrinsics[0]

        # FoundationGeo returns intrinsics in NORMALIZED image coordinates
        # (fx/W, fy/H, cx/W, cy/H) — matches utils3d's convention. DepthPro
        # and VGGT return pixel-space intrinsics, and downstream tools
        # (distance_3d, unproject) assume pixels. Rescale to pixel space so
        # the DepthResult contract is uniform across backends.
        H, W = depth_m.shape[-2:]
        # Top row (fx, 0, cx) scaled by width; middle row (0, fy, cy) by height.
        intrinsics = intrinsics.copy()
        intrinsics[0, :] *= W
        intrinsics[1, :] *= H
        focal_px = float(intrinsics[0, 0])

        # FG returns metric points directly; use them instead of re-unprojecting.
        try:
            xyz = output[points_key].cpu().numpy().astype(np.float32)
        except KeyError:
            xyz = _unproject(depth_m, intrinsics)

        # Fill invalid pixels (FG masks unreliable regions with nan/inf; the
        # ``mask`` output also flags them). Downstream sampling at bbox
        # centroids can otherwise land on nan and propagate to distance calcs.
        # Fill with the median of valid depth so distance_3d degrades to a
        # reasonable rather than NaN answer.
        mask_output = output.get("mask")
        invalid = ~np.isfinite(depth_m)
        if mask_output is not None:
            mask_np = mask_output.cpu().numpy().astype(bool)
            # broadcast/align to depth_m shape if needed
            if mask_np.shape == depth_m.shape:
                invalid = invalid | ~mask_np
        if invalid.any():
            valid_depths = depth_m[~invalid]
            fill = float(np.median(valid_depths)) if valid_depths.size else 0.0
            depth_m = np.where(invalid, fill, depth_m).astype(np.float32)
            # xyz has the same invalid columns/rows; sanitize consistently.
            if xyz is not None and xyz.ndim == 3:
                xyz_invalid = ~np.isfinite(xyz).all(axis=-1) | invalid
                if xyz_invalid.any():
                    xyz = np.where(
                        xyz_invalid[..., None], np.array([0.0, 0.0, fill], dtype=np.float32), xyz
                    ).astype(np.float32)

        return DepthResult(
            depth_m=depth_m,
            focal_px=focal_px,
            intrinsics_3x3=intrinsics,
            point_cloud_xyz=xyz,
            backend="foundationgeo",
        )


# ────────────────────────────────────────────────────────────────
# Shared geometry helpers used by both tiers
# ────────────────────────────────────────────────────────────────

def _unproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject a metric depth map to (H, W, 3) XYZ using pinhole intrinsics."""
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    z = depth_m
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def depth_at_point(depth: DepthResult, x: float, y: float) -> float:
    """Sample metric depth at a specific pixel (bilinear-ish nearest for speed)."""
    H, W = depth.depth_m.shape
    xi, yi = int(round(x)), int(round(y))
    xi = max(0, min(W - 1, xi))
    yi = max(0, min(H - 1, yi))
    return float(depth.depth_m[yi, xi])


def distance_3d_meters(depth: DepthResult, box_a, box_b) -> dict:
    """Metric 3D Euclidean distance between two detected objects.

    Uses each box's center pixel + the depth map to unproject to 3D, then
    computes Euclidean distance in meters. Deterministic once depth is known.

    Args:
        depth: DepthResult from either tier's metric_depth call.
        box_a: First bounding box (Box or [x1,y1,x2,y2]).
        box_b: Second bounding box.

    Returns:
        Dict with distance_m, dx_m, dy_m, dz_m, backend.
    """
    def _center(b):
        if hasattr(b, "center"):
            return b.center
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    cx_a, cy_a = _center(box_a)
    cx_b, cy_b = _center(box_b)

    xyz = depth.point_cloud_xyz
    if xyz is None:
        xyz = _unproject(depth.depth_m, depth.intrinsics_3x3)

    def _pt(x, y):
        H, W, _ = xyz.shape
        xi, yi = max(0, min(W - 1, int(round(x)))), max(0, min(H - 1, int(round(y))))
        return xyz[yi, xi]

    pa = _pt(cx_a, cy_a)
    pb = _pt(cx_b, cy_b)
    d = pb - pa
    return {
        "distance_m": round(float(np.linalg.norm(d)), 3),
        "dx_m": round(float(d[0]), 3),
        "dy_m": round(float(d[1]), 3),
        "dz_m": round(float(d[2]), 3),
        "backend": depth.backend,
    }


__all__ = [
    "DepthResult",
    "DepthProEstimator",
    "VggtEstimator",
    "depth_at_point",
    "distance_3d_meters",
]
