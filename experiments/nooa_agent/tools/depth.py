"""Metric depth + intrinsics + 3D point cloud tools, resource-tier aware.

Both tiers produce the same interface: `(depth_map_2d, focal_length, point_cloud_xyz)`.
Downstream tools (distance-in-meters, height, "on top of") consume the interface
uniformly regardless of which model produced it.

- CPU tier: **Apple DepthPro** (~330M) — metric depth + predicted focal length.
  ~1-3 s per image on modern CPUs. Continuity with VQASynth's pre-VGGT default
  depth model.
- GPU tier: **VGGT-1B** (via ``vqasynth.scene_fusion.SpatialSceneConstructor``).
  Matches the current VQASynth production path; also produces multi-view fusion
  if the annotator ever passes more than one image per call.

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
