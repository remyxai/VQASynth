"""Metric3D v2 backend: joint metric depth + learned surface normals.

Metric3D v2 (Yin et al., arXiv:2404.15506) is a monocular geometric foundation
model whose contribution is *joint* zero-shot metric depth and a learned
per-pixel surface-normal head. The NOOA depth surface
(:mod:`experiments.nooa_agent.tools.depth`) already hosts three metric-depth
backends — DepthPro / VGGT / FoundationGeo — behind the uniform
``metric_depth(image) -> DepthResult`` contract; this module adds Metric3D v2
as a fourth GPU backend on that same contract.

The piece of Metric3D v2 nothing else in the repo provides is the *learned*
normal head. :mod:`experiments.nooa_agent.tools.surface_normals` derives normals
from a depth point cloud by finite-difference tangent-plane fitting — a
parameter-free proxy of this same paper — and explicitly leaves
``NormalResult.confidence`` as ``None`` ("reserved for a future learned normal
head"). Metric3D v2's normal head *is* that head, and crucially its
``prediction_normal`` output carries a 4th channel that is a per-pixel normal
confidence (normal-angle uncertainty, arXiv:2109.09881). This module surfaces
it, finally populating ``NormalResult.confidence``.

Implementation mode — **Mode 1 (direct port)** for the model surface: the
Metric3D v2 ViT is loaded lazily through its documented
``torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', ...)`` entry point
(the model has no pip package; hub load is the supported install path) and run
via its native ``model.inference({'input': rgb})`` call, which returns
``(pred_depth, confidence, output_dict)`` with
``output_dict['prediction_normal']`` a 4-channel ``(normal_xyz, confidence)``
tensor. This mirrors :class:`FoundationGeoEstimator`'s lazy / defensive-load
discipline: the module imports cleanly with neither torch nor the hub repo
present, and the heavy forward only runs at first use.

The model's raw tensor outputs are adapted onto the repo's ``DepthResult`` /
``NormalResult`` contracts by the pure, weight-free helpers below — those are
the unit-tested core (no GPU or download needed) and the part that delivers the
joint depth+normal signal to the surfaces the agent already reasons over.

Frame convention: Metric3D v2 predicts normals pointing toward the camera,
which matches the convention :mod:`surface_normals` already documents (a
frontal plane -> normal ``(0, 0, -1)`` in the x-right / y-down / z-away depth
frame). Learned normals are therefore consumed in the same frame the geometry
proxy produces — no sign flip, just a defensive unit-normalize.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# DepthResult is the uniform depth-tool contract this backend joins; _unproject
# is reused (not reimplemented) so the point cloud matches the geometry every
# other backend / distance_3d_meters uses. _resolve_torch_dtype keeps the
# device/dtype alias convention identical to the other GPU backends. NormalResult
# is the surface the learned normal head fills (its `confidence` field).
from experiments.nooa_agent.tools.depth import DepthResult, _unproject
from experiments.nooa_agent.tools.florence import _resolve_torch_dtype
from experiments.nooa_agent.tools.surface_normals import NormalResult

# Metric3D v2's canonical-camera transform targets this focal length (px at the
# canonical processing resolution). In-the-wild inference sweeps a small set of
# focal candidates; a caller who knows the true focal (EXIF / intrinsics) can
# pin the canonical transform via `focal_px` — the focal-OOD case Metric3D v2
# was designed around. Defaults to Metric3D's own canonical value.
_CANONICAL_FOCAL_PX = 1000.0


class Metric3DEstimator:
    """Metric depth + learned per-pixel normals via Metric3D v2 (arXiv:2404.15506).

    Hosted on PyTorch Hub as ``yvanyin/metric3d`` (no pip package). The v2 ViT
    checkpoints expose the joint metric-depth + normal head used here.

    Args:
        device: torch device string (e.g. ``"cuda:0"``, ``"cpu"``).
        dtype: torch dtype or string alias resolved by ``_resolve_torch_dtype``.
            fp16/bf16 halve VRAM at negligible accuracy loss for the ViT metric
            head.
        model_name: hub entry point. ``"metric3d_vit_large"`` is the default
            (accuracy/VRAM tradeoff); ``"metric3d_vit_small"`` (lighter) and
            ``"metric3d_vit_giant2"`` (heavier) are also available.
        focal_px: override the canonical focal length (px) used to unproject the
            metric depth to a point cloud. Pass the true focal when known to pin
            Metric3D's canonical-camera transform out of its in-the-wild sweep —
            exactly the focal-OOD case the v2 normal head targets. ``None`` uses
            the Metric3D canonical default.
        trust_repo: forwarded to ``torch.hub.load`` (first load of a hub repo
            otherwise prompts).
    """

    HUB_REPO = "yvanyin/metric3d"
    MODEL_ID = "metric3d_vit_large"  # default hub entry point

    def __init__(
        self,
        device: str = "cuda",
        dtype: Any = None,
        *,
        model_name: str = "metric3d_vit_large",
        focal_px: float | None = None,
        trust_repo: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.focal_px = focal_px
        self.trust_repo = trust_repo
        self._model = None

    def _load(self):
        # Lazy import — Metric3D has no pip package; torch.hub fetches the repo
        # (yvanyin/metric3d) on first load. Importing this module never touches
        # torch, so the NOOA test host stays GPU/weight-free (same discipline as
        # FoundationGeoEstimator).
        try:
            import torch
        except ImportError as e:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "torch is required for the Metric3D v2 backend — install via "
                "`pip install torch`; the model then loads through "
                "`torch.hub.load('yvanyin/metric3d', ...)` on first use. "
                f"Original error: {e}"
            ) from e

        precision = _resolve_torch_dtype(self.dtype)
        model = torch.hub.load(
            self.HUB_REPO,
            self.model_name,
            pretrain=True,
            trust_repo=self.trust_repo,
        )
        model = model.to(torch.device(self.device)).eval()
        if precision is not None:
            model = model.to(precision)
        self._model = model

    def _preprocess(self, image):
        # Metric3D's hub `model.inference` expects a normalized (C, H, W) RGB
        # float tensor at the device; it performs its own canonical resize /
        # crop internally. PIL or (H, W, 3) ndarray -> (3, H, W) float32 in
        # [0, 1], matching the hubconf entry point's expected input.
        import torch

        if hasattr(image, "convert"):  # PIL.Image
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        return tensor.to(torch.device(self.device))

    def _infer(self, image):
        """Run Metric3D v2 once; return raw ``(depth_m, normal_4ch)`` arrays.

        ``model.inference({'input': rgb})`` returns
        ``(pred_depth, confidence, output_dict)`` where
        ``output_dict['prediction_normal']`` is a 4-channel tensor — channels
        0-2 are the learned surface normal, channel 3 its per-pixel confidence.
        Both :meth:`metric_depth` and :meth:`learned_normals` draw from this one
        forward; use :meth:`depth_and_normals` to get both without re-running.
        """
        if self._model is None:
            self._load()

        rgb = self._preprocess(image)
        pred_depth, _depth_conf, output_dict = self._model.inference({"input": rgb})

        depth_m = np.asarray(pred_depth, dtype=np.float32)
        depth_m = depth_m.squeeze()
        normal_4ch = np.asarray(output_dict["prediction_normal"], dtype=np.float32)
        # Drop a leading batch dim if present -> (4, H, W) or (H, W, 4).
        while normal_4ch.ndim > 3:
            normal_4ch = normal_4ch[0]
        return depth_m, normal_4ch

    # ── public contract ───────────────────────────────────────────────

    def metric_depth(self, image) -> DepthResult:
        """Predict metric depth (meters) + intrinsics + point cloud.

        Joins DepthPro / VGGT / FoundationGeo on the uniform depth-tool
        contract. Metric3D v2's depth is metric in meters under its
        canonical-camera transform; the point cloud is unprojected with the
        canonical intrinsics (overridable via ``focal_px``) so it lands in the
        same frame :func:`distance_3d_meters` reasons over.
        """
        depth_m, _normal_4ch = self._infer(image)
        return build_depth_result(depth_m, focal_px=self.focal_px)

    def learned_normals(self, image) -> NormalResult:
        """Learned per-pixel surface normals + confidence (NormalResult).

        The Metric3D v2 contribution nothing else in the repo provides: a
        *learned* normal head with a per-pixel confidence (channel 3 of
        ``prediction_normal``). This is the head
        :func:`surface_normals.surface_normals` proxies with geometry and
        reserves ``NormalResult.confidence`` for — populated here. Tagged
        ``"metric3d+learned_normals"`` so consumers can tell a learned result
        from the geometry proxy.
        """
        _depth_m, normal_4ch = self._infer(image)
        return build_normal_result(normal_4ch, backend="metric3d")

    def depth_and_normals(self, image) -> tuple[DepthResult, NormalResult]:
        """One Metric3D v2 forward -> ``(DepthResult, NormalResult)``.

        Metric3D v2 emits depth and normals jointly; this avoids the second
        forward that calling :meth:`metric_depth` then :meth:`learned_normals`
        would cost. Prefer this when a scene needs both (the common case).
        """
        depth_m, normal_4ch = self._infer(image)
        return (
            build_depth_result(depth_m, focal_px=self.focal_px),
            build_normal_result(normal_4ch, backend="metric3d"),
        )


# ────────────────────────────────────────────────────────────────────────
# Pure, weight-free adapters — the model-independent, unit-tested core.
# Metric3D v2's raw tensor outputs -> the repo's DepthResult / NormalResult.
# None of these import torch, so they run on the GPU-free test host.
# ────────────────────────────────────────────────────────────────────────

def _pixel_intrinsics(focal_px: float | None, H: int, W: int) -> np.ndarray:
    """Pinhole K (pixels) with the principal point at the image center."""
    f = float(focal_px) if focal_px is not None else _CANONICAL_FOCAL_PX
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)


def build_depth_result(depth_m: np.ndarray, *, focal_px: float | None = None) -> DepthResult:
    """Wrap a Metric3D v2 metric depth map into the uniform DepthResult contract.

    Metric3D v2 depth is already metric (meters); we attach the canonical
    intrinsics (overridable via ``focal_px``) and unproject to a point cloud
    with the *same* :func:`_unproject` helper every other backend uses, so the
    result drops into :func:`distance_3d_meters` / boxes3d / orientation
    unchanged. Non-finite pixels (masked regions) are filled with the median
    valid depth, mirroring FoundationGeo's invalid-pixel handling.
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    depth_m = depth_m.squeeze()
    H, W = depth_m.shape

    invalid = ~np.isfinite(depth_m)
    if invalid.any():
        valid = depth_m[~invalid]
        fill = float(np.median(valid)) if valid.size else 0.0
        depth_m = np.where(invalid, fill, depth_m).astype(np.float32)

    K = _pixel_intrinsics(focal_px, H, W)
    return DepthResult(
        depth_m=depth_m,
        focal_px=float(K[0, 0]),
        intrinsics_3x3=K,
        point_cloud_xyz=_unproject(depth_m, K),
        backend="metric3d",
    )


def _split_normal_4ch(normal_4ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split Metric3D v2's prediction_normal into ``(normal_xyz, confidence)``.

    Channels 0-2 are the learned surface normal; channel 3 is the per-pixel
    normal confidence. Accepts either a channel-last ``(H, W, 4)`` /
    ``(H, W, 3)`` layout or a channel-first ``(4, H, W)`` / ``(3, H, W)`` layout
    so the helper is robust to how the checkpoint / caller sliced the tensor.
    A 3-channel input has no confidence channel — synthesize 1.0 where the
    normal is finite and 0.0 otherwise, so the hook still gets a usable map.
    """
    arr = np.asarray(normal_4ch, dtype=np.float32)
    last = arr.shape[-1] if arr.ndim == 3 else None

    if arr.ndim == 3 and last == 4:
        return arr[..., :3], arr[..., 3]
    if arr.ndim == 3 and last == 3:
        finite = np.isfinite(arr).all(axis=-1)
        return arr, np.where(finite, 1.0, 0.0).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 4:  # (4, H, W)
        return np.transpose(arr[:3], (1, 2, 0)), arr[3]
    if arr.ndim == 3 and arr.shape[0] == 3:  # (3, H, W)
        normals = np.transpose(arr, (1, 2, 0))
        finite = np.isfinite(normals).all(axis=-1)
        return normals, np.where(finite, 1.0, 0.0).astype(np.float32)
    raise ValueError(
        f"prediction_normal has unexpected shape {arr.shape}; "
        "expected (H, W, 4), (H, W, 3), (4, H, W), or (3, H, W)."
    )


def _unit_normal_safe(normals: np.ndarray) -> np.ndarray:
    """Unit-normalize the last axis; non-finite / zero rows -> zero vector.

    Mirrors surface_normals._unit's contract so a downstream sample on a bad
    pixel is detectable (norm ~ 0) rather than a spurious unit vector.
    """
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    good = (norms > 1e-12) & np.isfinite(norms)
    safe = np.where(good, norms, 1.0)
    out = normals / safe
    return np.where(good, out, 0.0).astype(np.float32)


def _sanitize_confidence(confidence: np.ndarray) -> np.ndarray:
    """Clamp Metric3D v2's normal confidence to ``[0, 1]``; non-finite -> 0.

    Metric3D v2's channel-3 is a learned normal-angle uncertainty whose raw
    scale varies by checkpoint. We clip to ``[0, 1]`` so downstream tools can
    read it uniformly (1.0 = trust the normal, 0.0 = ignore it), and treat
    NaN/inf as "no signal".
    """
    conf = np.asarray(confidence, dtype=np.float32)
    conf = np.where(np.isfinite(conf), conf, 0.0)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def build_normal_result(normal_4ch: np.ndarray, *, backend: str = "metric3d") -> NormalResult:
    """Wrap Metric3D v2's learned normal head into a NormalResult (with confidence).

    This is the call that fills the hook surface_normals.py left open: the
    learned normal head's per-pixel confidence lands on
    ``NormalResult.confidence`` (``None`` for the geometry proxy, a real map
    here). Normals are defensively unit-normalized and bad pixels zeroed, in the
    same toward-camera convention :func:`surface_normals` already uses.
    """
    normals, confidence = _split_normal_4ch(normal_4ch)
    normals = _unit_normal_safe(normals)
    confidence = _sanitize_confidence(confidence)
    return NormalResult(
        normals_xyz=normals,
        backend=f"{backend}+learned_normals",
        confidence=confidence,
    )


__all__ = [
    "Metric3DEstimator",
    "build_depth_result",
    "build_normal_result",
]
