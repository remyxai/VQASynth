"""Per-pixel surface normals derived from a metric depth result.

Metric3D v2 (arXiv:2404.15506) showed that jointly estimating metric depth
*and* surface normals from a single image is the key to metric 3D recovery —
depth and normals are complementary geometric signals, and the normal head is
what disambiguates surface orientation that pure depth can't express. The
NOOA depth backends (DepthPro / VGGT / FoundationGeo in
:mod:`experiments.nooa_agent.tools.depth`) already produce a metric point
cloud on ``DepthResult.point_cloud_xyz`` but do NOT expose surface normals.
This module fills exactly that gap: it derives per-pixel surface normals from
the existing point cloud by local tangent-plane fitting, so the orientation /
3D-understanding tools can consume normals without a new model download or a
weight host.

**Adapted port (Mode 2).** The paper's *learned* per-pixel normal head is
substituted here with a parameter-free finite-difference estimate over the
depth backend's own point cloud. The metric depth itself is reused unchanged
from whichever backend produced the ``DepthResult`` — only the normal head is
proxied. Substitutions, explicitly: learned normal head → point-cloud
tangent-plane cross product; Metric3Dv2's metric-depth transformer → the
repo's existing DepthPro/VGGT/FoundationGeo backends; Metric3Dv2's CUDA
inference + NYU/KITTI benchmark suite → cut (evaluation belongs downstream).
The joint depth+normal signal — Metric3Dv2's actual contribution — is what's
delivered, on the surface the repo already has.

Frame convention (matches Metric3Dv2's "normal points toward the camera"):
the depth frame is x→right, y→down, z→away-from-camera. A surface facing the
viewer therefore has normal ≈ (0, 0, -1); a tabletop / floor facing up has
normal ≈ (0, -1, 0).

Pure-numpy — importing this module pulls in neither torch nor transformers,
so it stays importable on the NOOA test host without CUDA or weights (same
discipline as :mod:`experiments.nooa_agent.tools.depth`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# DepthResult is the uniform depth-tool contract this module consumes; _unproject
# is reused (not reimplemented) so a DepthResult whose backend skipped the point
# cloud still gets normals through the same geometry depth.py uses.
from experiments.nooa_agent.tools.depth import DepthResult, _unproject


@dataclass
class NormalResult:
    """Per-pixel surface normals derived from a :class:`DepthResult`.

    Fields:
        normals_xyz: ``(H, W, 3)`` float32 unit-norm surface normals in the
            depth camera frame (x→right, y→down, z→away-from-camera). Points
            toward the camera (Metric3Dv2 convention): a viewer-facing surface
            has normal ≈ (0, 0, -1). Pixels whose source point cloud was
            non-finite carry a zero vector — check ``np.linalg.norm > 0``
            before trusting a sample.
        backend: the depth backend the normals were derived from, suffixed
            with ``+pointcloud_normals`` so consumers can tell this is the
            derived proxy, not a learned head (e.g. ``"vggt+pointcloud_normals"``).
        confidence: reserved for a future learned normal head. The
            finite-difference proxy leaves this ``None``.
    """

    normals_xyz: np.ndarray
    backend: str
    confidence: np.ndarray | None = None

    def __repr__(self) -> str:
        # Compact one-liner — mirrors DepthResult.__repr__'s guard against
        # dumping the full (H, W, 3) array into a NOOA trace event (one fires
        # per tool call). Full normals remain accessible via ``normals_xyz``.
        H, W = self.normals_xyz.shape[:2]
        return (
            f"NormalResult(backend={self.backend!r}, shape=({H}, {W}), "
            f"has_confidence={self.confidence is not None})"
        )


def _unit(vecs: np.ndarray) -> np.ndarray:
    """Normalize the last axis; leave zero/NaN vectors as zeros (not NaN)."""
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    safe = np.where((norms > 1e-12) & np.isfinite(norms), norms, 1.0)
    out = vecs / safe
    # Rows that were zero or non-finite stay zero rather than silently unitizing.
    out = np.where((norms > 1e-12) & np.isfinite(norms), out, 0.0)
    return out


def _normals_from_pointcloud(xyz: np.ndarray, smooth: int = 1) -> np.ndarray:
    """Per-pixel unit normals via the local tangent-plane cross product.

    Tangents are finite differences in the +u (right) and +v (down) pixel
    directions, with a backward difference at the trailing row/column so every
    pixel has a valid tangent. ``normal = cross(dv, du)`` so a viewer-facing
    frontal plane yields ``(0, 0, -1)`` (toward camera) — the Metric3Dv2
    convention.

    ``smooth`` (≥0) repeats a 3×3 vector-average + renormalize that many
    times. Learned normals (Metric3Dv2) are smooth; raw cross products over a
    real depth map are noisy, so a single pass is the default. 0 keeps the raw
    cross product (useful for synthetic / noise-free inputs).
    """
    du = np.zeros_like(xyz)
    du[:, :-1] = xyz[:, 1:] - xyz[:, :-1]
    du[:, -1] = xyz[:, -1] - xyz[:, -2]
    dv = np.zeros_like(xyz)
    dv[:-1, :] = xyz[1:, :] - xyz[:-1, :]
    dv[-1, :] = xyz[-1, :] - xyz[-2, :]

    normals = np.cross(dv, du)  # → (0,0,-1) on a viewer-facing frontal plane
    normals = _unit(normals)

    for _ in range(max(0, smooth)):
        # 3×3 box average on each component, then renormalize. np.pad with
        # edge keeps the grid size fixed and avoids shrinking toward the
        # border. NaNs (from non-finite source points) are excluded from the
        # mean so they don't poison neighbors.
        padded = np.pad(normals, ((1, 1), (1, 1), (0, 0)), mode="edge")
        acc = np.zeros_like(normals)
        count = np.zeros(normals.shape[:2] + (1,), dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                win = padded[di : di + normals.shape[0], dj : dj + normals.shape[1]]
                finite = np.isfinite(win).all(axis=-1, keepdims=True)
                acc += np.where(finite, win, 0.0)
                count += finite.astype(np.float32)
        count = np.where(count > 0, count, 1.0)
        normals = _unit(acc / count)

    # Pixels whose source point was non-finite carry no real surface — zero
    # them so a downstream sample there is detectable rather than a spurious
    # unit vector.
    bad = ~np.isfinite(xyz).all(axis=-1)
    if bad.any():
        normals[bad] = 0.0
    return normals.astype(np.float32)


def surface_normals(depth: DepthResult, *, smooth: int = 1) -> NormalResult:
    """Derive per-pixel surface normals from a :class:`DepthResult`.

    Uses ``depth.point_cloud_xyz`` when the backend populated it (DepthPro
    unprojects; VGGT/FoundationGeo carry metric points directly); falls back
    to unprojecting ``depth.depth_m`` with ``depth.intrinsics_3x3`` otherwise
    — the same geometry :func:`distance_3d_meters` uses, so normals stay
    consistent with the 3D distances the agent already reasons over.

    Args:
        depth: a :class:`DepthResult` from any depth-tool backend.
        smooth: number of 3×3 vector-average smoothing passes (default 1).
            0 keeps raw cross-product normals.

    Returns:
        :class:`NormalResult` tagged ``"<backend>+pointcloud_normals"``.
    """
    xyz = depth.point_cloud_xyz
    if xyz is None:
        xyz = _unproject(depth.depth_m, depth.intrinsics_3x3)
    normals = _normals_from_pointcloud(np.asarray(xyz, dtype=np.float32), smooth=smooth)
    return NormalResult(
        normals_xyz=normals,
        backend=f"{depth.backend}+pointcloud_normals",
    )


def normal_at_pixel(normals: NormalResult | np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample the surface normal at a pixel (nearest, clipped to bounds).

    Mirrors :func:`depth_at_point`'s convention (x = column, y = row) so the
    agent can hand it the same box-centroid coordinates it already uses for
    depth sampling.
    """
    arr = normals.normals_xyz if isinstance(normals, NormalResult) else normals
    H, W = arr.shape[:2]
    xi = max(0, min(W - 1, int(round(x))))
    yi = max(0, min(H - 1, int(round(y))))
    return arr[yi, xi].astype(np.float32)


def surface_facing(normal: np.ndarray) -> str:
    """Natural-language facing for a single surface normal (pure function).

    Buckets by the dominant axis so the agent gets something quotable
    ("faces the camera", "faces upward") instead of a bare vector — the
    normal-head analog of :func:`orientation._describe_orientation`. The raw
    vector stays on :class:`NormalResult`; this string exists for prose.

    A zero / non-finite normal (e.g. sampled off a non-finite depth pixel)
    returns ``"unknown surface"``.
    """
    n = np.asarray(normal, dtype=np.float32).reshape(-1)
    if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-6:
        return "unknown surface"
    n = n / np.linalg.norm(n)
    ax, ay, az = abs(n)
    if az >= ay and az >= ax:
        return "faces the camera" if n[2] < 0 else "faces away from the camera"
    if ay >= ax:
        return "faces upward" if n[1] < 0 else "faces downward"
    return "faces left" if n[0] < 0 else "faces right"


def surface_angle(a: np.ndarray, b: np.ndarray) -> dict:
    """Angle between two surface normals (pure function).

    Use after sampling two normals (e.g. an object's contact face vs. the
    surface it rests on) to ground "is it lying flat?" in real geometry
    instead of guessing — the surface analog of :func:`orientation_delta`.
    Anti-parallel normals (≈180°) mean two faces meet flush (an object
    resting flat on a tabletop); parallel normals (≈0°) face the same way.

    Args:
        a, b: ``(3,)`` surface normals (any magnitude; unit-normalized internally).

    Returns:
        Dict with ``angle_deg`` (0..180), ``parallel`` / ``anti_parallel``
        flags, and a ``relation`` summary the agent can quote.
    """
    na = _unit(np.asarray(a, dtype=np.float32).reshape(-1))
    nb = _unit(np.asarray(b, dtype=np.float32).reshape(-1))
    cos = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos)))
    parallel = angle < 25.0
    anti_parallel = angle > 155.0
    if parallel:
        relation = "parallel — surfaces face the same way"
    elif anti_parallel:
        relation = "anti-parallel — faces meet flush (e.g. an object resting flat)"
    else:
        relation = f"tilted ~{angle:.0f}° apart"
    return {
        "angle_deg": round(angle, 1),
        "parallel": parallel,
        "anti_parallel": anti_parallel,
        "relation": relation,
    }


__all__ = [
    "NormalResult",
    "surface_normals",
    "normal_at_pixel",
    "surface_facing",
    "surface_angle",
]
