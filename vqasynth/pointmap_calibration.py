"""Pixel-wise metric calibration for back-projected point maps.

Adapted from FoundationGeo (arXiv:2607.11588), whose Stage-2 contribution is
that a single global metric scale plus ideal pinhole rays is *insufficient*
for metrically consistent 3D point maps. It adds two lightweight pixel-wise
calibration fields:

  1. a spatially-varying SCALE field for metric alignment, and
  2. a RAY-DIRECTION correction field that mitigates directional bias in the
     point-map geometry.

FoundationGeo *learns* both fields from data and further stresses
focal-length distribution coverage as the dominant driver of zero-shot metric
error (metric accuracy drops sharply when the test focal length falls outside
the training distribution).

This module is a Mode-2 (adapted) port: the core mechanism — replace global
scaling + ideal rays with pixel-wise scale and ray-direction corrections to
produce a metrically consistent point map — is kept at full fidelity, but the
*learned* field estimators are replaced with parameter-free, analytic proxies
that approximate the same signals. The paper's training infrastructure,
DINOv3 init, Blender focal-length data engine, and seven-benchmark eval suite
are deliberately out of scope (evaluation belongs in a downstream PR).

The two analytic proxies:

  * ``_scale_field`` — operationalizes the paper's headline finding (focal-length
    OOD ⇒ metric bias) as a per-pixel scale. The deviation of the inferred focal
    from a reference focal rescales metric depth, and that correction is made to
    grow with radial distance from the principal point — the geometric reason
    off-axis rays accumulate more positional error per unit of focal error. This
    is a genuine field, not the global scalar the paper explicitly improves upon.

  * ``_ray_direction_correction`` — the standard pinhole back-projection treats
    every pixel ray as ideal and centered on the principal point. We apply a
    one-term radial correction to the normalized ray direction, the analytic
    stand-in for the paper's learned ray-direction field (radial lens bias +
    principal-point re-centering).

Both corrections degrade gracefully to the identity when their bias signal is
absent (focal == reference, zero distortion) — i.e. no detected bias ⇒ no
correction — which is the correct behavior, not a no-op baseline.

Reference: arXiv:2607.11588 · https://github.com/remyxai/VQASynth
"""
from __future__ import annotations

import numpy as np

# Nominal horizontal field of view (degrees) used to derive a default reference
# focal when the caller does not supply one. ~53° corresponds to fx ≈ image
# width, the conventional "standard" lens many metric-depth models train around.
_DEFAULT_HFOV_DEG = 53.0


def _as_float64_array(x):
    """Coerce a torch tensor / numpy array / nested list to a float64 ndarray.

    Accepts torch tensors without importing torch (detected via the ``.numpy``
    attribute) and moves them to CPU fp32 first, so CUDA bf16/fp16 depth maps
    and intrinsics from the autocast'd VGGT forward pass convert safely.
    """
    if hasattr(x, "numpy") and callable(x.numpy):
        x = x.detach().to("cpu").float()
        return x.numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _parse_intrinsic(intrinsic):
    """Return ``(fx, fy, cx, cy)`` from a 3x3 (or batched) camera intrinsic.

    Tolerates the several intrinsic layouts VGGT can emit ((3,3), (1,3,3),
    torch tensors) by taking the trailing 3x3 block.
    """
    K = _as_float64_array(intrinsic)
    K = K.reshape(-1, 3, 3)[-1]  # last 3x3 block (drops any leading batch dim)
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _default_reference_focal(height, width):
    """Focal length (px) the metric-depth model was calibrated around.

    Derived from the image width assuming a ~53° horizontal FOV (fx ≈ W).
    This is the in-distribution reference the scale field measures the inferred
    focal against.
    """
    hfov = np.deg2rad(_DEFAULT_HFOV_DEG)
    return (width / 2.0) / np.tan(hfov / 2.0)


def _scale_field(xn, yn, focal_px, reference_focal, strength=0.5):
    """FoundationGeo's spatially-varying metric scale field (parameter-free proxy).

    Args:
        xn, yn: per-pixel normalized image coordinates (principal point at 0).
        focal_px: focal length (px) inferred for this image.
        reference_focal: focal length (px) of the nominal / in-distribution
            camera the depth model was metrically calibrated for.
        strength: how strongly the focal deviation scales with radius in [0, 1].

    Returns:
        Per-pixel multiplicative scale. Unity on-axis (r=0) and at the principal
        point — on-axis rays are insensitive to focal error — growing toward the
        image border when ``focal_px`` deviates from ``reference_focal``.
    """
    r = np.sqrt(xn * xn + yn * yn)
    r_max = float(r.max()) if r.size else 0.0
    r_norm = r / r_max if r_max > 0 else r
    # Signed log deviation of the inferred focal from the reference focal.
    log_dev = np.log(focal_px / reference_focal)
    # Field: exp(strength * r_norm * log_dev). Unity when focal == reference or
    # on-axis; spatially varying otherwise.
    return np.exp(strength * r_norm * log_dev)


def _ray_direction_correction(xn, yn, k=0.0):
    """FoundationGeo's ray-direction correction field (parameter-free proxy).

    Applies a one-term radial correction to the normalized pixel ray direction.
    ``k`` is the radial coefficient; ``k=0`` yields ideal pinhole rays (no
    directional bias ⇒ no correction).
    """
    r2 = xn * xn + yn * yn
    factor = 1.0 + k * r2
    return xn * factor, yn * factor


def _apply_extrinsic(points, extrinsic):
    """Transform camera-space points (..., 3) to world space via a [R|t] extrinsic.

    Accepts (3,4), (4,4), (3,3) or batched variants. ``None``/unrecognized
    layouts pass through unchanged (camera space).
    """
    if extrinsic is None:
        return points
    E = _as_float64_array(extrinsic)
    while E.ndim > 2 and E.shape[0] == 1:  # drop leading size-1 batch dims
        E = E[0]
    if E.shape == (3, 4):
        R, t = E[:, :3], E[:, 3]
    elif E.shape == (4, 4):
        R, t = E[:3, :3], E[:3, 3]
    elif E.shape == (3, 3):
        R, t = E, np.zeros(3)
    else:
        return points  # unrecognized layout — leave in camera space
    flat = points.reshape(-1, 3)
    flat = flat @ R.T + t
    return flat.reshape(points.shape)


def calibrate_point_map(
    depth_map,
    extrinsic,
    intrinsic,
    *,
    reference_focal=None,
    distortion=0.0,
    scale_strength=0.5,
):
    """Back-project a metric depth map to a *calibrated* 3D point map.

    Analytic alternative to a raw pinhole ``unproject`` that applies
    FoundationGeo's two Stage-2 pixel-wise calibration fields (scale + ray
    direction). Mirrors the I/O contract of VGGT's
    ``unproject_depth_map_to_point_map`` so it can drop into the same call site.

    Args:
        depth_map: (H, W) metric depth (metres). torch tensor or ndarray.
        extrinsic: camera pose [R|t] applied after back-projection (matches the
            raw unproject's world-frame output). May be ``None``.
        intrinsic: 3x3 camera intrinsic (torch tensor or ndarray).
        reference_focal: reference focal length (px) for the scale field. If
            ``None``, derived from the image width (see
            :func:`_default_reference_focal`).
        distortion: radial coefficient for the ray-direction correction field.
        scale_strength: radial modulation strength of the scale field in [0, 1].

    Returns:
        (H, W, 3) float64 point map in world space.
    """
    depth = _as_float64_array(depth_map)
    while depth.ndim > 2 and depth.shape[0] == 1:  # drop leading size-1 dims
        depth = depth[0]
    height, width = depth.shape[-2], depth.shape[-1]

    fx, fy, cx, cy = _parse_intrinsic(intrinsic)
    if reference_focal is None:
        reference_focal = _default_reference_focal(height, width)

    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    xn = (uu - cx) / fx
    yn = (vv - cy) / fy

    # 1) ray-direction correction field (parameter-free proxy)
    xc, yc = _ray_direction_correction(xn, yn, distortion)
    # 2) spatially-varying metric scale field (parameter-free proxy)
    scale = _scale_field(xn, yn, fx, reference_focal, strength=scale_strength)
    depth_cal = depth * scale

    # z-depth back-projection (matches VGGT's pinhole convention)
    points = np.stack([depth_cal * xc, depth_cal * yc, depth_cal], axis=-1)
    return _apply_extrinsic(points, extrinsic)
