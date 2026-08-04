"""Smoke tests for vqasynth.orientation.

Verifies the Orient-Anything decoding, object-isolation cropping, and the
datasets.map ``apply_transform`` plumbing against a minimal fake model — no
CUDA, no real DINOv2_MLP, no weight download. Real end-to-end orientation
validation belongs on a GPU host with the Orient-Anything repo on PYTHONPATH.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
import pytest

from vqasynth.orientation import (
    OrientationEstimator,
    crop_to_object,
    decode_angles,
)
# Pre-existing module in the package — exercised here so the test isn't a
# self-test of only the new code (filter_null is what the docker stage uses to
# drop failed rows downstream of apply_transform).
from vqasynth.utils import filter_null


# ---------------------------------------------------------------------------
# decode_angles — faithful port of Orient-Anything's get_3angle
# ---------------------------------------------------------------------------
def _logits_with_peaks(az_bin, polar_bin, rot_bin, in_dist_logit=10.0):
    """Build a (1, 902) logits vector with argmax peaks at the given bins.

    Layout matches upstream Orient-Anything's ``app.py`` construction —
    ``out_dim = 360+180+360+2``: azimuth [0:360], polar [360:540],
    rotation [540:900], confidence [900:902].
    """
    width = 360 + 180 + 360 + 2
    vec = torch.zeros(width)
    vec[az_bin] = 5.0
    vec[360 + polar_bin] = 5.0
    vec[360 + 180 + rot_bin] = 5.0
    # confidence head: [in-dist, out-dist]; make in-dist dominate.
    vec[360 + 180 + 360] = in_dist_logit
    vec[360 + 180 + 360 + 1] = 0.0
    return vec.unsqueeze(0)


def test_decode_angles_maps_bins_to_degrees():
    angles = decode_angles(_logits_with_peaks(az_bin=90, polar_bin=120, rot_bin=45))
    assert angles["azimuth"] == 90.0          # raw argmax
    assert angles["polar"] == 120 - 90        # polar shifted by -90
    assert angles["rotation"] == 45 - 180     # rotation shifted by -180
    assert angles["confidence"] == pytest.approx(
        float(torch.softmax(torch.tensor([10.0, 0.0]), dim=-1)[0]), abs=1e-5
    )


def test_decode_angles_accepts_numpy_and_1d():
    as_numpy = _logits_with_peaks(0, 0, 0).numpy()
    angles = decode_angles(as_numpy)
    assert angles["azimuth"] == 0.0
    assert angles["polar"] == -90.0
    assert angles["rotation"] == -180.0

    angles_1d = decode_angles(_logits_with_peaks(359, 179, 179).squeeze(0))
    assert angles_1d["azimuth"] == 359.0
    assert angles_1d["polar"] == 89.0
    assert angles_1d["rotation"] == -1.0


def test_decode_angles_covers_full_rotation_range():
    """Rotation is 360 bins (upstream ``out_dim = 360+180+360+2``), so bins
    in [180, 359] must map to positive degrees [0, 179]. An earlier
    revision used 180 rotation bins and silently truncated this half of
    the range — this test regresses that mistake."""
    # rot_bin=270  ->  rotation = 270 - 180 = 90 (positive, above old ceiling)
    angles = decode_angles(_logits_with_peaks(az_bin=0, polar_bin=90, rot_bin=270))
    assert angles["rotation"] == 90.0
    # rot_bin=359  ->  rotation = 359 - 180 = 179 (max rotation value)
    angles = decode_angles(_logits_with_peaks(az_bin=0, polar_bin=90, rot_bin=359))
    assert angles["rotation"] == 179.0


# ---------------------------------------------------------------------------
# crop_to_object — isolates one object per the model's "Best Practice"
# ---------------------------------------------------------------------------
def test_crop_to_object_isolates_and_squares():
    image = Image.new("RGB", (40, 30), (10, 20, 30))
    # A wide-but-short object: the square crop (side = max dim) is then taller
    # than the object, leaving background rows that should be whited out.
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[2:6, 3:11] = 255

    crop = crop_to_object(image, mask, padding=0.0)

    assert crop.size[0] == crop.size[1]          # square output
    arr = np.asarray(crop)
    assert arr.shape[2] == 3                      # RGB
    # Object pixels (the original fill color) are preserved ...
    assert np.any(np.all(arr == [10, 20, 30], axis=2))
    # ... and the background outside the mask is whited out.
    assert np.any(np.all(arr == [255, 255, 255], axis=2))


def test_crop_to_object_empty_mask_returns_original():
    image = Image.new("RGB", (16, 16), (7, 8, 9))
    mask = np.zeros((16, 16), dtype=np.uint8)
    crop = crop_to_object(image, mask)
    assert crop.size == image.size


def test_crop_to_object_object_at_edge_does_not_crash():
    image = Image.new("RGB", (20, 20), (1, 2, 3))
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:6, 14:20] = 255  # hugs the top-right corner
    crop = crop_to_object(image, mask, padding=0.2)
    assert crop.size[0] == crop.size[1]


def test_crop_to_object_shape_mismatch_raises():
    image = Image.new("RGB", (20, 20))
    mask = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_to_object(image, mask)


# ---------------------------------------------------------------------------
# OrientationEstimator.run — drives a fake model through decode_angles
# ---------------------------------------------------------------------------
class _FakeModel:
    """Stand-in for Orient-Anything's DINOv2_MLP: returns fixed logits."""

    def __init__(self, logits):
        self._logits = logits
        self.eval_calls = 0

    def eval(self):  # noqa: D401 - mirrors nn.Module surface
        self.eval_calls += 1
        return self

    def to(self, device):  # noqa: D401 - mirrors nn.Module surface
        return self

    def __call__(self, img_inputs):
        # Real DINOv2_MLP.forward takes the preprocessor dict; mirror that.
        batch = img_inputs["pixel_values"].shape[0]
        return self._logits.expand(batch, -1)


class _FakePreprocess:
    """Stand-in for the DINOv2 AutoImageProcessor."""

    def __call__(self, images, return_tensors="pt"):
        # Real processors hand back a list/tensor of pixel values; we mimic the
        # tensor path that OrientationEstimator.run handles.
        return {"pixel_values": torch.zeros((1, 3, 224, 224))}


def test_run_decodes_through_fake_model():
    logits = _logits_with_peaks(az_bin=200, polar_bin=45, rot_bin=10)
    est = OrientationEstimator(model=_FakeModel(logits), preprocess=_FakePreprocess())
    angles = est.run(Image.new("RGB", (32, 32)))

    assert angles["azimuth"] == 200.0
    assert angles["polar"] == 45 - 90
    assert angles["rotation"] == 10 - 180
    assert 0.0 <= angles["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# apply_transform — datasets.map integration (batched + unbatched)
# ---------------------------------------------------------------------------
def _two_object_masks():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    mask[12:18, 12:18] = 255
    return mask


def test_apply_transform_unbatched_adds_orientation_column():
    est = OrientationEstimator(
        model=_FakeModel(_logits_with_peaks(10, 20, 30)),
        preprocess=_FakePreprocess(),
    )
    example = {
        "image": Image.new("RGB", (20, 20)),
        "masks": [_two_object_masks(), np.zeros((20, 20), dtype=np.uint8)],
    }
    out = est.apply_transform(example, images="image")

    assert "orientation" in out
    # One orientation per mask, aligned by index.
    assert len(out["orientation"]) == 2
    assert out["orientation"][0]["azimuth"] == 10.0
    # The second mask is empty -> crop_to_object returns the whole image and we
    # still get a (synthetic) estimate, not a None.
    assert out["orientation"][1]["azimuth"] == 10.0


def test_apply_transform_batched_one_list_per_image():
    est = OrientationEstimator(
        model=_FakeModel(_logits_with_peaks(5, 6, 7)),
        preprocess=_FakePreprocess(),
    )
    example = {
        "image": [Image.new("RGB", (20, 20)), Image.new("RGB", (20, 20))],
        "masks": [
            [_two_object_masks()],
            [np.zeros((20, 20), dtype=np.uint8)],
        ],
    }
    out = est.apply_transform(example, images="image")

    assert len(out["orientation"]) == 2            # one list per image
    assert len(out["orientation"][0]) == 1
    assert out["orientation"][0][0]["azimuth"] == 5.0


def test_apply_transform_failure_falls_back_to_none():
    # A non-PIL image trips the outer try/except in apply_transform, which
    # degrades the whole row to None (run_objects' per-mask except can't fire
    # because image extraction fails before any mask is processed).
    est = OrientationEstimator(
        model=_FakeModel(_logits_with_peaks(0, 0, 0)),
        preprocess=_FakePreprocess(),
    )
    example = {"image": "not-an-image", "masks": []}
    out = est.apply_transform(example, images="image")
    assert out["orientation"] is None


# ---------------------------------------------------------------------------
# Integration with the pre-existing filter_null (used by the docker stage)
# ---------------------------------------------------------------------------
def test_filter_null_drops_failed_orientation_rows():
    # Simulate a batched dataset row where one image's orientation failed.
    batch = {
        "image": [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
        "orientation": [
            [{"azimuth": 1.0, "polar": 0.0, "rotation": 0.0, "confidence": 0.9}],
            None,  # the failure
        ],
    }
    keep = filter_null(batch)
    assert keep == [True, False]


# ---------------------------------------------------------------------------
# Shape-guard: layout must match the real Orient-Anything head
# ---------------------------------------------------------------------------
def test_output_shape_matches_upstream():
    """The bin-count constants must sum to Orient-Anything's authoritative
    ``out_dim = 360+180+360+2 = 902`` (see upstream ``app.py`` and
    ``inference.get_3angle``).

    This is a cheap correctness guard: the fake-model tests above verify
    the decode logic against whatever layout ``_logits_with_peaks``
    builds, so they pass under any self-consistent choice of constants.
    Only a check against the true model's output shape catches the case
    where the wrapper drifted from upstream. A first revision of this
    file used 180 rotation bins (out_dim=720) after misreading a README
    fragment against the actual code — the plumbing tests all passed,
    but half the rotation range was silently truncated on real weights.
    """
    from vqasynth.orientation import (
        _AZIMUTH_BINS, _POLAR_BINS, _ROTATION_BINS, _CONFIDENCE_BINS,
    )
    total = _AZIMUTH_BINS + _POLAR_BINS + _ROTATION_BINS + _CONFIDENCE_BINS
    assert total == 902, (
        f"Orient-Anything head layout drift: bins sum to {total}, "
        f"expected 902 (see upstream app.py `out_dim=360+180+360+2`)"
    )
    # Rotation is 360 bins upstream. This is the specific value the earlier
    # revision got wrong; pin it explicitly so a well-meaning "fix" to
    # `_ROTATION_BINS = 180` fails loudly here.
    assert _ROTATION_BINS == 360


@pytest.mark.skipif(
    "ORIENT_ANYTHING_PATH" not in __import__("os").environ,
    reason="Set ORIENT_ANYTHING_PATH=/path/to/Orient-Anything to run the "
           "real-model shape check (~30s CPU, ~90MB DINOv2 backbone "
           "download; no orient-anything weights required — untrained "
           "heads are enough for the shape assertion).",
)
def test_output_shape_against_real_orient_anything():
    """Optional integration: load the real DINOv2_MLP with no orient-
    anything weights, forward-pass a dummy image, assert the output
    shape is 902.

    Runs only when the Orient-Anything repo is on PYTHONPATH via
    ``ORIENT_ANYTHING_PATH``. No orient-anything checkpoint download —
    the shape assertion doesn't need trained heads; it only needs the
    DINOv2 backbone (fetched from HuggingFace's ``facebook/dinov2-
    large``, ~90MB) plus the untrained MLP heads.
    """
    import os, sys
    sys.path.insert(0, os.environ["ORIENT_ANYTHING_PATH"])
    from vision_tower import DINOv2_MLP
    from transformers import AutoImageProcessor
    import torch, numpy as np
    from PIL import Image

    dino = DINOv2_MLP(
        dino_mode="large", in_dim=1024, out_dim=360 + 180 + 360 + 2,
        evaluate=True, mask_dino=False, frozen_back=True,
    ).eval()
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    with torch.no_grad():
        inputs = proc(images=img, return_tensors="pt")
        out = dino(inputs)
    assert out.shape[-1] == 902, f"got out_dim={out.shape[-1]}, expected 902"
