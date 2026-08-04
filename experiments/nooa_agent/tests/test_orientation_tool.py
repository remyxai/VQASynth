"""Structural smoke tests for the NOOA orientation tool wrapper.

Verifies the OrientationResult dataclass shape, the bbox/mask → single-object
crop reduction, the prompt-mutual-exclusion guard, and the natural-language
description synthesis — all against a stub estimator. No CUDA, no real
DINOv2_MLP, no weight download. Real end-to-end orientation inference belongs
on a GPU host with the Orient-Anything repo on PYTHONPATH (see
``tests/test_orientation.py``).
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from experiments.nooa_agent.tools.orientation import (
    OrientationResult,
    OrientAnythingEstimator,
    _describe_orientation,
    orient_object,
    orientation_delta,
)
# Pre-existing package module — exercised here so the test isn't a self-test
# of only the new tool code: the mask path must hand off to crop_to_object
# verbatim (same square + white-out isolation the batch docker stage uses).
from vqasynth.orientation import crop_to_object


# ---------------------------------------------------------------------------
# Stub estimator — stands in for the module-level singleton
# ---------------------------------------------------------------------------
class _RecordingEstimator(OrientAnythingEstimator):
    """Capture the cropped image handed to ``estimate`` without loading weights.

    Subclasses OrientAnythingEstimator so isinstance / duck-typing stays honest;
    ``estimate`` records its argument and returns a fixed OrientationResult.
    """

    def __init__(self):  # noqa: D401 - test stub, no device/dtype plumbing
        # Skip the parent __init__ — no model, no lazy load.
        self.received: list[Image.Image] = []

    def estimate(self, image) -> OrientationResult:  # type: ignore[override]
        self.received.append(image)
        return OrientationResult(
            azimuth_deg=90.0,
            polar_deg=0.0,
            rotation_deg=0.0,
            confidence=0.9,
            description="facing east, level, upright, high confidence",
            backend="orient_anything_v1",
        )


@pytest.fixture
def stub_estimator(monkeypatch):
    """Patch the module-level singleton resolver to return a recording stub."""
    fake = _RecordingEstimator()
    monkeypatch.setattr(
        "experiments.nooa_agent.tools.orientation._get_default_estimator",
        lambda: fake,
    )
    return fake


# ---------------------------------------------------------------------------
# orient_object — reduces to a cropped single-object image before inferring
# ---------------------------------------------------------------------------
def test_orient_object_with_bbox_crops_before_inferring(stub_estimator):
    image = Image.new("RGB", (100, 80), (40, 50, 60))
    result = orient_object(image, bbox=(10, 20, 30, 40))  # 20x20 region

    assert isinstance(result, OrientationResult)
    assert len(stub_estimator.received) == 1
    crop = stub_estimator.received[0]
    # Direct crop path: the estimator saw exactly the bbox sub-image.
    assert crop.size == (20, 20)
    assert crop.size == image.crop((10, 20, 30, 40)).size


def test_orient_object_bbox_accepts_float_box_coords(stub_estimator):
    # Florence's Box uses floats; the bbox path must round, not crash.
    image = Image.new("RGB", (50, 50))
    orient_object(image, bbox=(1.4, 2.6, 11.4, 12.6))
    assert stub_estimator.received[0].size == (10, 10)


def test_orient_object_with_mask_uses_crop_to_object(stub_estimator):
    image = Image.new("RGB", (40, 30), (10, 20, 30))
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[2:6, 3:11] = 255  # a wide-but-short object hugging rows 2..5

    result = orient_object(image, mask=mask)

    assert isinstance(result, OrientationResult)
    assert len(stub_estimator.received) == 1
    crop = stub_estimator.received[0]
    # Mask path hands off to the pre-existing crop_to_object verbatim — square
    # output matching the batch-stage isolation, not a naive image.crop.
    assert crop.size[0] == crop.size[1]
    assert crop.size == crop_to_object(image, mask).size


def test_orient_object_requires_exactly_one_prompt():
    image = Image.new("RGB", (32, 32))
    mask = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly one"):
        orient_object(image, bbox=(0, 0, 10, 10), mask=mask)
    with pytest.raises(ValueError, match="exactly one"):
        orient_object(image)


def test_orient_object_rejects_non_pil_image(stub_estimator):
    with pytest.raises(ValueError, match="PIL"):
        orient_object("not-an-image", bbox=(0, 0, 1, 1))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OrientationResult.__repr__ — compact, mirrors DepthResult.__repr__'s guard
# ---------------------------------------------------------------------------
def test_orientation_result_repr_is_compact():
    """A NOOA trace event fires per tool call, so the repr must stay one line
    and must NOT re-dump the full float set — the raw values already live on
    the named fields and the ``description`` field carries the trace-readable
    view (mirrors ``DepthResult.__repr__``'s ~7MB-array guard)."""
    r = OrientationResult(
        azimuth_deg=123.456,
        polar_deg=-42.0,
        rotation_deg=57.0,
        confidence=0.8123,
        description="facing southeast, tilted downward, slight roll right, high confidence",
        backend="orient_anything_v1",
    )
    text = repr(r)
    assert len(text) < 200, f"repr is {len(text)} chars — probably dumping floats"
    assert "orient_anything_v1" in text
    # The raw angle floats must NOT be dumped verbatim ...
    assert "123.456" not in text
    assert "-42" not in text
    assert "57" not in text
    # ... but the trace-readable description IS surfaced.
    assert "facing southeast" in text


def test_orientation_result_repr_is_one_line():
    r = OrientationResult(
        azimuth_deg=0.0, polar_deg=0.0, rotation_deg=0.0, confidence=0.5,
        description="facing north, level, upright, moderate confidence",
    )
    assert "\n" not in repr(r)


# ---------------------------------------------------------------------------
# _describe_orientation — deterministic prose mapping (unit-tested in isolation)
# ---------------------------------------------------------------------------
def test_describe_orientation_is_deterministic():
    args = (azimuth := 90, polar := 10, rotation := -5, confidence := 0.9)
    assert _describe_orientation(*args) == _describe_orientation(*args)


def test_describe_orientation_maps_compass_and_tilt_and_confidence():
    d = _describe_orientation(azimuth=270, polar=70, rotation=120, confidence=0.2)
    assert "west" in d              # 270° → west
    assert "upward" in d            # polar 70 → steeply upward
    assert "right" in d             # rotation 120 → rolled hard right
    assert "low confidence" in d    # confidence 0.2 < 0.4


def test_describe_orientation_level_and_upright_near_zero():
    d = _describe_orientation(azimuth=0, polar=0, rotation=0, confidence=0.9)
    assert d == "facing north, level, upright, high confidence"


def test_result_from_angles_via_estimator_builds_description():
    """estimate() composes the underlying estimator and lifts its dict output
    into an OrientationResult that carries the synthesized description."""
    class _FakeUnderlying:
        def __init__(self, angles):
            self._angles = angles
            self.received = []

        def run(self, image):
            self.received.append(image)
            return self._angles

    fake = _FakeUnderlying(
        {"azimuth": 90.0, "polar": 0.0, "rotation": 0.0, "confidence": 0.95}
    )
    est = OrientAnythingEstimator()
    est._estimator = fake  # inject, bypass the lazy Orient-Anything load

    result = est.estimate(Image.new("RGB", (16, 16)))
    assert result.azimuth_deg == 90.0
    assert result.polar_deg == 0.0
    assert result.rotation_deg == 0.0
    assert result.confidence == 0.95
    assert result.backend == "orient_anything_v1"
    assert "east" in result.description
    # The underlying estimator saw the image exactly once.
    assert len(fake.received) == 1


# ---------------------------------------------------------------------------
# orientation_delta — signed + wrapped, narrative-language field present
# ---------------------------------------------------------------------------
def test_orientation_delta_signed_and_wrapped():
    a = OrientationResult(
        azimuth_deg=10, polar_deg=0, rotation_deg=0, confidence=0.9,
        description="a",
    )
    b = OrientationResult(
        azimuth_deg=350, polar_deg=5, rotation_deg=0, confidence=0.9,
        description="b",
    )
    delta = orientation_delta(a, b)
    # 350 - 10 = 340 → wrapped to -20 (minimal rotation the other way).
    assert delta["azimuth_delta_deg"] == -20.0
    assert delta["polar_delta_deg"] == 5.0
    assert delta["rotation_delta_deg"] == 0.0
    assert delta["backend"] == "orient_anything_v1"
    # Same shape contract as distance_3d_meters: a narrative-language field.
    assert isinstance(delta["b_is"], str) and delta["b_is"]


def test_orientation_delta_same_direction_narrative():
    a = OrientationResult(
        azimuth_deg=90, polar_deg=0, rotation_deg=0, confidence=0.9,
        description="a",
    )
    b = OrientationResult(
        azimuth_deg=95, polar_deg=0, rotation_deg=0, confidence=0.9,
        description="b",
    )
    delta = orientation_delta(a, b)
    assert "same direction" in delta["b_is"]
