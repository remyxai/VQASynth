"""
Tests for SpaceDG-style degradation-aware evaluation.

Covers both the standalone degradation operators (vqasynth.image_degradation)
and their integration into the existing evaluation stage via
vqasynth.benchmarks.BenchmarkRunner (a non-new module).
"""

import numpy as np
import pytest
from PIL import Image

from vqasynth.image_degradation import (
    DEGRADATION_TYPES,
    apply_degradation,
    degrade_images,
)
# Import the existing call-site module to exercise the wiring edit.
from vqasynth.benchmarks import BenchmarkRunner


def _sample_image(seed=0):
    """A small, non-uniform RGB image so every operator has signal to alter."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _max_abs_diff(a, b):
    return int(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max())


# ---------------------------------------------------------------------------
# Standalone operator behavior
# ---------------------------------------------------------------------------

def test_registry_has_nine_degradations():
    assert len(DEGRADATION_TYPES) == 9


@pytest.mark.parametrize("name", DEGRADATION_TYPES)
def test_each_degradation_changes_pixels_and_keeps_shape(name):
    img = _sample_image()
    out = apply_degradation(img, name, severity=3)
    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert _max_abs_diff(img, out) > 0, f"{name} left the image unchanged"


def test_severity_monotonically_increases_distortion():
    img = _sample_image()
    mild = apply_degradation(img, "gaussian_noise", severity=1)
    severe = apply_degradation(img, "gaussian_noise", severity=5)
    assert _max_abs_diff(img, severe) > _max_abs_diff(img, mild)


def test_degradation_is_deterministic():
    img = _sample_image()
    a = apply_degradation(img, "gaussian_noise", severity=4)
    b = apply_degradation(img, "gaussian_noise", severity=4)
    assert _max_abs_diff(a, b) == 0


def test_invalid_name_raises():
    with pytest.raises(ValueError):
        apply_degradation(_sample_image(), "nope", severity=2)


@pytest.mark.parametrize("bad", [0, 6, 2.5, "3"])
def test_invalid_severity_raises(bad):
    with pytest.raises(ValueError):
        apply_degradation(_sample_image(), "fog", severity=bad)


def test_degrade_images_passes_through_uncoercible_entries():
    img = _sample_image()
    out = degrade_images([img, None, 12345], "fog", severity=2)
    assert len(out) == 3
    assert _max_abs_diff(img, out[0]) > 0   # PIL image degraded
    assert out[1] is None                   # None passed through
    assert out[2] == 12345                  # non-image passed through


# ---------------------------------------------------------------------------
# Integration: BenchmarkRunner wiring (the call site)
# ---------------------------------------------------------------------------

def _fake_items():
    return [
        {
            "id": "0",
            "question": "Which is closer?",
            "answer": "A",
            "question_type": "multi-choice",
            "category": "cat",
            "subcategory": "sub",
            "options": ["A", "B"],
            "images": [_sample_image(seed=1)],
            "source": "fake",
        }
    ]


def test_runner_get_benchmark_items_degrades_when_configured(monkeypatch):
    clean = BenchmarkRunner(benchmarks="spatialscore")
    degraded = BenchmarkRunner(
        benchmarks="spatialscore", degradation="motion_blur", severity=4
    )
    # Avoid any network/dataset access: stub the loader for both runners.
    monkeypatch.setattr(clean, "load", lambda name, **kw: _fake_items())
    monkeypatch.setattr(degraded, "load", lambda name, **kw: _fake_items())

    clean_items = clean.get_benchmark_items("spatialscore")
    degraded_items = degraded.get_benchmark_items("spatialscore")

    original = _fake_items()[0]["images"][0]
    # Clean path leaves the image untouched...
    assert _max_abs_diff(original, clean_items[0]["images"][0]) == 0
    # ...degraded path actually perturbs it through the new operator.
    assert _max_abs_diff(original, degraded_items[0]["images"][0]) > 0


def test_runner_degrade_items_preserves_ground_truth(monkeypatch):
    runner = BenchmarkRunner(
        benchmarks="spatialscore", degradation="low_light", severity=3
    )
    items = runner.degrade_items(_fake_items())
    # Degradation must not disturb labels/metadata needed for scoring.
    assert items[0]["answer"] == "A"
    assert items[0]["question"] == "Which is closer?"
    original = _fake_items()[0]["images"][0]
    assert _max_abs_diff(original, items[0]["images"][0]) > 0


def test_runner_no_degradation_is_noop():
    runner = BenchmarkRunner(benchmarks="spatialscore")
    items = _fake_items()
    assert runner.degrade_items(items) is items
