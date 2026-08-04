"""Structural smoke tests for the NOOA describe-anything tool wrapper.

Verifies the RegionCaption dataclass shape, the bbox/mask → mask reduction, the
prompt-mutual-exclusion guard, the compact ``__repr__``, and the ``mask_bbox``
round-trip — all against a stub DescribeAnything. No CUDA, no real DAM, no
weight download. Real end-to-end DAM captioning belongs on a GPU host with the
DAM weights (see ``docker/describe_anything_stage/`` and
``tests/test_describe_anything.py``).

The stub goes through the *real* :meth:`DescribeAnything.describe` path (mask
normalization → injected stub DAM), so this is not a self-test of only the new
tool code: the wrapper must hand off to the pre-existing stage verbatim.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from experiments.nooa_agent.tools.describe import (
    DAMEstimator,
    RegionCaption,
    _bbox_of_mask,
    _mask_from_bbox,
    describe_region,
)
# Pre-existing package module (PR #130) — exercised here so the test proves the
# wrapper interoperates with the real stage the batch docker path uses, not
# just the new tool code. The stub DAM plugs into the same `dam=` seam
# tests/test_describe_anything.py uses.
from vqasynth.describe_anything import DescribeAnything


# ---------------------------------------------------------------------------
# Stub DAM — callable dam(image, mask_pil) -> str; records what it was called with
# ---------------------------------------------------------------------------
class _StubDAM:
    """Mimics the injected-DAM surface :meth:`DescribeAnything._call_dam` uses.

    Records the normalized mask (shape + foreground px) so tests can assert the
    wrapper handed DAM the right region, and returns a fixed caption.
    """

    def __init__(self, caption: str = "a wooden crate with metal edges"):
        self.calls: list[dict] = []
        self._caption = caption

    def __call__(self, image, mask_pil):
        arr = np.asarray(mask_pil)
        self.calls.append(
            {
                "image_size": image.size,
                "mask_shape": arr.shape,
                "foreground_px": int((arr > 0).sum()),
            }
        )
        return self._caption


@pytest.fixture
def stub_dam(monkeypatch):
    """Patch the module-level singleton with a DAMEstimator wrapping a stub DAM.

    Builds a real :class:`DescribeAnything` with the stub DAM injected, then
    hands it to a :class:`DAMEstimator` via the ``_stage`` seam — bypassing the
    lazy DAM load entirely. Returns the stub DAM so tests can inspect calls.
    """
    dam = _StubDAM()
    est = DAMEstimator()
    est._stage = DescribeAnything(dam=dam)  # inject, bypass the lazy DAM load
    monkeypatch.setattr(
        "experiments.nooa_agent.tools.describe._get_default_estimator",
        lambda: est,
    )
    return dam


# ---------------------------------------------------------------------------
# describe_region — reduces to a mask before captioning
# ---------------------------------------------------------------------------
def test_describe_region_with_mask_round_trips_bbox(stub_dam):
    image = Image.new("RGB", (100, 80), (40, 50, 60))
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[10:30, 20:50] = 255  # rows 10..29, cols 20..49

    result = describe_region(image, mask=mask)

    assert isinstance(result, RegionCaption)
    assert result.caption == "a wooden crate with metal edges"
    assert result.backend == "dam_3b_self_contained"
    # mask_bbox is the bbox of the mask's non-zero pixels: (x1, y1, x2, y2).
    assert result.mask_bbox == (20, 10, 50, 30)
    # DAM saw exactly one mask.
    assert len(stub_dam.calls) == 1
    assert stub_dam.calls[0]["mask_shape"] == (80, 100)


def test_describe_region_with_bbox_synthesizes_filled_mask(stub_dam):
    image = Image.new("RGB", (100, 80))
    # Same box the mask test used, expressed as a (x1, y1, x2, y2) prompt.
    result = describe_region(image, bbox=(20, 10, 50, 30))

    assert isinstance(result, RegionCaption)
    assert len(stub_dam.calls) == 1
    call = stub_dam.calls[0]
    # DAM prefers masks over boxes: it got a full-image-sized filled mask.
    assert call["mask_shape"] == (80, 100)
    # The filled rectangle covers exactly the bbox region (30 cols * 20 rows).
    assert call["foreground_px"] == (50 - 20) * (30 - 10)
    # The same box comes back on the result.
    assert result.mask_bbox == (20, 10, 50, 30)


def test_describe_region_bbox_accepts_float_box_coords(stub_dam):
    # Florence's Box uses floats; the bbox path must round, not crash, and the
    # rounded box round-trips onto mask_bbox.
    image = Image.new("RGB", (50, 50))
    result = describe_region(image, bbox=(1.4, 2.6, 11.4, 12.6))
    assert result.mask_bbox == (1, 3, 11, 13)


def test_describe_region_requires_exactly_one_prompt(stub_dam):
    image = Image.new("RGB", (32, 32))
    mask = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly one"):
        describe_region(image, bbox=(0, 0, 10, 10), mask=mask)
    with pytest.raises(ValueError, match="exactly one"):
        describe_region(image)
    # The mutual-exclusion guard fires before any DAM call.
    assert stub_dam.calls == []


def test_describe_region_rejects_non_pil_image(stub_dam):
    with pytest.raises(ValueError, match="PIL"):
        describe_region("not-an-image", bbox=(0, 0, 1, 1))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RegionCaption.__repr__ — compact, mirrors DepthResult.__repr__'s guard
# ---------------------------------------------------------------------------
def test_region_caption_repr_does_not_dump_full_caption():
    """A NOOA trace event fires per tool call, so the repr must stay one line
    and must NOT re-dump the full caption — the full text already lives on the
    ``.caption`` field (mirrors ``DepthResult.__repr__``'s ~7MB-array guard and
    ``OrientationResult.__repr__``'s float guard)."""
    long_caption = "wooden crate " + ("with detailed metal edges and weathering " * 30)
    r = RegionCaption(
        caption=long_caption,
        mask_bbox=(10, 20, 30, 40),
        backend="dam_3b_self_contained",
    )

    text = repr(r)
    assert len(text) < 200, f"repr is {len(text)} chars — probably dumping the caption"
    assert "dam_3b_self_contained" in text
    # The full long caption must NOT be present verbatim ...
    assert long_caption not in text
    # ... but a readable prefix IS surfaced, and it stays on one line.
    assert "wooden crate" in text
    assert "\n" not in text


def test_region_caption_repr_short_caption_passes_through():
    r = RegionCaption(
        caption="a red mug",
        mask_bbox=(0, 0, 10, 12),
    )
    text = repr(r)
    assert "a red mug" in text
    assert "dam_3b_self_contained" in text


# ---------------------------------------------------------------------------
# DAMEstimator.describe — composes the stage + lifts caption into RegionCaption
# ---------------------------------------------------------------------------
def test_estimator_describe_composes_stage_and_lifts_caption():
    """describe() composes the underlying DescribeAnything stage and lifts its
    plain-string caption into a RegionCaption carrying the mask's bbox."""
    dam = _StubDAM(caption="a red toolbox")
    est = DAMEstimator()
    est._stage = DescribeAnything(dam=dam)  # inject, bypass the lazy DAM load

    image = Image.new("RGB", (40, 30))
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[5:15, 10:25] = 255

    result = est.describe(image, mask)

    assert isinstance(result, RegionCaption)
    assert result.caption == "a red toolbox"
    assert result.backend == "dam_3b_self_contained"
    # mask_bbox is the bbox of the mask's non-zero pixels.
    assert result.mask_bbox == (10, 5, 25, 15)
    assert len(dam.calls) == 1


def test_estimator_defaults_match_underlying_stage():
    # Default model_id mirrors DescribeAnything's, so the tool and the batch
    # docker stage resolve the same weights.
    est = DAMEstimator()
    assert est.model_id == "nvidia/DAM-3B-Self-Contained"
    assert est.BACKEND == "dam_3b_self_contained"
    assert est._stage is None  # nothing loaded at construction


def test_get_default_estimator_is_a_singleton():
    from experiments.nooa_agent.tools import describe as describe_mod

    # Reset the module-level singleton so this test is order-independent.
    saved = describe_mod._DEFAULT_ESTIMATOR
    describe_mod._DEFAULT_ESTIMATOR = None
    try:
        first = describe_mod._get_default_estimator()
        second = describe_mod._get_default_estimator()
        assert first is second
        assert isinstance(first, DAMEstimator)
    finally:
        describe_mod._DEFAULT_ESTIMATOR = saved


# ---------------------------------------------------------------------------
# _bbox_of_mask / _mask_from_bbox — deterministic geometry helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype, fill", [
    (np.uint8, 255),
    (bool, True),
    (np.float32, 1.0),
])
def test_bbox_of_mask_each_convention(dtype, fill):
    base = np.zeros((30, 40), dtype=dtype)
    base[5:15, 10:25] = fill
    # All three mask conventions (uint8 0/255, bool, float) resolve to the same
    # foreground bbox — matching DescribeAnything._normalize_mask's rule.
    assert _bbox_of_mask(base) == (10, 5, 25, 15)


def test_bbox_of_mask_empty():
    assert _bbox_of_mask(np.zeros((10, 10), dtype=np.uint8)) == (0, 0, 0, 0)


def test_mask_from_bbox_round_trips_and_clamps():
    # Round-trip: the synthesized mask is full-image-sized (H, W) and its bbox
    # is the input box.
    mask = _mask_from_bbox((20, 10, 50, 30), width=100, height=80)
    assert mask.shape == (80, 100)
    assert _bbox_of_mask(mask) == (20, 10, 50, 30)
    # Float coords are rounded (matches how PIL.crop treats a box).
    assert _bbox_of_mask(_mask_from_bbox((1.4, 2.6, 11.4, 12.6), 50, 50)) == (1, 3, 11, 13)
