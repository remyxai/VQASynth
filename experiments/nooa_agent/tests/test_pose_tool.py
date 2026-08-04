"""Structural smoke tests for the NOOA pose tool wrapper.

Verifies the agent-facing surface of ``pose_keypoints`` against a stub backend:
the :class:`PoseResult` / :class:`Keypoint` dataclass shapes, the compact
``__repr__`` (bounded regardless of keypoint count), the no-``bbox`` "one result
per person" contract with an inferred pose bbox, and the ``bbox`` path that
restricts detection to a region and reports it back. No CUDA, no mediapipe /
ultralytics install, no model download — a :class:`vqasynth.pose.StubPoseBackend`
drives everything deterministically (the same stub pattern ``tests/test_pose.py``
uses).

This file does not merely self-test the new module: it exercises the
pre-existing :mod:`vqasynth.pose` backend ABI (:class:`StubPoseBackend`,
:func:`pose_from_keypoints`, :class:`PoseBackend`) that ``pose_keypoints``
wraps, and the AGPL opt-in contract is asserted through
:func:`vqasynth.pose._resolve_backend`. Real end-to-end pose inference belongs
on a host with MediaPipe installed (see ``tests/test_pose.py``).
"""
from __future__ import annotations

import pytest
from PIL import Image

from experiments.nooa_agent.tools.pose import (
    Keypoint,
    PoseEstimator,
    PoseResult,
    pose_keypoints,
)
# Pre-existing package modules — exercised here so the test isn't a self-test
# of only the new tool code: the stub backend + canonical pose normalization +
# AGPL opt-in refusal all come from vqasynth.pose (PR #132), which is what
# pose_keypoints wraps.
from vqasynth.pose import (
    PoseBackend,
    StubPoseBackend,
    pose_from_keypoints,
)


# A synthetic 17-keypoint pose for a 100x100 image, in COCO order. Only the
# joints named below are "visible"; the rest are None (COCO "not labeled").
W, H = 100, 100
VISIBLE = {
    "nose": [50, 12],
    "left_shoulder": [30, 25],
    "right_shoulder": [70, 25],
    "left_wrist": [20, 50],
    "right_ankle": [75, 90],
}
# A second, distinct person so multi-person scenes are exercisable.
VISIBLE_2 = {
    "nose": [80, 10],
    "left_shoulder": [60, 22],
    "right_shoulder": [90, 22],
    "left_hip": [62, 70],
    "right_hip": [92, 70],
}


def _intersects(a, b) -> bool:
    """Axis-aligned bbox intersection test (for the bbox-path contract)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class _MultiPersonBackend(PoseBackend):
    """Stub backend that returns N people, so the "one result per person"
    contract is testable without a real pose model."""

    def __init__(self, people, image_size):
        # Normalize each {name: [x, y]} fixture to the per-person list form the
        # backend contract expects (17 entries, [x, y] or None), via
        # pose_from_keypoints — exactly what StubPoseBackend does internally.
        self._people = [
            [k["xy"] for k in pose_from_keypoints(person, image_size)["keypoints"]]
            for person in people
        ]
        self._image_size = image_size

    def extract(self, image):
        # Return a shallow copy per person so callers can't mutate the fixture.
        return [list(person) for person in self._people], self._image_size


@pytest.fixture
def stub_estimator(monkeypatch):
    """Patch the module-level singleton resolver to return a PoseEstimator
    backed by StubPoseBackend — no MediaPipe install required. The real
    pose-dict → PoseResult lifting runs; only the heavy backend is stubbed."""
    fake = PoseEstimator(backend=StubPoseBackend(VISIBLE, (W, H)))
    monkeypatch.setattr(
        "experiments.nooa_agent.tools.pose._get_default_estimator",
        lambda: fake,
    )
    return fake


# ---------------------------------------------------------------------------
# Keypoint — fields render correctly (name, x, y, confidence)
# ---------------------------------------------------------------------------
def test_keypoint_fields_and_repr():
    kp = Keypoint(name="nose", x=50.0, y=12.0, confidence=0.9)
    assert kp.name == "nose"
    assert kp.x == 50.0
    assert kp.y == 12.0
    assert kp.confidence == 0.9
    # Default dataclass repr surfaces every field — the brief wants them visible.
    text = repr(kp)
    for token in ("nose", "50", "12", "0.9"):
        assert token in text


# ---------------------------------------------------------------------------
# PoseResult.__repr__ — compact, bounded regardless of keypoint count
# ---------------------------------------------------------------------------
def test_pose_result_repr_compact_regardless_of_keypoint_count():
    """A NOOA trace event fires per tool call and a scene may have several
    people × 17-33 joints each, so the repr must stay one line and must NOT
    enumerate keypoints (mirrors DepthResult / OrientationResult's guard)."""
    many = [
        Keypoint(name=f"joint_{i}", x=float(i), y=float(i * 2), confidence=1.0)
        for i in range(33)
    ]
    r = PoseResult(keypoints=many, person_bbox=(1, 2, 33, 66), backend="mediapipe")
    text = repr(r)
    assert len(text) < 140, f"repr is {len(text)} chars — probably dumping keypoints"
    assert text.count("\n") == 0
    assert "mediapipe" in text
    assert "n_keypoints=33" in text
    assert "(1, 2, 33, 66)" in text
    # Per-joint data must NOT be enumerated in the repr ...
    assert "joint_5" not in text
    assert "joint_32" not in text


def test_pose_result_repr_handles_empty_and_none_bbox():
    r = PoseResult(keypoints=[], person_bbox=None, backend="mediapipe")
    text = repr(r)
    assert "n_keypoints=0" in text
    assert "bbox=None" in text
    assert text.count("\n") == 0


# ---------------------------------------------------------------------------
# pose_keypoints — no bbox: one result per detected person, inferred bbox
# ---------------------------------------------------------------------------
def test_pose_keypoints_no_bbox_returns_one_per_person(stub_estimator):
    image = Image.new("RGB", (W, H), (10, 20, 30))
    results = pose_keypoints(image)

    assert isinstance(results, list)
    assert len(results) == 1  # the stub reports a single person
    r = results[0]
    assert isinstance(r, PoseResult)
    # backend reflects the backend actually used — "stub" under the fixture,
    # "mediapipe" under the default singleton (asserted separately).
    assert r.backend == stub_estimator.backend_name
    # Only visible joints are surfaced (the stub's VISIBLE set).
    assert len(r.keypoints) == len(VISIBLE)
    assert {kp.name for kp in r.keypoints} == set(VISIBLE)


def test_pose_keypoints_infers_person_bbox_from_visible_keypoints(stub_estimator):
    image = Image.new("RGB", (W, H))
    r = pose_keypoints(image)[0]
    # person_bbox is the min/max envelope of the visible keypoints.
    xs = [kp.x for kp in r.keypoints]
    ys = [kp.y for kp in r.keypoints]
    expected = (
        int(round(min(xs))), int(round(min(ys))),
        int(round(max(xs))), int(round(max(ys))),
    )
    assert r.person_bbox == expected


def test_pose_keypoints_multiple_people_one_result_each(monkeypatch):
    backend = _MultiPersonBackend([VISIBLE, VISIBLE_2], (W, H))
    fake = PoseEstimator(backend=backend)
    monkeypatch.setattr(
        "experiments.nooa_agent.tools.pose._get_default_estimator",
        lambda: fake,
    )

    results = pose_keypoints(Image.new("RGB", (W, H)))
    assert len(results) == 2  # one PoseResult per detected person
    assert all(isinstance(r, PoseResult) for r in results)
    # Each person's keypoints are distinct.
    assert {kp.name for kp in results[0].keypoints} == set(VISIBLE)
    assert {kp.name for kp in results[1].keypoints} == set(VISIBLE_2)


# ---------------------------------------------------------------------------
# pose_keypoints — bbox: restricts to the region, reports it back
# ---------------------------------------------------------------------------
def test_pose_keypoints_with_bbox_restricts_to_region(stub_estimator):
    image = Image.new("RGB", (W, H), (1, 2, 3))
    bbox = (10, 20, 60, 90)
    results = pose_keypoints(image, bbox=bbox)

    assert len(results) == 1
    r = results[0]
    # Contract: when a bbox is supplied, person_bbox reflects the supplied region.
    assert r.person_bbox == (10, 20, 60, 90)
    # ... which intersects the input bbox (trivially, since it equals it).
    assert _intersects(r.person_bbox, bbox)


def test_pose_keypoints_with_float_bbox_rounds_to_ints(stub_estimator):
    # Florence's Box uses floats; the bbox path must round, not crash, and
    # person_bbox must be the integer tuple the contract promises.
    image = Image.new("RGB", (W, H))
    r = pose_keypoints(image, bbox=(1.4, 2.6, 51.4, 52.6))[0]
    assert r.person_bbox == (1, 3, 51, 53)
    assert all(isinstance(v, int) for v in r.person_bbox)


def test_pose_keypoints_rejects_non_pil_image():
    with pytest.raises(ValueError, match="PIL"):
        pose_keypoints("not-an-image")  # type: ignore[arg-type]


def test_pose_keypoints_rejects_bad_bbox_shape(stub_estimator):
    image = Image.new("RGB", (W, H))
    with pytest.raises(ValueError, match="bbox"):
        pose_keypoints(image, bbox=(10, 20, 30))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PoseEstimator — lazy default backend, dtype alias validation, AGPL opt-in
# ---------------------------------------------------------------------------
def test_pose_estimator_rejects_bad_dtype_alias():
    """dtype is resolved via _resolve_torch_dtype (mirrors florence/depth/
    orientation); a bad alias must surface loudly, before any backend loads.
    _resolve_torch_dtype maps aliases onto torch dtypes, so it needs torch
    installed — guard like tests/test_pose.py guards its MediaPipe test."""
    pytest.importorskip("torch")
    est = PoseEstimator(dtype="not-a-dtype")
    with pytest.raises(ValueError, match="dtype alias"):
        est.extract(Image.new("RGB", (16, 16)))


def test_yolov8_backend_remains_opt_in():
    """YOLOv8-pose is AGPL-3.0; selecting it by name must surface the opt-in
    refusal from vqasynth.pose (never a silent default, never an auto-install).
    Per PR #132's guidance and the module docstring's licensing note."""
    est = PoseEstimator(backend="yolov8")
    with pytest.raises(ValueError, match="AGPL"):
        est.extract(Image.new("RGB", (16, 16)))


def test_pose_estimator_default_backend_name_is_mediapipe():
    # None and "mediapipe" both surface the MediaPipe default on PoseResult.
    assert PoseEstimator().backend_name == "mediapipe"
    assert PoseEstimator(backend="mediapipe").backend_name == "mediapipe"
    # A stub instance derives a short name from its class.
    assert PoseEstimator(backend=StubPoseBackend(VISIBLE, (W, H))).backend_name == "stub"


# ── API-drift guards vs upstream vqasynth.pose ───────────────────────────
#
# The wrapper composes vqasynth.pose.KeypointExtractor by passing a single
# ``backend=`` kwarg. Stub-based tests here never touch the real
# constructor, so upstream drift (kwarg rename, default change, AGPL guard
# regression) would slip past them. These guards close the gap in
# milliseconds without importing MediaPipe or ultralytics.


def test_keypoint_extractor_accepts_backend_kwarg():
    """The wrapper passes ``backend=`` to :class:`vqasynth.pose.KeypointExtractor`
    on every load — if upstream renames or drops the kwarg the stub tests
    still pass but a real run breaks. Pin the parameter name explicitly."""
    import inspect
    from vqasynth.pose import KeypointExtractor
    accepted = set(inspect.signature(KeypointExtractor.__init__).parameters)
    assert "backend" in accepted, (
        f"PoseEstimator passes backend= to KeypointExtractor, but that class "
        f"no longer accepts it (accepts: {sorted(accepted)})"
    )


def test_default_backend_name_matches_upstream_default():
    """The wrapper's ``PoseEstimator.BACKEND`` must match the string
    :class:`vqasynth.pose.KeypointExtractor` uses as its own default (both
    should be ``"mediapipe"``). A drift here would ship a wrapper whose
    default backend disagreed with the underlying default — silently."""
    import inspect
    from vqasynth.pose import KeypointExtractor
    from experiments.nooa_agent.tools.pose import PoseEstimator
    upstream_default = inspect.signature(
        KeypointExtractor.__init__
    ).parameters["backend"].default
    assert PoseEstimator.BACKEND == upstream_default, (
        f"wrapper default {PoseEstimator.BACKEND!r} != upstream default "
        f"{upstream_default!r}"
    )


def test_agpl_guard_holds_upstream():
    """The wrapper's docstring promises ``KeypointExtractor(backend="yolov8")``
    raises with an AGPL-refusal message. That behavior lives in upstream's
    :func:`_resolve_backend`; if it's dropped or its message changes, the
    wrapper's user-visible refusal path breaks silently."""
    import pytest as _pytest
    from vqasynth.pose import KeypointExtractor
    with _pytest.raises(ValueError, match="AGPL"):
        KeypointExtractor(backend="yolov8")
