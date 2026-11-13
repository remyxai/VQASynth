"""CPU tests for the Phase-3 grounded-state record builder in vqasynth.prompts.

``PromptGenerator.build_grounded_state`` assembles the compact, JSON-serializable
scene-state record that the prompt stage carries forward and that
``SoftTraceReasoner`` conditions traces on. Following the CPU-mock philosophy of
the other stage tests, ``vqasynth.scene_fusion`` is stubbed by a fake module so
the builder's lazy ``from vqasynth.scene_fusion import restore_pointclouds``
resolves to a controllable identity — no open3d / torch / vggt import, no CPU-box
segfault. The fake clouds expose only ``get_center()``, all the builder reads.
The final test closes the loop: a record produced here must render through
``SoftTraceReasoner._format_grounded_block`` (A -> B contract).
"""
from __future__ import annotations

import json
import random
import sys
import types

import numpy as np
import pytest

from vqasynth.prompts import PromptGenerator
from vqasynth.r1_reasoning import SoftTraceReasoner


class _FakeCloud:
    def __init__(self, center):
        self._c = np.asarray(center, dtype=float)

    def get_center(self):
        return self._c


@pytest.fixture(autouse=True)
def fake_scene_fusion(monkeypatch):
    """Stub vqasynth.scene_fusion so build_grounded_state's lazy import gets a
    controllable restore_pointclouds without importing the real (torch/open3d/
    vggt) module. setitem auto-restores sys.modules after each test."""
    fake = types.ModuleType("vqasynth.scene_fusion")
    fake.restore_pointclouds = lambda pcs: pcs  # identity: feed fake clouds through
    monkeypatch.setitem(sys.modules, "vqasynth.scene_fusion", fake)
    random.seed(0)  # pair sampling is shuffled; keep the run deterministic
    return fake


CAPTIONS = ["chair", "mug", "lamp"]
CLOUDS = [_FakeCloud([0.1, 0.0, 2.1]), _FakeCloud([-0.3, 0.1, 0.7]), _FakeCloud([0.5, 0.0, 3.0])]


def _build(**kw):
    return json.loads(PromptGenerator().build_grounded_state(CAPTIONS, CLOUDS, **kw))


def test_objects_carry_label_center_and_depth_conf():
    rec = _build(depth_conf=[0.9, 0.3, 0.75], extrinsics=[object(), object()])
    assert rec["frames"] == 2
    objs = {o["label"]: o for o in rec["objects"]}
    assert set(objs) == {"chair", "mug", "lamp"}
    assert objs["chair"]["center_m"] == [0.1, 0.0, 2.1]
    assert objs["chair"]["depth_conf"] == 0.9
    assert objs["mug"]["depth_conf"] == 0.3


def test_pairs_carry_calibrated_distance_and_inlier_ratio():
    rec = _build(depth_conf=[0.9, 0.3, 0.75], inlier_ratio=0.82)
    assert len(rec["pairs"]) == 3  # 3 objects -> 3 pairs, all within max_pairs=5
    by_key = {frozenset((p["a"], p["b"])): p for p in rec["pairs"]}
    chair_mug = by_key[frozenset(("chair", "mug"))]
    expected = float(np.linalg.norm(np.array([0.4, -0.1, 1.4])))  # 1.4595...
    assert chair_mug["distance_m"] == pytest.approx(round(expected, 3), abs=1e-3)
    assert all(p["inlier_ratio"] == 0.82 for p in rec["pairs"])


def test_single_frame_defaults_and_no_conf():
    rec = _build()
    assert rec["frames"] == 1
    assert all("depth_conf" not in o for o in rec["objects"])
    assert all("inlier_ratio" not in p for p in rec["pairs"])


def test_inlier_ratio_list_is_reduced_to_scene_scalar():
    rec = _build(inlier_ratio=[0.8, 0.6, None, "bad"])  # mean of {0.8,0.6} = 0.7
    assert all(p["inlier_ratio"] == 0.7 for p in rec["pairs"])


def test_scene_scalar_helper():
    f = PromptGenerator._scene_scalar
    assert f(0.82) == 0.82
    assert f([0.8, 0.6]) == 0.7
    assert f([None, "x"]) is None
    assert f(None) is None


def test_restore_failure_degrades_to_empty(fake_scene_fusion):
    def _boom(_):
        raise RuntimeError("no open3d")

    fake_scene_fusion.restore_pointclouds = _boom
    assert PromptGenerator().build_grounded_state(CAPTIONS, CLOUDS) == "{}"


def test_record_renders_through_soft_trace_reasoner():
    # A -> B contract: the builder's output must be consumable by the reasoner.
    gs = PromptGenerator().build_grounded_state(
        CAPTIONS, CLOUDS, depth_conf=[0.9, 0.3, 0.75], inlier_ratio=0.82,
        extrinsics=[object(), object()],
    )
    r = SoftTraceReasoner(api_key="x", model="m", image_column="image", text_column="messages")
    block = r._format_grounded_block(gs)
    assert "chair" in block and "mug" in block
    assert "fairly confident" in block                   # chair depth_conf 0.9
    assert "low confidence, a rough estimate" in block   # mug depth_conf 0.3
    assert "2 views" in block
    assert "solid" in block                              # inlier_ratio 0.82 >= 0.7
