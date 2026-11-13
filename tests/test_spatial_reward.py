"""CPU tests for the SpatialThinker-style dense reward (vqasynth.spatial_reward).

Verifies the ported reward (arXiv:2511.07403): CIoU, <scene> parsing, the four
components (format / count / accuracy / spatial), and the lexicographic gating of
the total. Pure CPU (scipy Hungarian matching); no model, no GPU.
"""
from __future__ import annotations

import pytest

from vqasynth import spatial_reward as sr

GT_SCENE = {
    "objects": [
        {"id": 0, "label": "red cube", "bbox": [10, 10, 50, 50]},
        {"id": 1, "label": "green sphere", "bbox": [100, 100, 140, 140]},
    ],
    "relations": [{"subject": 0, "predicate": "left of", "object": 1}],
}

GOOD_OUTPUT = (
    "<observe>A red cube and a green sphere.</observe>"
    '<scene>{"objects":[{"id":0,"label":"red cube","bbox":[10,10,50,50]},'
    '{"id":1,"label":"green sphere","bbox":[100,100,140,140]}],'
    '"relations":[{"subject":0,"predicate":"left of","object":1}]}</scene>'
    "<think>The cube sits clearly to the left.</think>"
    "<answer>(A) left</answer>"
)


def test_ciou_identical_and_disjoint():
    assert sr.ciou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    disjoint = sr.ciou([0, 0, 10, 10], [100, 100, 110, 110])
    assert disjoint < 0  # far apart -> IoU 0 minus center/aspect penalties


def test_parse_scene():
    scene = sr.parse_scene(GOOD_OUTPUT)
    assert scene is not None and len(scene["objects"]) == 2 and len(scene["relations"]) == 1
    assert sr.parse_scene("<scene>not json</scene>") is None
    assert sr.parse_scene("no tags here") is None


def test_format_reward():
    assert sr.format_reward(GOOD_OUTPUT) == 1.0
    # missing <observe> tag
    assert sr.format_reward(GOOD_OUTPUT.replace("<observe>A red cube and a green sphere.</observe>", "")) == 0.0
    # scene present but object lacks bbox
    bad = (
        "<observe>x</observe><scene>{\"objects\":[{\"id\":0,\"label\":\"c\"}]}</scene>"
        "<think>y</think><answer>z</answer>"
    )
    assert sr.format_reward(bad) == 0.0


def test_accuracy_reward_choice_letter_aware():
    assert sr.accuracy_reward(GOOD_OUTPUT, "(A) left") == 1.0
    assert sr.accuracy_reward(GOOD_OUTPUT, "A") == 1.0  # choice-letter match
    assert sr.accuracy_reward(GOOD_OUTPUT, "(B) right") == 0.0
    assert sr.accuracy_reward("no answer tag", "A") == 0.0


def test_count_reward():
    pred = sr.parse_scene(GOOD_OUTPUT)
    assert sr.count_reward(pred, GT_SCENE) == pytest.approx(1.0)
    fewer = {"objects": [GT_SCENE["objects"][0]], "relations": []}
    # obj rel-err 1/2=0.5, rel rel-err 1/1=1.0 -> 1 - mean(0.5,1.0)=0.25
    assert sr.count_reward(fewer, GT_SCENE) == pytest.approx(0.25)


def test_spatial_reward_matches_and_labels():
    pred = sr.parse_scene(GOOD_OUTPUT)
    assert sr.spatial_reward(pred, GT_SCENE) == pytest.approx(1.0, abs=1e-6)
    # right boxes, wrong labels -> label-respecting grounding scores 0
    mislabeled = {
        "objects": [
            {"id": 0, "label": "banana", "bbox": [10, 10, 50, 50]},
            {"id": 1, "label": "apple", "bbox": [100, 100, 140, 140]},
        ],
        "relations": [],
    }
    assert sr.spatial_reward(mislabeled, GT_SCENE) == 0.0
    # shifted boxes -> partial
    shifted = {
        "objects": [
            {"id": 0, "label": "red cube", "bbox": [20, 20, 60, 60]},
            {"id": 1, "label": "green sphere", "bbox": [100, 100, 140, 140]},
        ],
        "relations": [],
    }
    s = sr.spatial_reward(shifted, GT_SCENE)
    assert 0.0 < s < 1.0


def test_dense_reward_gating():
    # perfect output -> all components fire: 0.1+0.2+0.5+0.2 = 1.0
    full = sr.dense_reward(GOOD_OUTPUT, GT_SCENE, "(A) left")
    assert full["total"] == pytest.approx(1.0, abs=1e-6)
    assert full["spatial"] == pytest.approx(1.0, abs=1e-6)

    # correct format + scene but WRONG answer -> accuracy 0, spatial gated off
    wrong = sr.dense_reward(GOOD_OUTPUT.replace("(A) left", "(B) right"), GT_SCENE, "(A) left")
    assert wrong["accuracy"] == 0.0
    assert wrong["spatial"] == 0.0  # gated by accuracy
    # total = 0.1*1 + 0.2*1 + 0.5*0 = 0.3
    assert wrong["total"] == pytest.approx(0.3, abs=1e-6)

    # bad format -> everything gated to zero
    broken = sr.dense_reward("<answer>(A) left</answer>", GT_SCENE, "(A) left")
    assert broken["total"] == 0.0 and broken["format"] == 0.0
