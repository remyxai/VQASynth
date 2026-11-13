"""SpatialThinker-style dense reward for scene-graph-grounded spatial reasoning.

Ports the dense verifiable reward from SpatialThinker (Batra, Tu, Chen, Lin, Xie,
Clark — "SpatialThinker: Reinforcing Scene Graph-Grounded Spatial Reasoning via
Dense Rewards", arXiv:2511.07403, CC-BY-4.0) for VQASynth++ Stage-2 RL (GRPO) on
the structured ``<observe>/<scene>/<think>/<answer>`` traces that
:class:`~vqasynth.r1_reasoning.GroundedBBoxTraceReasoner` produces.

The reward has four components combined with lexicographic gating::

    R_total = 1[format==1] * ( w_f*R_format + w_c*R_count + w_a*R_accuracy
                               + 1[accuracy==1] * w_s*R_spatial )

with the paper's weights (format 0.1, count 0.2, accuracy 0.5, spatial 0.2).
``R_spatial`` is the mean Complete-IoU (CIoU) over a Hungarian matching of
predicted-to-ground-truth boxes, computed only once the answer is correct — so
spatial grounding is rewarded only when the reasoning actually lands.

Scene JSON schema (both predicted ``<scene>`` and the GT subgraph)::

    {"objects":   [{"id": 0, "label": "chair", "bbox": [x1, y1, x2, y2]}, ...],
     "relations": [{"subject": 0, "predicate": "left of", "object": 1}, ...]}

bbox is absolute pixel space (x1, y1, x2, y2). Pure-CPU; scipy for the matching.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

# Paper weights (Sec. dense reward).
W_FORMAT = 0.1
W_COUNT = 0.2
W_ACCURACY = 0.5
W_SPATIAL = 0.2

_SCENE_RE = re.compile(r"<scene>(.*?)</scene>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_REQUIRED_TAGS = ("observe", "scene", "think", "answer")


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ciou(a, b):
    """Complete IoU of two (x1, y1, x2, y2) boxes (penalizes center distance and
    aspect-ratio mismatch on top of IoU). Range ~[-1, 1]."""
    iou = _box_iou(a, b)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    center_d2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    ex1, ey1 = min(ax1, bx1), min(ay1, by1)
    ex2, ey2 = max(ax2, bx2), max(ay2, by2)
    diag2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    if diag2 == 0:
        return iou
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    v = alpha = 0.0
    if ah > 0 and bh > 0:
        v = (4.0 / math.pi ** 2) * (math.atan(aw / ah) - math.atan(bw / bh)) ** 2
        denom = (1.0 - iou) + v
        alpha = v / denom if denom > 0 else 0.0
    return iou - center_d2 / diag2 - alpha * v


def parse_scene(text):
    """Extract and parse the ``<scene>`` JSON subgraph. Returns a dict with
    ``objects`` / ``relations`` lists, or None if absent/malformed."""
    if isinstance(text, dict):
        scene = text
    else:
        m = _SCENE_RE.search(text or "")
        if not m:
            return None
        try:
            scene = json.loads(m.group(1).strip())
        except (ValueError, TypeError):
            return None
    if not isinstance(scene, dict):
        return None
    scene.setdefault("objects", [])
    scene.setdefault("relations", [])
    if not isinstance(scene["objects"], list) or not isinstance(scene["relations"], list):
        return None
    return scene


def _valid_object(o):
    if not isinstance(o, dict):
        return False
    box = o.get("bbox")
    return (
        "label" in o
        and isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(c, (int, float)) for c in box)
    )


def format_reward(text):
    """1.0 iff all four tags are present AND the <scene> JSON parses with at least
    one well-formed object (id/label/bbox). Else 0.0. The gate for everything."""
    if not text:
        return 0.0
    low = text.lower()
    if not all(f"<{t}>" in low and f"</{t}>" in low for t in _REQUIRED_TAGS):
        return 0.0
    scene = parse_scene(text)
    if scene is None or not scene["objects"]:
        return 0.0
    return 1.0 if all(_valid_object(o) for o in scene["objects"]) else 0.0


def _norm_answer(s):
    s = (s or "").strip().lower()
    # pull a leading choice letter if present, e.g. "(c) ..." -> "c"
    m = re.match(r"\(?([a-z])\)?[\.\):\s]", s)
    return m.group(1) if m else re.sub(r"\s+", " ", s)


def accuracy_reward(text, gt_answer):
    """Binary exact-match on the <answer> content (choice-letter aware)."""
    m = _ANSWER_RE.search(text or "")
    if not m:
        return 0.0
    return 1.0 if _norm_answer(m.group(1)) == _norm_answer(gt_answer) else 0.0


def count_reward(pred_scene, gt_scene):
    """1 - mean relative error over object and relation counts, clamped [0, 1]."""
    if pred_scene is None or gt_scene is None:
        return 0.0
    errs = []
    for key in ("objects", "relations"):
        n_gt = len(gt_scene.get(key, []) or [])
        n_pred = len(pred_scene.get(key, []) or [])
        errs.append(abs(n_pred - n_gt) / max(n_gt, 1))
    return float(max(0.0, 1.0 - float(np.mean(errs))))


def spatial_reward(pred_scene, gt_scene):
    """Mean CIoU over a Hungarian matching of predicted to GT boxes.

    Objects are matched by maximizing total IoU (cost = 1 - IoU); a match only
    contributes its CIoU when the labels agree (label-respecting grounding).
    Returns a value clamped to [0, 1] (0 when either side has no boxes).
    """
    if pred_scene is None or gt_scene is None:
        return 0.0
    pred = [o for o in pred_scene.get("objects", []) if _valid_object(o)]
    gt = [o for o in gt_scene.get("objects", []) if _valid_object(o)]
    if not pred or not gt:
        return 0.0
    cost = np.ones((len(pred), len(gt)), dtype=float)
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            cost[i, j] = 1.0 - _box_iou(p["bbox"], g["bbox"])
    rows, cols = linear_sum_assignment(cost)
    scores = []
    for i, j in zip(rows, cols):
        if str(pred[i].get("label", "")).strip().lower() == str(gt[j].get("label", "")).strip().lower():
            scores.append(max(0.0, ciou(pred[i]["bbox"], gt[j]["bbox"])))
        else:
            scores.append(0.0)
    # normalize over GT objects: unmatched GT boxes count as 0 (recall-sensitive)
    return float(np.clip(sum(scores) / len(gt), 0.0, 1.0))


def dense_reward(text, gt_scene, gt_answer, weights=None):
    """SpatialThinker dense reward with lexicographic gating.

    Returns a dict with the total and each component so the RL loop can log the
    breakdown. ``gt_scene`` is the GT subgraph dict; ``gt_answer`` the GT answer.
    """
    w = {"format": W_FORMAT, "count": W_COUNT, "accuracy": W_ACCURACY, "spatial": W_SPATIAL}
    if weights:
        w.update(weights)

    r_format = format_reward(text)
    if r_format < 1.0:
        return {"total": 0.0, "format": 0.0, "count": 0.0, "accuracy": 0.0, "spatial": 0.0}

    pred_scene = parse_scene(text)
    r_count = count_reward(pred_scene, gt_scene)
    r_acc = accuracy_reward(text, gt_answer)
    r_spatial = spatial_reward(pred_scene, gt_scene) if r_acc >= 1.0 else 0.0

    total = (
        w["format"] * r_format
        + w["count"] * r_count
        + w["accuracy"] * r_acc
        + (w["spatial"] * r_spatial if r_acc >= 1.0 else 0.0)
    )
    return {
        "total": float(total),
        "format": float(r_format),
        "count": float(r_count),
        "accuracy": float(r_acc),
        "spatial": float(r_spatial),
    }


def grounded_state_to_scene(grounded_state):
    """Build a GT scene graph ``{"objects", "relations"}`` from a prompt-stage
    ``grounded_state`` record (dict or JSON string) so :func:`dense_reward` has a
    ground truth at RL time.

    Only objects carrying a 2D ``bbox`` become GT objects (the spatial component
    needs boxes); each grounded pair becomes a relation mapping its ``a``/``b``
    labels to the emitted object ids (a coarse ``"related"`` predicate — enough to
    drive the count reward; richer spatial predicates can replace it later).
    Returns None for a malformed record.
    """
    gs = grounded_state
    if isinstance(gs, str):
        try:
            gs = json.loads(gs)
        except (ValueError, TypeError):
            return None
    if not isinstance(gs, dict):
        return None

    objects, label_to_id = [], {}
    for o in gs.get("objects", []) or []:
        box = o.get("bbox") if isinstance(o, dict) else None
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        oid = len(objects)
        label = o.get("label", "object")
        objects.append({"id": oid, "label": label, "bbox": [float(c) for c in box]})
        label_to_id.setdefault(label, oid)

    relations = []
    for p in gs.get("pairs", []) or []:
        if not isinstance(p, dict):
            continue
        a, b = label_to_id.get(p.get("a")), label_to_id.get(p.get("b"))
        if a is not None and b is not None:
            relations.append({"subject": a, "predicate": "related", "object": b})

    return {"objects": objects, "relations": relations}
