"""Phase-4 data-validation gate for VQASynth++ grounded / soft reasoning traces.

Cheap, pre-training quality checks on the traces the ``r1_reasoning`` stage emits,
so low-quality rows are filtered BEFORE training compute is spent — mirroring the
auditing spirit of :mod:`vqasynth.visual_credit`. Four checks:

  * **grounding-correctness** — the trace's ``<scene>`` cites objects that
    actually exist in the row's ``grounded_state``, with boxes inside the image
    bounds (penalizes hallucinated objects / out-of-frame boxes);
  * **trace-faithfulness** — the ``<answer>`` is consistent with the QA answer,
    the reasoning references the grounded state, and low-confidence measurements
    are hedged rather than asserted (the soft-composition contract);
  * **chance / difficulty calibration** (dataset-level) — the answer-choice
    distribution is not degenerate (a model shouldn't ace the set by always
    guessing one choice);
  * **dedup** (dataset-level) — near-duplicate question/answer rows are dropped.

Pure-CPU; reuses :mod:`vqasynth.spatial_reward` parsing. No model, no GPU. Row
drops are always counted and returned (never silent truncation).
"""
from __future__ import annotations

import json
import re
from collections import Counter

from vqasynth.spatial_reward import accuracy_reward, parse_scene

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_HEDGE_WORDS = (
    "rough", "roughly", "approx", "uncertain", "not sure", "unsure", "low confidence",
    "hard to tell", "about ", "around ", "estimate", "maybe", "might", "i think",
)
LOW_CONF = 0.5  # depth_conf below this is "low confidence" and should be hedged


def _coerce_state(grounded_state):
    gs = grounded_state
    if isinstance(gs, str):
        try:
            gs = json.loads(gs)
        except (ValueError, TypeError):
            return None
    return gs if isinstance(gs, dict) else None


def _grounded_labels(gs):
    return {
        str(o.get("label", "")).strip().lower()
        for o in (gs.get("objects", []) or [])
        if isinstance(o, dict) and o.get("label")
    }


def grounding_correctness(output_text, grounded_state, image_size=None):
    """Fraction of the predicted scene that is grounded: scene object labels that
    appear in grounded_state, and boxes within the image bounds.

    Returns ``{"score", "verifiable", "issues"}``. ``verifiable`` is False when
    the row carries no grounded reference (nothing to check against) — such rows
    are not penalized (score 1.0) but are flagged.
    """
    issues = []
    scene = parse_scene(output_text)
    if scene is None:
        return {"score": 0.0, "verifiable": True, "issues": ["no_scene"]}

    gs = _coerce_state(grounded_state)
    labels = _grounded_labels(gs) if gs else set()
    scene_objs = [o for o in scene.get("objects", []) if isinstance(o, dict)]
    if not scene_objs:
        return {"score": 0.0, "verifiable": True, "issues": ["empty_scene"]}

    # label precision against the grounded reference (hallucination check)
    if labels:
        grounded = [str(o.get("label", "")).strip().lower() in labels for o in scene_objs]
        label_prec = sum(grounded) / len(scene_objs)
        if label_prec < 1.0:
            issues.append("hallucinated_objects")
    else:
        label_prec = None  # unverifiable

    # boxes within image bounds
    box_ok = None
    if image_size and len(image_size) == 2:
        w, h = float(image_size[0]), float(image_size[1])
        oks = []
        for o in scene_objs:
            b = o.get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                x1, y1, x2, y2 = (float(c) for c in b)
                oks.append(0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h)
        if oks:
            box_ok = sum(oks) / len(oks)
            if box_ok < 1.0:
                issues.append("boxes_out_of_bounds")

    parts = [p for p in (label_prec, box_ok) if p is not None]
    if not parts:
        return {"score": 1.0, "verifiable": False, "issues": ["no_grounded_reference"]}
    return {"score": float(sum(parts) / len(parts)), "verifiable": True, "issues": issues}


def _confidence_hedged(output_text, gs):
    """If the grounded state has a low-confidence object, the <think> text should
    hedge somewhere. Returns (applicable, hedged)."""
    if not gs:
        return False, True
    has_lowconf = any(
        isinstance(o, dict) and isinstance(o.get("depth_conf"), (int, float)) and o["depth_conf"] < LOW_CONF
        for o in (gs.get("objects", []) or [])
    )
    if not has_lowconf:
        return False, True
    m = _THINK_RE.search(output_text or "")
    think = (m.group(1) if m else (output_text or "")).lower()
    return True, any(w in think for w in _HEDGE_WORDS)


def trace_faithfulness(output_text, gt_answer, grounded_state):
    """Answer consistency + grounded-state reference + confidence hedging.

    Returns ``{"score", "issues"}`` averaging the applicable binary checks.
    """
    issues = []
    checks = []

    acc = accuracy_reward(output_text, gt_answer)
    checks.append(acc)
    if acc < 1.0:
        issues.append("answer_mismatch")

    gs = _coerce_state(grounded_state)
    labels = _grounded_labels(gs) if gs else set()
    if labels:
        low = (output_text or "").lower()
        referenced = any(lbl and lbl in low for lbl in labels)
        checks.append(1.0 if referenced else 0.0)
        if not referenced:
            issues.append("no_grounded_reference_in_trace")

    applicable, hedged = _confidence_hedged(output_text, gs)
    if applicable:
        checks.append(1.0 if hedged else 0.0)
        if not hedged:
            issues.append("unhedged_low_confidence")

    score = float(sum(checks) / len(checks)) if checks else 1.0
    return {"score": score, "issues": issues}


def validate_trace(output_text, gt_answer, grounded_state, image_size=None,
                   grounding_min=0.99, faithfulness_min=0.99):
    """Per-row gate. Returns ``{"passed", "grounding", "faithfulness", "reasons"}``."""
    g = grounding_correctness(output_text, grounded_state, image_size=image_size)
    f = trace_faithfulness(output_text, gt_answer, grounded_state)
    passed = g["score"] >= grounding_min and f["score"] >= faithfulness_min
    return {
        "passed": bool(passed),
        "grounding": g,
        "faithfulness": f,
        "reasons": g["issues"] + f["issues"],
    }


# --- dataset-level ---------------------------------------------------------

def answer_distribution(answers):
    """Counter over normalized choice answers (choice letter when present)."""
    norm = []
    for a in answers:
        s = str(a or "").strip().lower()
        m = re.match(r"\(?([a-z])\)?[\.\):\s]", s)
        norm.append(m.group(1) if m else s)
    return Counter(norm)


def chance_calibration(answers, degenerate_frac=0.9):
    """Flag a degenerate answer distribution (one choice dominates), which a model
    could exploit by always guessing. Returns ``{"degenerate", "majority_frac",
    "distribution"}``."""
    dist = answer_distribution(answers)
    n = sum(dist.values())
    majority = max(dist.values()) / n if n else 0.0
    return {
        "degenerate": bool(majority >= degenerate_frac),
        "majority_frac": float(majority),
        "distribution": dict(dist),
    }


def _dedup_key(question, answer):
    q = re.sub(r"\s+", " ", str(question or "").strip().lower())
    a = re.sub(r"\s+", " ", str(answer or "").strip().lower())
    return f"{q}||{a}"


def dedup(rows, question_of, answer_of):
    """Drop near-duplicate rows by normalized (question, answer). ``question_of`` /
    ``answer_of`` extract those fields from a row. Returns (kept, n_dropped)."""
    seen, kept, dropped = set(), [], 0
    for row in rows:
        key = _dedup_key(question_of(row), answer_of(row))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, dropped


def filter_dataset(rows, output_of, answer_of, grounded_of, question_of,
                   image_size_of=None, grounding_min=0.99, faithfulness_min=0.99,
                   degenerate_frac=0.9):
    """Apply the per-row gate + dedup to a list of trace rows.

    The ``*_of`` callables extract fields from a row. Returns a report dict with
    ``kept`` rows, drop counts by reason, and the chance-calibration summary. Drops
    are always counted (never silent).
    """
    passed, dropped_quality, reasons = [], 0, Counter()
    for row in rows:
        v = validate_trace(
            output_of(row), answer_of(row), grounded_of(row),
            image_size=image_size_of(row) if image_size_of else None,
            grounding_min=grounding_min, faithfulness_min=faithfulness_min,
        )
        if v["passed"]:
            passed.append(row)
        else:
            dropped_quality += 1
            reasons.update(v["reasons"] or ["unspecified"])

    kept, dropped_dup = dedup(passed, question_of, answer_of)
    calib = chance_calibration([answer_of(r) for r in kept], degenerate_frac=degenerate_frac)
    return {
        "kept": kept,
        "n_in": len(rows),
        "n_kept": len(kept),
        "n_dropped_quality": dropped_quality,
        "n_dropped_dup": dropped_dup,
        "drop_reasons": dict(reasons),
        "chance_calibration": calib,
    }
