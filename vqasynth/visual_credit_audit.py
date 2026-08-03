"""
Visual Credit Audit for forced-choice spatial reasoning benchmarks.

Adapted from "Visual Credit Audit for Multimodal Spatial Reasoning"
(arXiv:2607.27069). That work shows that closed yes/no (and other
forced-choice) spatial benchmarks reward correct answers even when the image
adds no support beyond a text-only / blank control: a sizable fraction of
correct decisions are "correct but uncredited" -- the model would have answered
identically from the question alone. Dependence-credited correctness (D-CC)
re-scores a correct decision as credited only when the image actually moved
the model toward the gold answer relative to that control.

This module implements D-CC for the repo's forced-choice items (yes/no
*judgment* items and letter *multi-choice* items), reusing the existing
extract_yes_no / extract_option primitives. The blank-image control that VCA
compares against is produced through the existing VLMInference.predict path
(see BenchmarkRunner.audit_visual_credit), so no new inference machinery is
required.

Implementation mode (Mode 2 -- adapted port): the paper's core D-CC mechanism
is kept at full fidelity (image decision vs. blank-control decision; a correct
item is credited iff the control does not already yield gold). The paper's
bespoke control construction, image-permutation null check, fixed-pixel
relation contrasts, and 3x3 evidence-source factorial are intentionally out of
scope here: they are diagnostics layered on top of the primitive, not the
scoring primitive this repo's eval stage calls.
"""

from __future__ import annotations

from PIL import Image

from vqasynth.evaluation import extract_option, extract_yes_no


def blank_control_image(size=(224, 224), fill=128):
    """
    Neutral solid-color image used as the no-image control.

    HuggingFace vision-language models require an image token, so a uniform
    blank image is the practical equivalent of the paper's "blank control" and
    lets the existing VLMInference.predict run unchanged for the control
    variant.

    Returns a PIL.Image.
    """
    return Image.new("RGB", size, (fill, fill, fill))


def extract_decision(text, kind):
    """
    Extract a forced-choice decision from free text.

    kind == "judgment"      -> bool (yes/no) via extract_yes_no
    kind == "multi-choice"  -> option letter ("A".."F") via extract_option

    Returns the decision, or None when one cannot be extracted.
    """
    if kind == "judgment":
        return extract_yes_no(text)
    return extract_option(text)


def credit_decision(image_pred, control_pred, gold, kind):
    """
    Dependence-credited correctness (D-CC) for a single forced-choice item.

    Faithful to the paper: on a correct item the image is *credited* iff the
    blank/text-only control does not already yield the gold decision (i.e. the
    image supplied gold-aligned positive support beyond the control).

    Returns 1.0 (correct AND credited), 0.0 (correct-but-uncredited, or
    incorrect), or None (a decision could not be extracted -> skip the item).
    """
    image_d = extract_decision(image_pred, kind)
    control_d = extract_decision(control_pred, kind)
    gold_d = extract_decision(gold, kind)

    if image_d is None or control_d is None or gold_d is None:
        return None

    correct = image_d == gold_d
    credited = correct and (control_d != gold_d)
    return 1.0 if credited else 0.0


def audit_decisions(records):
    """
    Aggregate D-CC across forced-choice control records.

    Each record is a dict with keys "image_pred", "control_pred", "gold", and
    "kind" ("judgment" or "multi-choice"). Items whose decisions cannot be
    extracted are skipped. Returns:

        n                       -> number of scored items
        accuracy                -> fraction correct
        d_cc                    -> fraction dependence-credited correct
        correct_uncredited_rate -> of the correct items, the fraction the
                                   model would have reached from the control
                                   alone (the paper's headline 12-26% number)
    """
    n = 0
    correct = 0
    credited = 0
    correct_uncredited = 0

    for rec in records:
        kind = rec.get("kind", "judgment")
        image_d = extract_decision(rec["image_pred"], kind)
        control_d = extract_decision(rec["control_pred"], kind)
        gold_d = extract_decision(rec["gold"], kind)
        if image_d is None or control_d is None or gold_d is None:
            continue

        n += 1
        if image_d == gold_d:
            correct += 1
            if control_d == gold_d:
                correct_uncredited += 1
        if image_d == gold_d and control_d != gold_d:
            credited += 1

    if n == 0:
        return {"n": 0, "accuracy": 0.0, "d_cc": 0.0, "correct_uncredited_rate": 0.0}

    return {
        "n": n,
        "accuracy": correct / n,
        "d_cc": credited / n,
        "correct_uncredited_rate": (correct_uncredited / correct) if correct else 0.0,
    }
