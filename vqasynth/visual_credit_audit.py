"""
Visual Credit Audit (VCA) for spatial yes/no questions.

Adapted from "Visual Credit Audit for Multimodal Spatial Reasoning"
(arXiv:2607.27069). VCA complements the eval stage's correctness scorer
(``vqasynth.benchmarks``) by asking whether a VLM's yes/no spatial decision is
actually *credited to the image* or merely recoverable from a text-only (or
blank) control.

This module delivers the paper's two headline pieces against null controls:

  * **First audit (training- and label-free).** For a forced yes/no question,
    read the model's next-token probability for "yes"/"no" under the real
    benchmark image and under a control (text-only, or a blank uniform image).
    The real image *credits* the decision when the declared answer gets more
    probability under the real image than under the control. Needs no labels and
    no answer flip. The next-token scoring reuses the already-loaded HuggingFace
    VLM (``vqasynth.inference.VLMInference``) — one forward pass per condition.

  * **Dependence-credited correctness (D-CC).** Applying the gold label: on a
    correct item, the image must give the *gold* answer positive probability
    gain over the control for the decision to count as credited. A correct item
    with non-positive gold gain is "correct but uncredited" — the paper's
    12.73-26.25% finding of decisions the model gets right without genuinely
    using the image.

Intentionally out of scope (they need bespoke infra the eval stage does not
host): the fixed-pixel relation contrasts, the 3x3 evidence-source factorial,
the prediction-alignment extension to errors, and the 108-edit natural-image
correspondence check. Those are downstream-audit territory; this module ships
the core image-credit decomposition against null controls.
"""

from __future__ import annotations

import torch

from vqasynth.evaluation import classify_question, extract_yes_no

# Forces a well-posed yes/no next token under the fixed forced-choice interface.
FORCED_CHOICE_SUFFIX = "\nAnswer with yes or no."


def make_blank_image(size=(224, 224), fill=128):
    """Uniform grey PIL image used as the *blank* null control."""
    from PIL import Image

    return Image.new("RGB", size, (fill, fill, fill))


def select_comparison_yn(items):
    """Keep the ``comparison_yn`` subset via the shared question classifier.

    Mirrors the eval stage's scoring taxonomy: only yes/no predicate questions
    ("Is the X ...?", "Does the X ...?", "Can you confirm ...?") are items a
    yes/no forced choice is well-posed for. ``classify_question`` also tags
    SpatialScore ``judgment`` items (which are predicate-shaped) as
    ``comparison_yn``, so this is the single canonical selector.
    """
    return [
        it for it in items
        if classify_question(it.get("question", "")) == "comparison_yn"
    ]


def yesno_token_ids(tokenizer):
    """First sub-token ids for " yes" / " no" — the forced-choice targets.

    Tries a leading space first (the token that naturally follows the prompt),
    then a bare token as a fallback.
    """
    yes = tokenizer.encode(" yes", add_special_tokens=False) or tokenizer.encode(
        "yes", add_special_tokens=False
    )
    no = tokenizer.encode(" no", add_special_tokens=False) or tokenizer.encode(
        "no", add_special_tokens=False
    )
    if not yes or not no:
        raise ValueError(
            "Tokenizer produced no tokens for ' yes'/' no'; cannot force-choice score."
        )
    return yes[0], no[0]


def forced_choice_probs(model, processor, question, image=None):
    """Forced-choice ``P(yes)`` / ``P(no)`` over the next token after the prompt.

    One forward pass on the already-loaded VLM. ``image=None`` is the text-only
    control (no image content block in the prompt); passing a PIL image scores
    that condition (use ``make_blank_image`` for the blank control). Returns a
    dict ``{"yes": p_yes, "no": p_no}`` with ``p_yes + p_no == 1`` (softmax over
    just the two option logits, matching the paper's forced-choice interface).
    """
    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]

    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        proc_kwargs = {"text": prompt, "return_tensors": "pt"}
        if image is not None:
            proc_kwargs["images"] = [image]
        inputs = processor(**proc_kwargs)
    else:
        proc_kwargs = {"text": question, "return_tensors": "pt"}
        if image is not None:
            proc_kwargs["images"] = image
        inputs = processor(**proc_kwargs)

    target = getattr(model, "device", None)
    if target is not None and hasattr(inputs, "to"):
        inputs = inputs.to(target)

    with torch.no_grad():
        out = model(**inputs)

    last = out.logits[0, -1, :].float()
    yes_id, no_id = yesno_token_ids(processor.tokenizer)
    pair = torch.stack([last[yes_id], last[no_id]])
    probs = torch.softmax(pair, dim=0)
    return {"yes": float(probs[0]), "no": float(probs[1])}


def score_item(vlm, question, image, control_image=None):
    """Score one item under the real image and a single null control.

    ``control_image=None`` selects the text-only control; pass
    ``make_blank_image()`` for the blank control. Returns per-condition
    probabilities plus the declared decision, its image-credit flag, and the
    decision's probability gain over the control (the label-free first audit).
    """
    q = question + FORCED_CHOICE_SUFFIX
    real = forced_choice_probs(vlm.model, vlm.processor, q, image=image)
    ctrl = forced_choice_probs(vlm.model, vlm.processor, q, image=control_image)

    decision = "yes" if real["yes"] >= real["no"] else "no"
    decision_gain = real[decision] - ctrl[decision]
    return {
        "real": real,
        "control": ctrl,
        "decision": decision,
        "decision_gain": decision_gain,
        "image_credited": decision_gain > 0,
    }


def aggregate_vca(per_item, control="text"):
    """Roll per-item records into the VCA report metrics.

    ``per_item`` is a list of dicts as produced by :func:`run_visual_credit_audit`
    (each carrying ``decision``, ``image_credited``, ``decision_gain``, and —
    when a gold label was extractable — ``gold_label`` / ``correct`` /
    ``gold_gain``). Returns the aggregate report dict.
    """
    scored = [r for r in per_item if r.get("real") is not None]
    n = len(scored)
    if n == 0:
        zero = 0.0
        return {
            "control": control, "n_items": 0, "n_labeled": 0,
            "overall_accuracy": zero, "dependence_credited_correctness": zero,
            "correct_but_uncredited_rate": zero, "image_credited_rate": zero,
            "mean_decision_gain": zero, "per_item": per_item,
        }

    labeled = [r for r in scored if r.get("gold_label") is not None]
    correct = [r for r in labeled if r["correct"]]
    dcc = [r for r in correct if r["gold_gain"] > 0]
    correct_uncredited = [r for r in correct if r["gold_gain"] <= 0]
    image_credited = [r for r in scored if r["image_credited"]]

    return {
        "control": control,
        "n_items": n,
        "n_labeled": len(labeled),
        "overall_accuracy": (len(correct) / len(labeled)) if labeled else 0.0,
        "dependence_credited_correctness": (len(dcc) / len(labeled)) if labeled else 0.0,
        "correct_but_uncredited_rate": (len(correct_uncredited) / len(correct)) if correct else 0.0,
        "image_credited_rate": len(image_credited) / n,
        "mean_decision_gain": sum(r["decision_gain"] for r in scored) / n,
        "per_item": per_item,
    }


def run_visual_credit_audit(vlm, items, control="text"):
    """Run the Visual Credit Audit over ``comparison_yn``-style items.

    Args:
        vlm: a ``vqasynth.inference.VLMInference`` instance — the already-loaded
            HuggingFace VLM whose ``.model`` / ``.processor`` are reused for the
            next-token scoring (no second model load).
        items: normalized benchmark items (``question`` / ``answer`` / ``images``).
            Pre-filter with :func:`select_comparison_yn` for the yes/no subset.
        control: ``"text"`` (text-only, default) or ``"blank"`` (uniform image).

    Returns the aggregate VCA report (see :func:`aggregate_vca`).
    """
    control_image = make_blank_image() if control == "blank" else None

    per_item = []
    for item in items:
        images = item.get("images") or []
        image = None
        if images:
            # Lazy: imported only when an image is actually present, so the
            # text-only / no-image audit path stays free of the VLM stack.
            from vqasynth.inference import _to_pil

            image = _to_pil(images[0])

        rec = score_item(vlm, item.get("question", ""), image, control_image=control_image)
        rec["id"] = item.get("id")

        gold = extract_yes_no(item.get("answer", ""))
        if gold is None:
            rec["gold_label"] = None
            rec["correct"] = False
            rec["gold_gain"] = None
            rec["credited"] = False
        else:
            gold_label = "yes" if gold else "no"
            rec["gold_label"] = gold_label
            rec["correct"] = rec["decision"] == gold_label
            rec["gold_gain"] = rec["real"][gold_label] - rec["control"][gold_label]
            rec["credited"] = rec["correct"] and rec["gold_gain"] > 0

        per_item.append(rec)

    return aggregate_vca(per_item, control=control)


def format_vca_report(report):
    """Human-readable summary of a VCA report (mirrors ``format_benchmark_report``)."""
    lines = [
        "", "=" * 70,
        "VISUAL CREDIT AUDIT (comparison_yn)", "=" * 70,
        f"control: {report['control']}    items: {report['n_items']} (labeled: {report['n_labeled']})",
        f"  overall accuracy                   : {report['overall_accuracy']:.1%}",
        f"  dependence-credited correct (D-CC) : {report['dependence_credited_correctness']:.1%}",
        f"  correct but uncredited             : {report['correct_but_uncredited_rate']:.1%} of correct",
        f"  image-credited decisions (label-free): {report['image_credited_rate']:.1%}",
        f"  mean decision support gain         : {report['mean_decision_gain']:+.4f}",
        "=" * 70, "",
    ]
    return "\n".join(lines)
