"""Tests for vqasynth.visual_credit_audit.

Mirrors the philosophy of ``tests/test_vggt_speedups.py`` and
``tests/test_correspondence.py``: verify the audit mechanics against small fake
objects — no GPU, no real VLM, no download. The next-token scoring path is
exercised through a fake model/processor whose forward returns controlled
logits. Real end-to-end validation against a loaded HuggingFace VLM belongs on a
GPU host.
"""
from __future__ import annotations

import pytest
import torch

# classify_question / extract_yes_no come from the EXISTING scoring module —
# importing them here proves the audit is wired into the eval taxonomy, not a
# self-contained island.
from vqasynth.evaluation import classify_question, extract_yes_no
from vqasynth.visual_credit_audit import (
    aggregate_vca,
    forced_choice_probs,
    make_blank_image,
    run_visual_credit_audit,
    score_item,
    select_comparison_yn,
    yesno_token_ids,
)

# ---------------------------------------------------------------------------
# Fakes for the model + processor surface that forced_choice_probs touches.
# ---------------------------------------------------------------------------

class _Out:
    def __init__(self, logits):
        self.logits = logits


_FAKE_ENCODE = {" yes": [5], "yes": [5], " no": [7], "no": [7]}


class _FakeTokenizer:
    """Encodes " yes" -> [5] and " no" -> [7]."""

    def encode(self, text, add_special_tokens=False):
        return _FAKE_ENCODE.get(text, [0])


class _FakeProcessor:
    """Has apply_chat_template + __call__ + .tokenizer, like a real one."""

    def __init__(self, tokenizer, seq_len=4, vocab=64):
        self.tokenizer = tokenizer
        self._seq_len = seq_len
        self._vocab = vocab

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "PROMPT"

    def __call__(self, text=None, images=None, return_tensors="pt"):
        return {"input_ids": torch.zeros(1, self._seq_len, dtype=torch.long)}


def _fake_model(favor, yes_id=5, no_id=7, vocab=64):
    """Returns logits whose last position favors ``yes`` or ``no``."""

    class _M:
        # NOTE: no `.device` attr on purpose -> inputs.to(...) is skipped, so the
        # fake dict inputs are forwarded unchanged.

        def __call__(self, **kwargs):
            seq = kwargs["input_ids"].shape[1]
            logits = torch.zeros(1, seq, vocab)
            last = logits[0, -1]
            if favor == "yes":
                last[yes_id], last[no_id] = 5.0, 1.0
            else:
                last[no_id], last[yes_id] = 5.0, 1.0
            return _Out(logits)

    return _M()


# ---------------------------------------------------------------------------
# Selection (integration with vqasynth.evaluation.classify_question)
# ---------------------------------------------------------------------------

def test_select_comparison_yn_keeps_only_predicate_yesno():
    items = [
        {"id": "1", "question": "Is the cup on the table?", "answer": "Yes"},   # yn
        {"id": "2", "question": "How far is the car from the sign?", "answer": "3m"},  # distance
        {"id": "3", "question": "Which is taller, the lamp or the chair?", "answer": "lamp"},  # choice
        {"id": "4", "question": "Does the box fit under the shelf?", "answer": "No"},  # yn
    ]
    kept = select_comparison_yn(items)
    assert [k["id"] for k in kept] == ["1", "4"]
    # Cross-check against the existing classifier the selector is built on.
    for k in kept:
        assert classify_question(k["question"]) == "comparison_yn"


# ---------------------------------------------------------------------------
# Controls + token targeting
# ---------------------------------------------------------------------------

def test_make_blank_image_is_uniform_grey():
    img = make_blank_image(size=(8, 8), fill=128)
    assert img.size == (8, 8) and img.mode == "RGB"
    px = img.getpixel((0, 0))
    assert px == (128, 128, 128)
    assert img.getpixel((7, 7)) == px  # uniform


def test_yesno_token_ids_picks_leading_space_token():
    assert yesno_token_ids(_FakeTokenizer()) == (5, 7)


# ---------------------------------------------------------------------------
# Next-token forced-choice scoring (fake forward pass)
# ---------------------------------------------------------------------------

def test_forced_choice_probs_normalize_and_track_logits():
    proc = _FakeProcessor(_FakeTokenizer())
    probs = forced_choice_probs(_fake_model("yes"), proc, "Is it left?", image=None)
    assert set(probs) == {"yes", "no"}
    assert probs["yes"] + probs["no"] == pytest.approx(1.0)
    assert probs["yes"] > probs["no"]  # yes-favoring logits -> P(yes) higher

    probs_no = forced_choice_probs(_fake_model("no"), proc, "Is it left?", image=None)
    assert probs_no["no"] > probs_no["yes"]


def test_score_item_declares_decision_and_flags_text_shortcut():
    proc = _FakeProcessor(_FakeTokenizer())

    class _VLM:
        model = _fake_model("yes")
        processor = proc

    # Same fake ignores the image, so real == text-only control -> zero gain ->
    # decision is NOT credited to the image (the "textual shortcut" case).
    rec = score_item(_VLM, "Is the cat on the mat?", image=None, control_image=None)
    assert rec["decision"] == "yes"
    assert rec["decision_gain"] == pytest.approx(0.0)
    assert rec["image_credited"] is False


# ---------------------------------------------------------------------------
# Aggregation (D-CC + correct-but-uncredited, the paper's headline metrics)
# ---------------------------------------------------------------------------

def _record(decision, gold_label, gold_gain, decision_gain=0.0):
    """Build a minimal scored record accepted by aggregate_vca."""
    correct = (gold_label is not None) and (decision == gold_label)
    return {
        "real": {"yes": 0.0, "no": 0.0}, "control": {"yes": 0.0, "no": 0.0},
        "decision": decision, "decision_gain": decision_gain,
        "image_credited": decision_gain > 0,
        "gold_label": gold_label, "correct": correct, "gold_gain": gold_gain,
        "credited": correct and (gold_gain is not None) and gold_gain > 0,
    }


def test_aggregate_vca_counts_correct_but_uncredited():
    per_item = [
        _record("yes", "yes", gold_gain=+0.2),   # correct AND credited
        _record("yes", "yes", gold_gain=-0.1),   # correct but uncredited
        _record("yes", "yes", gold_gain=0.0),    # correct but uncredited (no gain)
        _record("yes", "no", gold_gain=+0.3),    # wrong decision
        _record("yes", None, gold_gain=None),    # unlabeled (e.g. unparseable gold)
    ]
    report = aggregate_vca(per_item, control="text")
    assert report["n_items"] == 5
    assert report["n_labeled"] == 4
    assert report["overall_accuracy"] == pytest.approx(3 / 4)          # 3 correct of 4 labeled
    assert report["dependence_credited_correctness"] == pytest.approx(1 / 4)  # only the credited one
    assert report["correct_but_uncredited_rate"] == pytest.approx(2 / 3)      # 2 of 3 correct


def test_aggregate_vca_empty_report_is_zeroed():
    report = aggregate_vca([], control="blank")
    assert report["n_items"] == 0
    assert report["overall_accuracy"] == 0.0
    assert report["image_credited_rate"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end through run_visual_credit_audit (fake VLM, no images -> no VLM stack)
# ---------------------------------------------------------------------------

def test_run_visual_credit_audit_text_only_marks_decisions_uncredited():
    proc = _FakeProcessor(_FakeTokenizer())

    class _VLM:
        model = _fake_model("yes")
        processor = proc

    items = [
        {"id": "a", "question": "Is the bike left of the car?", "answer": "Yes.", "images": []},
        {"id": "b", "question": "Is the book on the shelf?", "answer": "No", "images": []},
    ]
    # Sanity: the gold labels parse via the existing extractor the audit reuses.
    assert extract_yes_no("Yes.") is True
    assert extract_yes_no("No") is False

    report = run_visual_credit_audit(_VLM, items, control="text")

    assert report["n_items"] == 2
    assert report["n_labeled"] == 2
    assert report["overall_accuracy"] == pytest.approx(0.5)            # only item "a" matches decision=yes
    # With identical text-only real/control forwards, no decision gains image
    # support -> nothing is credited, so D-CC collapses to 0 and every correct
    # decision is "correct but uncredited" (the shortcut the audit exposes).
    assert report["dependence_credited_correctness"] == 0.0
    assert report["correct_but_uncredited_rate"] == 1.0
    assert report["image_credited_rate"] == 0.0
    assert {r["id"] for r in report["per_item"]} == {"a", "b"}
