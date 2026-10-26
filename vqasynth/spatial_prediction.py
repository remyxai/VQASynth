"""
Predictive spatial reasoning benchmark support (SpaceCast-Bench).

Adapted from "SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in
Vision-Language Models" (arXiv:2610.12402). The benchmarks already registered
in :mod:`vqasynth.benchmarks` score relations that can be read off the input
image. SpaceCast-Bench instead uses an *observe-transform-infer* shape:
several observation views of a real scene, a transformation premise stated in
the question, and an answer about the unobserved outcome. This module adds
that benchmark to the evaluation stage as a loader + scorer pair:

  * :func:`load_spacecast` — normalizes the HuggingFace dataset
    (``hongxingli/SpaceCast-Bench``) into the common item schema consumed by
    ``BenchmarkRunner``, mapping the paper's three-level taxonomy (L1 static
    perception / L2 local prediction / L3 global prediction) onto ``category``
    so ``by_category`` reads as the paper's diagnostic ladder.
  * :func:`score_spacecast_item` — option-letter accuracy via
    :func:`vqasynth.evaluation.extract_option`, with an option-text fallback
    and exact-set scoring for the benchmark's multi-select questions.
  * ``include_bridge_views=False`` reproduces the paper's bridge-view
    ablation: the intermediate views connecting two disjoint observations are
    dropped while question, options and scoring stay fixed.

This is an ADAPTED PORT (Mode 2). Substituted: the paper's own inference
harness (custom API endpoints and per-model decoding settings) is replaced by
the repo's ``BenchmarkRunner`` + ``vqasynth.evaluation`` scoring primitives.
Out of scope: the paper's fine-tuning recipe and its controlled
3D-evidence-vs-generated-outcome-image comparison.
"""

from __future__ import annotations

import os
import re

from vqasynth.evaluation import extract_option, score_distance_mra

SPACECAST_REPO_ID = "hongxingli/SpaceCast-Bench"

# The paper's three capability levels, in diagnostic order. Prefixed with the
# raw level code so sorted category reports keep the L1 -> L3 ladder order.
SPACECAST_LEVELS = {
    "L1": "L1 static perception",
    "L2": "L2 local prediction",
    "L3": "L3 global prediction",
}

BRIDGE_ROLE = "bridge"

_STANDALONE_LETTER_RE = re.compile(r"\b([A-F])\b")
_ANSWER_ANCHOR_RE = re.compile(r"answer(?:\s+is)?\s*[:\-]?\s*(.+)$", re.IGNORECASE)
_SINGLE_LETTER_RE = re.compile(r"^[A-F]$")


def extract_option_letters(text):
    """
    Extract every option letter (A-F) a free-text answer commits to.

    Multi-select answers name several options ("A and C", "B, D"), so the
    single-letter cascade in :func:`vqasynth.evaluation.extract_option` is not
    enough. If an "answer is ..." / "Answer: ..." span is present only that
    span is read, otherwise the whole text. Order is preserved, duplicates
    dropped.
    """
    text = (text or "").strip()
    if not text:
        return []
    anchor = _ANSWER_ANCHOR_RE.search(text)
    segment = anchor.group(1) if anchor else text
    letters = []
    for letter in _STANDALONE_LETTER_RE.findall(segment.upper()):
        if letter not in letters:
            letters.append(letter)
    return letters


def _split_answers(raw_answers, label_by_option):
    """Split a raw answer list into option letters and free-text values.

    Answers are normally option letters, but task types that phrase the answer
    as text ("back-right") are mapped back to their label when the text matches
    an option.
    """
    letters, values = [], []
    for raw in raw_answers:
        answer = str(raw).strip()
        if not answer:
            continue
        if _SINGLE_LETTER_RE.match(answer.upper()):
            letters.append(answer.upper())
        elif answer.lower() in label_by_option:
            letters.append(label_by_option[answer.lower()])
        else:
            values.append(answer)
    return letters, values


def _is_bridge_view(index, roles, image_paths, bridge_names):
    """Is the image at ``index`` one of the bridging observation views?"""
    role = roles[index].lower() if index < len(roles) else ""
    if role:
        return role.startswith(BRIDGE_ROLE)
    if bridge_names and index < len(image_paths):
        return os.path.basename(image_paths[index]) in bridge_names
    return False


def _select_images(entry, include_bridge_views):
    """Return the (images, image_roles) pair shown to the model.

    Bridge views are the paper's mechanism for integrating two disjoint
    observations of a scene; dropping them (``include_bridge_views=False``)
    reproduces its bridge-view ablation with everything else held fixed.
    """
    images = list(entry.get("images") or [])
    roles = [str(r) for r in entry.get("image_roles") or []]
    if include_bridge_views:
        return images, roles

    bridge_names = {os.path.basename(str(p)) for p in entry.get("bridge_view_names") or []}
    image_paths = [str(p) for p in entry.get("image_paths") or []]
    kept_images, kept_roles = [], []
    for index, image in enumerate(images):
        if _is_bridge_view(index, roles, image_paths, bridge_names):
            continue
        kept_images.append(image)
        kept_roles.append(roles[index] if index < len(roles) else "")
    return kept_images, kept_roles


def normalize_spacecast_entry(entry, include_bridge_views=True):
    """
    Normalize one SpaceCast-Bench row into the common benchmark item schema.

    ``category`` carries the paper's level and ``subcategory`` its task type,
    so ``BenchmarkRunner`` aggregations report the paper's diagnostic breakdown
    without any extra plumbing.
    """
    options = [str(o) for o in entry.get("options") or []]
    option_labels = [str(l).upper() for l in entry.get("option_labels") or []]
    if not option_labels:
        option_labels = [chr(65 + i) for i in range(len(options))]
    label_by_option = {
        opt.strip().lower(): label
        for opt, label in zip(options, option_labels)
        if label
    }

    letters, values = _split_answers(entry.get("answer") or [], label_by_option)
    correct_values = values + [
        str(v).strip() for v in entry.get("correct_values") or [] if str(v).strip()
    ]

    question = str(entry.get("question", ""))
    level_key = str(entry.get("level", "")).strip().upper()
    images, image_roles = _select_images(entry, include_bridge_views)
    item_id = str(entry.get("question_uid") or "").strip()
    if not item_id:
        # question_uid is always present in the released data; this only keeps
        # locally exported rows scorable. sum(ord) is deterministic per process.
        item_id = f"{entry.get('scene_id', 'scene')}#{sum(map(ord, question))}"

    return {
        "id": item_id,
        "question": question,
        "answer": ", ".join(letters),
        "answer_text": ", ".join(correct_values),
        "question_type": "multi-choice" if options else "open-ended",
        "category": SPACECAST_LEVELS.get(level_key, level_key or "unknown"),
        "subcategory": str(
            entry.get("type") or entry.get("task") or entry.get("source_type") or "unknown"
        ),
        "options": options,
        "images": images,
        "source": "SpaceCast-Bench",
        # SpaceCast-specific fields the scorer and downstream tooling need
        "answers": letters,
        "correct_values": correct_values,
        "multi_select": bool(entry.get("multi_select")),
        "level": level_key,
        "scene_id": str(entry.get("scene_id", "")),
        "image_roles": image_roles,
    }


def _fetch_spacecast(split, streaming):
    """
    Yield raw SpaceCast-Bench rows from the HuggingFace Hub.

    ``datasets`` is imported lazily so this module (and the registry that
    imports it) stays importable in minimal environments, mirroring the lazy
    open3d/numpy imports in :mod:`vqasynth.detection_3d`.
    """
    from datasets import load_dataset

    return load_dataset(SPACECAST_REPO_ID, split=split, streaming=streaming)


def load_spacecast(
    split="test", include_bridge_views=True, streaming=False, max_items=None
):
    """
    Load SpaceCast-Bench from HuggingFace (hongxingli/SpaceCast-Bench).

    3,862 questions over 182 real-world scenes; the ``test`` split is the only
    one published.

    Args:
        split: Dataset split to read.
        include_bridge_views: Keep the bridging observation views. Set False to
            reproduce the paper's bridge-view ablation.
        streaming: Read rows lazily from the Hub.
        max_items: Stop after this many items (smoke runs).

    Returns list of normalized items.
    """
    items = []
    for entry in _fetch_spacecast(split, streaming):
        items.append(normalize_spacecast_entry(entry, include_bridge_views))
        if max_items is not None and len(items) >= max_items:
            break
    return items


def _score_multi_select(item, prediction):
    """Exact-set accuracy over the letters a multi-select answer commits to."""
    gold = {letter for letter in item.get("answers") or []}
    predicted = set(extract_option_letters(prediction))
    if not gold:
        return None
    if not predicted:
        return 0.0
    return 1.0 if predicted == gold else 0.0


def score_spacecast_item(item, prediction):
    """
    Score one SpaceCast-Bench item, returning a value in [0, 1].

    Single-select questions score an option-letter match (falling back to
    option-text, then to the shared numeric primitives for open-ended
    quantitative answers); multi-select questions require the exact set of
    letters. Returns None when nothing in the item can be scored, which lets
    ``BenchmarkRunner`` fall back to its LLM judge.
    """
    prediction = prediction or ""

    if item.get("multi_select"):
        score = _score_multi_select(item, prediction)
        if score is not None:
            return score

    gold_letters = item.get("answers") or []
    if gold_letters:
        predicted = extract_option(prediction)
        if predicted:
            return 1.0 if predicted == gold_letters[0] else 0.0

    # Option-text fallback: models that answer "back-right" instead of "C".
    gold_text = (item.get("answer_text") or "").strip().lower()
    if gold_text:
        lowered = prediction.strip().lower()
        if any(value and value in lowered for value in gold_text.split(", ")):
            return 1.0

    # Open-ended quantitative answers share the numeric scorers.
    if any(char.isdigit() for char in gold_text):
        mra = score_distance_mra(prediction, gold_text)
        if mra is not None:
            return mra
        return 1.0 if prediction.strip().lower() == gold_text else 0.0

    if gold_letters or gold_text:
        return 0.0
    return None
