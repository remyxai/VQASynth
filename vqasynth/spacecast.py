"""
SpaceCast-Bench loader and scorer: predictive spatial reasoning evaluation.

Adapted from "SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in
Vision-Language Models" (arXiv 2610.12402). Where the existing benchmarks
test spatial perception (relations visible in the input), SpaceCast-Bench
tests predictive spatial reasoning on an observe-transform-infer framework:
models integrate multi-view observations (main view A, bridge views, main
view B), apply a described transformation, and reason about the unseen
outcome. Its 3,862 questions span 16 task types at three capability levels:

    L1 - static perception
    L2 - local prediction
    L3 - global prediction

The loader normalizes rows from hongxingli/SpaceCast-Bench into the common
benchmark item schema consumed by vqasynth.benchmarks.BenchmarkRunner, and
the scorer mirrors the official protocol: answers are exact letter-set
matches with no partial credit for multi-select. spacecast_macro_type_accuracy
reproduces the paper's headline metric, the unweighted mean accuracy over
the 16 (level, task-type) categories.
"""

import re
from collections import defaultdict

from huggingface_hub import hf_hub_download

from vqasynth.evaluation import extract_option

try:
    # Optional at import time so the scoring helpers stay importable in
    # environments without a working pyarrow/datasets (mirrors the guarded
    # heavy-dep pattern in tests/test_correspondence.py).
    from datasets import load_dataset
except ImportError:  # pragma: no cover - environment dependent
    load_dataset = None

SPACECAST_REPO_ID = "hongxingli/SpaceCast-Bench"

# Capability levels (paper: static perception, local prediction, global
# prediction) -> report display names.
SPACECAST_LEVELS = {
    "L1": "Static Perception",
    "L2": "Local Prediction",
    "L3": "Global Prediction",
}


def _value_to_letter(value, options):
    """Map an option value back to its chr(65 + i) letter, if present."""
    for i, opt in enumerate(options):
        if str(opt).strip().lower() == str(value).strip().lower():
            return {chr(65 + i)}
    return set()


def _answer_letters(row):
    """
    Gold answer as a set of option letters in chr(65 + i) space.

    The released schema stores ``answer`` as a list of option labels
    (e.g. ["C"]); option values are mapped through their index as a fallback
    for robustness against schema drift, with ``correct_values`` as a last
    resort.
    """
    options = row.get("options") or []
    answers = [str(a).strip() for a in (row.get("answer") or []) if str(a).strip()]

    letters = set()
    for a in answers:
        if len(a) == 1 and "A" <= a.upper() <= "Z":
            letters.add(a.upper())
        else:
            letters |= _value_to_letter(a, options)
    if not letters:
        for v in row.get("correct_values") or []:
            letters |= _value_to_letter(str(v), options)
    return letters


def _row_images(row, repo_id):
    """
    Resolve a row's images: embedded payloads (PIL / bytes / HF image dicts)
    pass through unchanged, otherwise image_paths are fetched from the
    HuggingFace cache, preserving the benchmark's protocol order
    (main view A, bridge views, main view B).
    """
    embedded = [img for img in (row.get("images") or []) if img is not None]
    if embedded:
        return embedded
    return [
        hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")
        for path in (row.get("image_paths") or [])
    ]


def load_spacecast(repo_id=SPACECAST_REPO_ID, split="test", config=None, streaming=False):
    """
    Load SpaceCast-Bench from HuggingFace (auto-downloading on first access,
    cached afterwards).

    ``category`` carries the task type (16 task types -> the paper's macro
    categories) and ``subcategory`` the capability level, so BenchmarkRunner
    reports break down along both axes.

    Returns list of normalized items.
    """
    if load_dataset is None:
        raise ImportError(
            "SpaceCast-Bench loading requires the `datasets` package "
            "(pip install datasets)"
        )
    ds = load_dataset(repo_id, name=config, split=split, streaming=streaming)

    items = []
    for entry in ds:
        level = str(entry.get("level", ""))
        task_type = (
            entry.get("type") or entry.get("task") or entry.get("question_type")
        )
        letters = _answer_letters(entry)
        items.append({
            "id": str(
                entry.get("question_uid", f"{entry.get('scene_id', 'item')}_{len(items)}")
            ),
            "question": entry.get("question", ""),
            "answer": ",".join(sorted(letters)),
            "question_type": "multi-choice",
            "category": str(task_type or "unknown"),
            "subcategory": SPACECAST_LEVELS.get(level, level or "unknown"),
            "options": [str(o) for o in (entry.get("options") or [])],
            "images": _row_images(entry, repo_id),
            "source": "SpaceCast-Bench",
            "multi_select": bool(entry.get("multi_select", False)),
            "correct_values": [str(v) for v in (entry.get("correct_values") or [])],
            "level": level,
        })
    return items


def _item_target_letters(item):
    """Gold letters for a normalized item (answer string) or raw row (list)."""
    answer = item.get("answer")
    if isinstance(answer, str):
        return {p.strip().upper() for p in answer.split(",") if p.strip()}
    return _answer_letters(item)


def _extract_letter_set(text, num_options):
    """
    Extract the set of option letters (A..) chosen in a prediction.

    Scoped to text after an explicit answer marker when one is present, so
    chain-of-thought output that merely mentions option letters does not
    count as an answer. Used for multi-select and for option sets wider
    than the shared A-F extractor supports.
    """
    if not text:
        return set()
    scope = text
    marker = re.search(
        r"(?:answer\s*(?:is|:)|final\s+answer|<conclusion>)\s*[:>]*\s*(.*)$",
        text,
        re.IGNORECASE,
    )
    if marker:
        scope = marker.group(1)
    last = chr(65 + num_options - 1)
    found = re.findall(rf"\b([A-{last}])(?![a-zA-Z])", scope)
    return {f.upper() for f in found}


def score_spacecast_item(item, prediction):
    """
    Score a SpaceCast-Bench item: exact letter-set match against the gold
    answer, no partial credit for multi-select (the official protocol).

    Single-select items with up to six options reuse the shared
    ``extract_option`` cascade from vqasynth.evaluation; multi-select and
    wider option sets parse a letter set scoped to answer markers. Falls
    back to matching the correct option's text (SpaCE-10 convention).
    """
    options = item.get("options") or []
    if not options:
        return 0.0

    target = _item_target_letters(item)
    if not target:
        return 0.0

    prediction = prediction or ""
    multi = bool(item.get("multi_select"))

    if multi or len(options) > 6:
        predicted = _extract_letter_set(prediction, len(options))
    else:
        letter = extract_option(prediction)
        predicted = {letter} if letter else set()

    if predicted:
        return 1.0 if predicted == target else 0.0

    correct_values = item.get("correct_values") or []
    if not multi and correct_values:
        return 1.0 if correct_values[0].lower() in prediction.lower() else 0.0
    return 0.0


def spacecast_macro_type_accuracy(score_result):
    """
    Reproduce the paper's headline ``macro_type_accuracy``: the unweighted
    mean accuracy over the (level, task-type) categories (16 in the full
    benchmark; same-name tasks at different levels never merge).

    Args:
        score_result: Report dict from BenchmarkRunner.score(), whose
            per-item rows carry ``category`` = task type and
            ``subcategory`` = capability level.
    """
    per_item = score_result.get("per_item") or []
    groups = defaultdict(list)
    for row in per_item:
        key = (row.get("subcategory"), row.get("category"))
        groups[key].append(row.get("score", 0.0))
    if not groups:
        return 0.0
    return sum(sum(g) / len(g) for g in groups.values()) / len(groups)
