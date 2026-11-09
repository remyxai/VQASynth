"""
Mirror-induced RGB-D ambiguity — MIRA-Bench loader, scorer, and depth-delta audit.

Adapted from "When Depth Lies: Benchmarking Vision-Language Models on
Mirror-Induced RGB-D Ambiguity" (arXiv:2610.41651, MIRA-Bench). A mirror
reproduces an object's appearance while placing its depth reading *behind* the
glass, so the "depth is faithful" assumption behind VQASynth's VGGT point-cloud
stage breaks exactly where the pipeline is most confident. MIRA-Bench measures
that failure: 14,140 QA pairs over 1,011 RGB-D mirror scenes, organized as eight
subtasks (14 subtask/variant labels in the released annotations) across three
cognitive levels, with per-item flags for whether the question needs depth or
mirror reasoning.

This is an ADAPTED PORT (Mode 2): the QA schema, the subtask -> cognitive-level
grouping, the option/yes-no/count scoring dispatch and the partial-credit
(Jaccard) multi-select rule follow the benchmark; the paper's own vLLM serving
harness, its mirror-mask / bbox visual-prompt caches and its normalized-accuracy
aggregation are intentionally out of scope. Answer extraction and yes/no
scoring reuse the existing :mod:`vqasynth.evaluation` primitives, and
:func:`register_mira_bench` plugs the loader/scorer into the
:mod:`vqasynth.benchmarks` registries so the benchmark flows through
``BenchmarkRunner`` (and its LLM-judge fallback) like SpatialScore or MindCube.

The paper's headline result is reproduced as an audit rather than a training
run: :func:`depth_delta` compares the same items answered under an RGB input
against an RGB-D input and reports whether adding depth *helped or hurt* —
MIRA-Bench finds depth consistently degrades reasoning in mirror scenes.

Annotations are CC BY-NC 4.0 and are consumed eval-only via the HuggingFace
cache, like the repo's other benchmark loaders.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from vqasynth.evaluation import extract_number, extract_option, score_yes_no

MIRA_BENCH_NAME = "mirabench"
MIRA_REPO_ID = "evayinhao/MIRA-Bench"
MIRA_QUESTIONS_FILE = "annotations/questions.json"

# MIRA question_type -> the common question_type vocabulary shared with the
# other benchmark loaders (see vqasynth/benchmarks.py).
MIRA_QUESTION_TYPES = {
    "yes_no": "judgment",
    "single_choice": "multi-choice",
    "multi_choice": "multi-choice",
    "single_choice_plus_text": "multi-choice",
    "count": "open-ended",
    "spatial_3d_axes": "open-ended",
}

# The paper's three cognitive levels, with the subtask labels each rolls up
# from (the released annotations carry these 14 subcategory names).
REFLECTION_PERCEPTION = "reflection_perception"
SPATIAL_REASONING = "spatial_reasoning"
SCENE_DECISION_MAKING = "scene_decision_making"

COGNITIVE_LEVELS = {
    REFLECTION_PERCEPTION: (
        "visibility_classification",
        "mirror_correspondence",
        "mirror_correspondence_r2m",
        "mirror_correspondence_m2r",
        "same_object_verification",
    ),
    SPATIAL_REASONING: (
        "spatial_relation_reasoning",
        "mirror_to_real_spatial_inference",
        "relative_distance_ordering",
        "counting_based_spatial_reasoning",
    ),
    SCENE_DECISION_MAKING: (
        "reachability",
        "navigation_movement_planning",
        "safety_feasibility_awareness",
        "next_step_action_selection_real_target",
        "collision_risk_awareness",
    ),
}

_LEVEL_BY_SUBCATEGORY = {
    subcategory: level
    for level, subcategories in COGNITIVE_LEVELS.items()
    for subcategory in subcategories
}

# A standalone capital letter A-F (the tail stops "A" matching inside words).
_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Fa-f])(?![A-Za-z])")


def cognitive_level(subcategory):
    """Map a MIRA subtask name to its cognitive level (``other`` if unknown)."""
    return _LEVEL_BY_SUBCATEGORY.get(str(subcategory or "").strip(), "other")


def _maybe_json(value):
    """MIRA serializes list/tuple fields as JSON text; decode those in place."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", '"')):
            try:
                return json.loads(text)
            except ValueError:
                return value
    return value


def _resolve(path, data_root):
    """Resolve a dataset-relative path against ``data_root`` when one is given."""
    if not path or not isinstance(path, str):
        return None
    if os.path.isabs(path):
        return path
    if data_root:
        return os.path.join(str(data_root), path)
    return path


def normalize_mira_entry(entry, index, data_root=None, seen=None):
    """
    Normalize one ``questions.json`` record into the common benchmark schema.

    Fields follow the MIRA annotation format: ``question_id`` (not guaranteed
    unique), ``question``/``choices``/``answer``, ``question_type``,
    ``category``/``subcategory``, ``image_path`` plus a depth path, and the
    ``requires_depth`` / ``requires_mirror_reasoning`` flags. ``category``
    carries the fine-grained subtask and ``subcategory`` its cognitive level,
    which is how ``BenchmarkRunner`` reports both granularities.
    """
    seen = seen if seen is not None else set()
    question_id = str(entry.get("question_id", index))
    item_id = question_id
    while item_id in seen:  # question_id collisions get an index suffix
        item_id = f"{question_id}_{index}"
    seen.add(item_id)

    subtask = str(
        _maybe_json(entry.get("subcategory")) or entry.get("category") or "unknown"
    ).strip()
    mira_type = str(entry.get("question_type", "")).strip()
    choices = _maybe_json(entry.get("choices"))
    choices = [str(c) for c in choices] if isinstance(choices, list) else []
    answer = _maybe_json(entry.get("answer"))
    image_path = _resolve(entry.get("image_path"), data_root)
    depth_path = _resolve(
        entry.get("input_depth_image_path") or entry.get("depth_image_path"), data_root
    )

    return {
        "id": item_id,
        "question_id": question_id,
        "question": str(entry.get("question", "")),
        "answer": answer if answer is not None else "",
        "question_type": MIRA_QUESTION_TYPES.get(mira_type, "open-ended"),
        "mira_question_type": mira_type,
        "category": subtask,
        "subcategory": cognitive_level(subtask),
        "options": choices,
        "images": [image_path] if image_path else [],
        "depth_images": [depth_path] if depth_path else [],
        "requires_depth": bool(entry.get("requires_depth", False)),
        "requires_mirror_reasoning": bool(entry.get("requires_mirror_reasoning", True)),
        "source": "MIRA-Bench",
    }


def load_mira_bench(
    questions_json=None, data_root=None, categories=None, max_items=0
):
    """
    Load MIRA-Bench QA items, downloading the annotations from HuggingFace
    (evayinhao/MIRA-Bench, ~21 MB) on first access.

    Args:
        questions_json: Path to a local ``questions.json``. Defaults to the
            HuggingFace-cached copy (downloaded if missing).
        data_root: Root the annotation's relative image paths resolve against
            (the MIRA ``Dataset/`` directory). Images are referenced, not
            loaded — third-party frames are distributed separately upstream.
        categories: Optional list of MIRA subtask names to keep.
        max_items: Cap on returned items (0 = no cap); useful for smoke runs.

    Returns list of normalized items.
    """
    if questions_json is None:
        from huggingface_hub import hf_hub_download  # deferred: keeps import light

        try:
            questions_json = hf_hub_download(
                repo_id=MIRA_REPO_ID, filename=MIRA_QUESTIONS_FILE, repo_type="dataset"
            )
        except Exception as error:  # gated repo / offline: point at the manual path
            raise RuntimeError(
                f"could not fetch {MIRA_QUESTIONS_FILE} from {MIRA_REPO_ID}: {error}. "
                f"The MIRA-Bench release is gated — accept its terms, run "
                f"`hf download {MIRA_REPO_ID} --repo-type dataset --local-dir Dataset`, "
                f"then pass questions_json=Dataset/annotations/questions.json."
            ) from error

    with open(questions_json, "r", encoding="utf-8") as handle:
        raw_entries = json.load(handle)
    if not isinstance(raw_entries, list):
        raise ValueError(
            f"{questions_json} expected a JSON list of QA entries, "
            f"got {type(raw_entries).__name__}"
        )

    wanted = {str(c) for c in categories} if categories else None
    items, seen = [], set()
    for index, entry in enumerate(raw_entries):
        item = normalize_mira_entry(entry, index, data_root=data_root, seen=seen)
        if wanted is not None and item["category"] not in wanted:
            continue
        items.append(item)
        if max_items and len(items) >= max_items:
            break
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def gold_option_set(item):
    """
    The set of correct option letters for a normalized item.

    MIRA answers are letters, integer indices into ``choices``, option text, or
    (for multi-select) a list of those; all four shapes normalize to letters so
    scoring can compare against the option letters extracted from a free-text
    prediction.
    """
    options = item.get("options") or []
    answer = _maybe_json(item.get("answer"))
    answers = answer if isinstance(answer, list) else [answer]

    letters = set()
    for ans in answers:
        if isinstance(ans, bool):
            continue
        if isinstance(ans, int) and 0 <= ans < len(options):
            letters.add(chr(65 + ans))
            continue
        text = str(ans).strip()
        if re.fullmatch(r"[A-Fa-f]", text):
            letters.add(text.upper())
            continue
        for position, option in enumerate(options):
            if option.strip().lower() == text.lower():
                letters.add(chr(65 + position))
    return letters


def _gold_option_text(item, letters):
    """The option text for the first of ``letters`` that has one."""
    options = item.get("options") or []
    for letter in sorted(letters):
        position = ord(letter) - 65
        if position < len(options):
            return options[position]
    return None


def score_mira_item(item, prediction):
    """
    Score one MIRA item.

    Judgment items reuse :func:`vqasynth.evaluation.score_yes_no`, choice items
    cascade ``extract_option`` -> option-text containment (single-select) or
    Jaccard over the mentioned letters (multi-select, the paper's partial
    credit), and count items compare extracted numbers. Returns ``None`` when
    nothing could be extracted, which routes the item to ``BenchmarkRunner``'s
    LLM-judge fallback.
    """
    prediction = prediction or ""
    question_type = item["question_type"]

    if question_type == "judgment":
        return score_yes_no(prediction, str(item["answer"]))

    if question_type == "multi-choice":
        gold = gold_option_set(item)
        if not gold:
            return None
        if len(gold) == 1:
            letter = extract_option(prediction)
            if letter is not None:
                return 1.0 if letter in gold else 0.0
            gold_text = _gold_option_text(item, gold)
            if gold_text:
                return 1.0 if gold_text.lower() in prediction.lower() else 0.0
            return None
        mentioned = {m.group(1).upper() for m in _LETTER_RE.finditer(prediction)}
        if not mentioned:
            return None
        union = gold | mentioned
        return len(gold & mentioned) / len(union) if union else 0.0

    # open-ended: count (and spatial_3d_axes free text) -> numeric comparison,
    # anything else falls through to the LLM judge.
    gold_number = extract_number(str(item["answer"]))
    pred_number = extract_number(prediction)
    if gold_number is not None and pred_number is not None:
        return 1.0 if abs(pred_number - gold_number) < 1e-6 else 0.0
    return None


def register_mira_bench():
    """
    Register MIRA-Bench in the :mod:`vqasynth.benchmarks` loader/scorer
    registries, after which ``BenchmarkRunner`` (and the eval stage built on
    it) handles ``mirabench`` exactly like the four built-in benchmarks.

    Registration is an explicit call rather than an import side effect, so
    ``--benchmark all`` runs keep their default benchmark set unless the caller
    opts in. Returns the registered benchmark name.
    """
    from vqasynth import benchmarks  # deferred: the loader layer pulls in `datasets`

    benchmarks.BENCHMARK_LOADERS.setdefault(MIRA_BENCH_NAME, load_mira_bench)
    benchmarks.BENCHMARK_SCORERS.setdefault(MIRA_BENCH_NAME, score_mira_item)
    return MIRA_BENCH_NAME


# ---------------------------------------------------------------------------
# RGB vs RGB-D depth-delta audit
# ---------------------------------------------------------------------------

@dataclass
class MirrorDeltaReport:
    """Does adding depth help or hurt on mirror-ambiguous scenes?"""

    total: int
    rgb_accuracy: float
    rgbd_accuracy: float
    depth_delta: float                 # rgbd - rgb; negative == depth hurts
    depth_harm_rate: float             # correct on RGB, wrong once depth is added
    depth_help_rate: float             # wrong on RGB, correct with depth
    decision_change_rate: float        # the answer changed at all (label-free)
    by_category: dict = field(default_factory=dict)
    by_level: dict = field(default_factory=dict)
    by_requires_depth: dict = field(default_factory=dict)
    per_item: list = field(default_factory=list)


def _prediction(record, *keys):
    for key in keys:
        if isinstance(record, dict) and record.get(key):
            return str(record[key])
    return None


def depth_delta(items, predictions, scorer=None):
    """
    Compare per-item correctness under an RGB input against an RGB-D input.

    This is the paper's controlled-input comparison as an audit: the caller
    collects the model's answers under both conditions (same items, same
    question) and this function reports the accuracy delta plus the harm/help
    flip rates. ``predictions`` maps item id -> ``{"rgb": str, "rgbd": str}``
    (``pred_rgb`` / ``pred_rgbd`` also accepted); items missing either
    prediction are skipped.

    Returns a :class:`MirrorDeltaReport` with per-subtask, per-cognitive-level
    and per-``requires_depth`` breakdowns.
    """
    if scorer is None:
        scorer = score_mira_item

    per_item = []
    groups = {"by_category": {}, "by_level": {}, "by_requires_depth": {}}
    rgb_n = rgbd_n = harmed = helped = changed = 0

    for item in items:
        record = predictions.get(item["id"], predictions.get(item["question_id"], {}))
        pred_rgb = _prediction(record, "rgb", "pred_rgb")
        pred_rgbd = _prediction(record, "rgbd", "pred_rgbd")
        if pred_rgb is None or pred_rgbd is None:
            continue

        # Same convention as BenchmarkRunner.score: an unextractable answer
        # scores 0 (that path is where its LLM-judge fallback would have run).
        rgb_score = float(scorer(item, pred_rgb) or 0.0)
        rgbd_score = float(scorer(item, pred_rgbd) or 0.0)

        per_item.append(
            {
                "id": item["id"],
                "category": item["category"],
                "level": item["subcategory"],
                "rgb_score": rgb_score,
                "rgbd_score": rgbd_score,
                "harmed": rgb_score > rgbd_score,
                "helped": rgbd_score > rgb_score,
            }
        )
        rgb_n += rgb_score
        rgbd_n += rgbd_score
        harmed += int(rgb_score > rgbd_score)
        helped += int(rgbd_score > rgb_score)
        changed += int(pred_rgb.strip() != pred_rgbd.strip())

        for key, group_key in (
            ("by_category", item["category"]),
            ("by_level", item["subcategory"]),
            ("by_requires_depth", bool(item.get("requires_depth"))),
        ):
            bucket = groups[key].setdefault(group_key, {"rgb": 0.0, "rgbd": 0.0, "n": 0})
            bucket["rgb"] += rgb_score
            bucket["rgbd"] += rgbd_score
            bucket["n"] += 1

    total = len(per_item)

    def _rate(numerator):
        return numerator / total if total else 0.0

    breakdowns = {
        key: {
            group: {
                "rgb_accuracy": bucket["rgb"] / bucket["n"] if bucket["n"] else 0.0,
                "rgbd_accuracy": bucket["rgbd"] / bucket["n"] if bucket["n"] else 0.0,
                "depth_delta": (
                    (bucket["rgbd"] - bucket["rgb"]) / bucket["n"] if bucket["n"] else 0.0
                ),
                "count": bucket["n"],
            }
            for group, bucket in sorted(groups[key].items(), key=lambda kv: str(kv[0]))
        }
        for key in groups
    }

    return MirrorDeltaReport(
        total=total,
        rgb_accuracy=_rate(rgb_n),
        rgbd_accuracy=_rate(rgbd_n),
        depth_delta=_rate(rgbd_n) - _rate(rgb_n),
        depth_harm_rate=_rate(harmed),
        depth_help_rate=_rate(helped),
        decision_change_rate=_rate(changed),
        by_category=breakdowns["by_category"],
        by_level=breakdowns["by_level"],
        by_requires_depth=breakdowns["by_requires_depth"],
        per_item=per_item,
    )


def format_mirror_report(report):
    """Format a :class:`MirrorDeltaReport`, matching the repo's report layout."""
    lines = [
        "",
        "=" * 70,
        "MIRROR-INDUCED RGB-D AMBIGUITY (RGB vs RGB-D)",
        "=" * 70,
        "",
        f"  ({report.total} items answered under both conditions)",
        "",
        f"  {'Metric':<34} {'Value':>12}",
        f"  {'-' * 48}",
        f"  {'Accuracy (RGB)':<34} {report.rgb_accuracy:>11.1%}",
        f"  {'Accuracy (RGB-D)':<34} {report.rgbd_accuracy:>11.1%}",
        f"  {'Depth delta':<34} {report.depth_delta:>+11.1%}",
        f"  {'Depth harm rate':<34} {report.depth_harm_rate:>11.1%}",
        f"  {'Depth help rate':<34} {report.depth_help_rate:>11.1%}",
        f"  {'Decision change rate':<34} {report.decision_change_rate:>11.1%}",
    ]
    for title, breakdown in (
        ("cognitive level", report.by_level),
        ("subtask", report.by_category),
    ):
        if not breakdown:
            continue
        lines.append("")
        lines.append(f"  By {title}")
        lines.append(f"  {'-' * 60}")
        lines.append(f"  {'Group':<36} {'RGB':>8} {'RGB-D':>8} {'Delta':>8} {'N':>6}")
        for group, bucket in breakdown.items():
            lines.append(
                f"  {str(group):<36} {bucket['rgb_accuracy']:>7.1%} "
                f"{bucket['rgbd_accuracy']:>7.1%} {bucket['depth_delta']:>+7.1%} "
                f"{bucket['count']:>6}"
            )
    lines.extend(["", "=" * 70, ""])
    return "\n".join(lines)
