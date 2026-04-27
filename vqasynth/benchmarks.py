"""
Benchmark dataset loaders for spatial reasoning evaluation.

Loads and normalizes datasets from SpatialScore, OmniSpatial, SpaCE-10,
and MindCube into a common format for evaluation.
"""

import io
import json
import os
import re
import zipfile
from collections import defaultdict

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from vqasynth.evaluation import (
    classify_question,
    extract_option,
    extract_yes_no,
    llm_judge,
    score_choice,
    score_distance,
    score_distance_mra,
    score_yes_no,
)


def _ensure_zip_extracted(repo_id, filename, repo_type="dataset"):
    """
    Download a zip from HuggingFace (cached) and extract it next to the cached
    file on first access. Returns the directory containing extracted contents.
    """
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    extract_dir = zip_path + ".extracted"
    marker = os.path.join(extract_dir, ".extracted_ok")
    if not os.path.exists(marker):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        with open(marker, "w") as f:
            f.write("ok")
    return extract_dir


# ---------------------------------------------------------------------------
# Common item schema
# ---------------------------------------------------------------------------
# Each loader normalizes benchmark items to:
# {
#     "id": str,
#     "question": str,
#     "answer": str,             # ground truth answer text
#     "question_type": str,      # "multi-choice", "judgment", "open-ended"
#     "category": str,           # benchmark-specific category
#     "subcategory": str,        # benchmark-specific subcategory
#     "options": list[str],      # answer options for multi-choice
#     "images": list,            # PIL images (loaded lazily if possible)
#     "source": str,             # benchmark name
# }


# ---------------------------------------------------------------------------
# SpatialScore Loader
# ---------------------------------------------------------------------------

def load_spatialscore():
    """
    Load SpatialScore benchmark, auto-downloading the zip from HuggingFace
    (haoningwu/SpatialScore) on first access. The zip is ~11 GB; first run
    will be slow, subsequent runs read from the HF cache.

    Returns list of normalized items.
    """
    extract_dir = _ensure_zip_extracted("haoningwu/SpatialScore", "SpatialScore.zip")
    json_path = None
    for root, _, files in os.walk(extract_dir):
        if "SpatialScore.json" in files:
            json_path = os.path.join(root, "SpatialScore.json")
            break
    if json_path is None:
        raise FileNotFoundError(
            f"SpatialScore.json not found under {extract_dir}; the zip layout may have changed"
        )
    dataset_root = os.path.dirname(json_path)

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    items = []
    for entry in raw_data:
        question_type = entry.get("question_type", "open-ended")

        options = []
        if question_type == "multi-choice":
            q_text = entry.get("question", "")
            opt_matches = re.findall(r"\(([A-F])\)\s*([^(]+?)(?=\([A-F]\)|$)", q_text)
            options = [m[1].strip() for m in opt_matches]

        # img_paths in the JSON are relative to dataset root; resolve them
        img_paths = entry.get("img_paths", [])
        resolved = [os.path.join(dataset_root, p) if not os.path.isabs(p) else p
                    for p in img_paths]

        items.append({
            "id": str(entry.get("id", entry.get("index", len(items)))),
            "question": entry.get("question", ""),
            "answer": entry.get("answer", ""),
            "question_type": question_type,
            "category": entry.get("category", "unknown"),
            "subcategory": entry.get("subcategory", "unknown"),
            "options": options,
            "images": resolved,
            "source": "SpatialScore",
        })

    return items


# ---------------------------------------------------------------------------
# OmniSpatial Loader
# ---------------------------------------------------------------------------

def load_omnispatial(split="test"):
    """
    Load OmniSpatial benchmark, auto-downloading the zip from HuggingFace
    (qizekun/OmniSpatial) on first access. Test split is ~1.6 GB.

    The zip ships a JSON metadata file plus an images/ directory; this loader
    walks the extracted tree to find them.

    Returns list of normalized items.
    """
    zip_filename = f"OmniSpatial-{split}.zip"
    extract_dir = _ensure_zip_extracted("qizekun/OmniSpatial", zip_filename)

    json_path = None
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            if fn.endswith(".json") and ("test" in fn.lower() or "data" in fn.lower() or "metadata" in fn.lower()):
                json_path = os.path.join(root, fn)
                break
        if json_path:
            break

    if json_path is None:
        # Fallback: pick the largest .json in the tree
        candidates = []
        for root, _, files in os.walk(extract_dir):
            for fn in files:
                if fn.endswith(".json"):
                    p = os.path.join(root, fn)
                    candidates.append((os.path.getsize(p), p))
        if not candidates:
            raise FileNotFoundError(f"No JSON metadata found under {extract_dir}")
        candidates.sort(reverse=True)
        json_path = candidates[0][1]

    dataset_root = os.path.dirname(json_path)

    with open(json_path, "r") as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):
        # Some distributions wrap the list under a key
        for k in ("data", "items", "questions", "test"):
            if k in raw_data and isinstance(raw_data[k], list):
                raw_data = raw_data[k]
                break

    items = []
    for entry in raw_data:
        answer_field = entry.get("answer", entry.get("gt_answer", 0))
        options = entry.get("options", [])

        if isinstance(answer_field, int):
            gt_letter = chr(65 + answer_field)
            gt_text = options[answer_field] if answer_field < len(options) else gt_letter
        else:
            gt_letter = str(answer_field).strip().upper()[:1] if str(answer_field).strip() else ""
            gt_text = str(answer_field)

        # Resolve image path
        image_field = entry.get("image", entry.get("image_path", entry.get("img_path", "")))
        images = []
        if isinstance(image_field, str) and image_field:
            resolved = image_field if os.path.isabs(image_field) else os.path.join(dataset_root, image_field)
            images = [resolved]
        elif isinstance(image_field, list):
            images = [
                p if os.path.isabs(p) else os.path.join(dataset_root, p)
                for p in image_field if isinstance(p, str)
            ]

        items.append({
            "id": str(entry.get("id", len(items))),
            "question": entry.get("question", ""),
            "answer": gt_letter,
            "answer_text": gt_text,
            "question_type": "multi-choice",
            "category": entry.get("task_type", entry.get("category", "unknown")),
            "subcategory": entry.get("sub_task_type", entry.get("subcategory", "unknown")),
            "options": options,
            "images": images,
            "source": "OmniSpatial",
        })

    return items


# ---------------------------------------------------------------------------
# SpaCE-10 Loader
# ---------------------------------------------------------------------------

SPACE10_SUBSETS = ["ep", "eq", "fr", "oo", "os", "sa", "sp", "sq"]

SPACE10_SUBSET_NAMES = {
    "ep": "Entity Presence",
    "eq": "Entity Quantification",
    "fr": "Functional Reasoning",
    "oo": "Object-Object Spatial Relationship",
    "os": "Object-Scene Spatial Relationship",
    "sa": "Size Assessment",
    "sp": "Spatial Planning",
    "sq": "Scene Quantification",
}


def load_space10(subsets=None, streaming=False):
    """
    Load SpaCE-10 benchmark from HuggingFace.

    Dataset: Cusyoung/SpaCE-10

    Args:
        subsets: List of subset abbreviations (e.g., ["ep", "oo"]).
                 Defaults to all 8 subsets.
        streaming: If True, returns items lazily.

    Returns list of normalized items.
    """
    if subsets is None:
        subsets = SPACE10_SUBSETS

    items = []
    for subset in subsets:
        data_dir = f"single-choice/{subset}"
        try:
            ds = load_dataset(
                "Cusyoung/SpaCE-10",
                data_dir=data_dir,
                split="test",
                streaming=streaming,
            )
        except Exception as e:
            print(f"Warning: Could not load SpaCE-10 subset '{subset}': {e}")
            continue

        for entry in ds:
            # Build options list
            options = []
            for letter in ["A", "B", "C", "D", "E", "F"]:
                if letter in entry and entry[letter]:
                    options.append(entry[letter])

            # Parse images from bytes
            images = []
            if "image" in entry and entry["image"] is not None:
                img_data = entry["image"]
                if isinstance(img_data, list):
                    images = img_data  # list of PIL images or bytes
                else:
                    images = [img_data]

            items.append({
                "id": f"{subset}_{entry.get('index', len(items))}",
                "question": entry.get("question", ""),
                "answer": entry.get("answer", ""),
                "question_type": "multi-choice",
                "category": subset,
                "subcategory": SPACE10_SUBSET_NAMES.get(subset, subset),
                "options": options,
                "images": images,
                "source": "SpaCE-10",
            })

    return items


# ---------------------------------------------------------------------------
# MindCube Loader
# ---------------------------------------------------------------------------

MINDCUBE_SETTINGS = ["rotation", "among", "around", "translation"]


def _mindcube_setting_from_id(item_id):
    """Extract setting type from MindCube item ID."""
    item_id_lower = item_id.lower()
    for setting in ["around", "rotation", "translation", "among"]:
        if setting in item_id_lower:
            return setting
    return "other"


def load_mindcube(jsonl_name="MindCube_tinybench.jsonl"):
    """
    Load MindCube benchmark, auto-downloading data.zip from HuggingFace
    (MLL-Lab/MindCube) on first access (~640 MB).

    The zip contains data/raw/ with three JSONL files:
      - MindCube_tinybench.jsonl  (~570 items, default)
      - MindCube.jsonl            (full bench, ~17 MB)
      - MindCube_train.jsonl      (training split)
    and data/other_all_image/ holding the referenced images.

    Args:
        jsonl_name: Which JSONL file to read inside data/raw/.

    Returns list of normalized items.
    """
    extract_dir = _ensure_zip_extracted("MLL-Lab/MindCube", "data.zip")
    data_dir = os.path.join(extract_dir, "data")
    jsonl_path = os.path.join(data_dir, "raw", jsonl_name)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"{jsonl_path} not found in extracted MindCube data")

    items = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            item_id = str(entry.get("id", len(items)))
            setting = _mindcube_setting_from_id(item_id)

            # Resolve image paths relative to data/ directory inside zip
            raw_images = entry.get("images", [])
            resolved = [
                p if os.path.isabs(p) else os.path.join(data_dir, p)
                for p in raw_images
            ]

            items.append({
                "id": item_id,
                "question": entry.get("question", ""),
                "answer": entry.get("gt_answer", entry.get("answer", "")),
                "question_type": "multi-choice",
                "category": setting,
                "subcategory": setting,
                "options": [],
                "images": resolved,
                "source": "MindCube",
            })

    return items


# ---------------------------------------------------------------------------
# Unified Loader
# ---------------------------------------------------------------------------

BENCHMARK_LOADERS = {
    "spatialscore": load_spatialscore,
    "omnispatial": load_omnispatial,
    "space10": load_space10,
    "mindcube": load_mindcube,
}


def load_benchmark(name, **kwargs):
    """
    Load a benchmark dataset by name.

    Args:
        name: Benchmark name (spatialscore, omnispatial, space10, mindcube).
        **kwargs: Passed to the specific loader.

    Returns list of normalized items.
    """
    name_lower = name.lower()
    if name_lower not in BENCHMARK_LOADERS:
        valid = ", ".join(BENCHMARK_LOADERS.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Valid: {valid}")
    return BENCHMARK_LOADERS[name_lower](**kwargs)


# ---------------------------------------------------------------------------
# Benchmark Scorer
# ---------------------------------------------------------------------------

# Scoring dispatch per (benchmark, question_type)
# Each benchmark defines how to score its own question types

def _score_spatialscore_item(item, prediction):
    """Score a SpatialScore item using its native metrics."""
    qt = item["question_type"]
    gt = item["answer"]

    if qt == "multi-choice":
        pred_opt = extract_option(prediction)
        gt_opt = extract_option(gt)
        if pred_opt and gt_opt:
            return 1.0 if pred_opt == gt_opt else 0.0
        return 1.0 if prediction.strip().upper() == gt.strip().upper() else 0.0

    elif qt == "judgment":
        return score_yes_no(prediction, gt)

    else:  # open-ended
        # Check if it's a distance question
        has_unit = any(
            u in gt.lower()
            for u in ["meter", "meters", "cm", "feet", "inch", "inches", "ft", "m "]
        )
        if has_unit:
            score = score_distance(prediction, gt, tolerance=2.0)
            if score is not None:
                return score

        # Try MRA for numeric answers
        mra = score_distance_mra(prediction, gt)
        if mra is not None:
            return mra

        # Exact match fallback
        return 1.0 if prediction.strip().lower() == gt.strip().lower() else 0.0


def _score_omnispatial_item(item, prediction):
    """Score an OmniSpatial item (always multi-choice accuracy)."""
    gt_letter = item["answer"]
    pred_opt = extract_option(prediction)

    if pred_opt:
        return 1.0 if pred_opt == gt_letter.upper() else 0.0

    # Fallback: check if prediction contains the correct option text
    if item.get("answer_text"):
        return 1.0 if item["answer_text"].lower() in prediction.lower() else 0.0

    return 0.0


def _score_space10_item(item, prediction):
    """Score a SpaCE-10 item (single-choice accuracy)."""
    gt = item["answer"].strip().upper()
    pred = extract_option(prediction)

    if pred:
        return 1.0 if pred == gt else 0.0

    # Try matching by option text
    options = item.get("options", [])
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        if letter == gt and opt.lower() in prediction.lower():
            return 1.0

    return 0.0


def _score_mindcube_item(item, prediction):
    """Score a MindCube item (answer accuracy A-E)."""
    gt = item["answer"].strip().upper()
    pred = extract_option(prediction)

    if pred:
        return 1.0 if pred == gt else 0.0

    return 1.0 if prediction.strip().upper() == gt else 0.0


BENCHMARK_SCORERS = {
    "spatialscore": _score_spatialscore_item,
    "omnispatial": _score_omnispatial_item,
    "space10": _score_space10_item,
    "mindcube": _score_mindcube_item,
}


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs evaluation against external benchmark datasets.

    Loads benchmark data, scores predictions, and generates reports
    with benchmark-native categories and sub-categories.
    """

    def __init__(self, benchmarks=None, llm_client=None, llm_model="gpt-4o"):
        """
        Args:
            benchmarks: List of benchmark names, or "all".
            llm_client: Optional OpenAI client for LLM judge fallback.
            llm_model: Model name for LLM judge.
        """
        if benchmarks is None or benchmarks == "all":
            self.benchmarks = list(BENCHMARK_LOADERS.keys())
        elif isinstance(benchmarks, str):
            self.benchmarks = [benchmarks.lower()]
        else:
            self.benchmarks = [b.lower() for b in benchmarks]

        self.llm_client = llm_client
        self.llm_model = llm_model

    def load(self, benchmark_name, **kwargs):
        """Load a benchmark dataset."""
        return load_benchmark(benchmark_name, **kwargs)

    def score(self, benchmark_name, items, predictions):
        """
        Score predictions against benchmark items.

        Args:
            benchmark_name: Name of the benchmark.
            items: List of normalized benchmark items.
            predictions: Dict mapping item ID -> prediction string,
                        or list of prediction strings (same order as items).

        Returns dict with per-item scores and aggregated results.
        """
        scorer = BENCHMARK_SCORERS.get(benchmark_name.lower())
        if scorer is None:
            raise ValueError(f"No scorer for benchmark '{benchmark_name}'")

        # Normalize predictions to dict
        if isinstance(predictions, list):
            pred_map = {item["id"]: predictions[i] for i, item in enumerate(items) if i < len(predictions)}
        else:
            pred_map = predictions

        per_item = []
        category_scores = defaultdict(list)
        subcategory_scores = defaultdict(list)
        all_scores = []

        for item in items:
            pred = pred_map.get(item["id"], "")
            score = scorer(item, pred)

            # LLM judge fallback if scorer returned None
            if score is None and self.llm_client is not None:
                score = llm_judge(
                    self.llm_client, self.llm_model,
                    item["question"], pred, item["answer"],
                )
            if score is None:
                score = 0.0

            per_item.append({
                "id": item["id"],
                "score": score,
                "category": item["category"],
                "subcategory": item["subcategory"],
                "question_type": item["question_type"],
            })

            all_scores.append(score)
            category_scores[item["category"]].append(score)
            subcategory_scores[item["subcategory"]].append(score)

        def _agg(scores):
            return {"accuracy": sum(scores) / len(scores) if scores else 0.0, "count": len(scores)}

        return {
            "benchmark": benchmark_name,
            "overall_accuracy": _agg(all_scores)["accuracy"],
            "total": len(all_scores),
            "by_category": {cat: _agg(s) for cat, s in sorted(category_scores.items())},
            "by_subcategory": {sub: _agg(s) for sub, s in sorted(subcategory_scores.items())},
            "per_item": per_item,
        }

    def run(self, predictions_by_benchmark, load_kwargs=None):
        """
        Run evaluation across configured benchmarks.

        Args:
            predictions_by_benchmark: Dict mapping benchmark name to predictions.
                Each value is either:
                - A dict mapping item ID -> prediction string
                - A list of prediction strings (same order as loaded items)
            load_kwargs: Optional dict mapping benchmark name to loader kwargs.

        Returns a full report dict.
        """
        load_kwargs = load_kwargs or {}
        report = {
            "summary": {},
            "benchmarks": {},
        }

        for bname in self.benchmarks:
            if bname not in predictions_by_benchmark:
                continue

            kwargs = load_kwargs.get(bname, {})
            items = self.load(bname, **kwargs)
            preds = predictions_by_benchmark[bname]
            result = self.score(bname, items, preds)

            report["benchmarks"][result["benchmark"]] = {
                "overall_accuracy": result["overall_accuracy"],
                "total": result["total"],
                "by_category": result["by_category"],
                "by_subcategory": result["by_subcategory"],
            }
            report["summary"][result["benchmark"]] = result["overall_accuracy"]

        return report

    def get_benchmark_items(self, benchmark_name, **kwargs):
        """
        Load benchmark items for model inference.

        Returns list of dicts with 'id', 'question', 'options', 'images'
        that can be passed to a model. Ground truth is excluded.
        """
        items = self.load(benchmark_name, **kwargs)
        return [
            {
                "id": item["id"],
                "question": item["question"],
                "options": item["options"],
                "images": item["images"],
                "category": item["category"],
                "subcategory": item["subcategory"],
            }
            for item in items
        ]


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------

def format_benchmark_report(report):
    """
    Format a benchmark report as a human-readable string.

    Args:
        report: Report dict from BenchmarkRunner.run().

    Returns formatted string.
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("SPATIAL REASONING BENCHMARK REPORT")
    lines.append("=" * 70)

    if report.get("summary"):
        lines.append("")
        lines.append(f"{'Benchmark':<25} {'Overall Accuracy':>18} {'Samples':>10}")
        lines.append("-" * 55)
        for bname, acc in report["summary"].items():
            total = report["benchmarks"][bname]["total"]
            lines.append(f"{bname:<25} {acc:>17.1%} {total:>10}")

    for bname, bdata in report.get("benchmarks", {}).items():
        lines.append("")
        lines.append(f"--- {bname} ---")
        lines.append(f"  Overall: {bdata['overall_accuracy']:.1%} ({bdata['total']} samples)")

        if bdata.get("by_category"):
            lines.append("")
            lines.append(f"  {'Category':<40} {'Accuracy':>10} {'Count':>8}")
            lines.append(f"  {'-' * 60}")
            for cat, cdata in bdata["by_category"].items():
                lines.append(f"  {cat:<40} {cdata['accuracy']:>9.1%} {cdata['count']:>8}")

        if bdata.get("by_subcategory") and bdata["by_subcategory"] != bdata.get("by_category"):
            # Only show subcategories if they differ from categories
            subcats = bdata["by_subcategory"]
            cats = bdata.get("by_category", {})
            if set(subcats.keys()) != set(cats.keys()):
                lines.append("")
                lines.append(f"  {'Subcategory':<40} {'Accuracy':>10} {'Count':>8}")
                lines.append(f"  {'-' * 60}")
                for sub, sdata in subcats.items():
                    lines.append(f"  {sub:<40} {sdata['accuracy']:>9.1%} {sdata['count']:>8}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("")
    return "\n".join(lines)
