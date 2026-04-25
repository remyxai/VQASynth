"""
Evaluation module for VQASynth spatial reasoning QA.

Implements scoring methods adapted from:
- SpatialScore: answer extraction, distance ratio tolerance, Mean Relative Accuracy
- OmniSpatial: multi-choice accuracy, LLM judge fallback, per-category breakdown
- SpaCE-10: cascading rule-based extraction with GPT fallback
- MindCube: multi-choice extraction with cascading regex

Supports running individual benchmarks or generating a full report across all.
"""

import re
import json
import math
from collections import defaultdict
import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Answer Extractors
# ---------------------------------------------------------------------------

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

UNIT_TO_CM = {
    "m": 100, "meter": 100, "meters": 100, "metre": 100, "metres": 100,
    "cm": 1, "centimeter": 1, "centimeters": 1, "centimetre": 1, "centimetres": 1,
    "mm": 0.1, "millimeter": 0.1, "millimeters": 0.1,
    "ft": 30.48, "foot": 30.48, "feet": 30.48,
    "in": 2.54, "inch": 2.54, "inches": 2.54,
}


def extract_yes_no(text):
    """
    Extract yes/no judgment from free text.

    Handles synonyms and patterns from SpatialScore:
    yes/yeah/yep/correct/true/right/indeed -> True
    no/nope/false/incorrect/wrong/not -> False

    Returns True, False, or None if ambiguous.
    """
    text_lower = text.strip().lower()

    # Check for explicit conclusion tags (SpatialScore pattern)
    tag_match = re.search(r"<conclusion>\s*(yes|no)\s*</conclusion>", text_lower)
    if tag_match:
        return tag_match.group(1) == "yes"

    yes_patterns = r"\b(yes|yeah|yep|correct|true|right|indeed|affirmative)\b"
    no_patterns = r"\b(no|nope|false|incorrect|wrong|negative)\b"

    # Check first word for direct responses
    first_word = text_lower.split(",")[0].split(".")[0].strip()
    if re.match(yes_patterns, first_word):
        return True
    if re.match(no_patterns, first_word):
        return False

    # Search full text
    has_yes = bool(re.search(yes_patterns, text_lower))
    has_no = bool(re.search(no_patterns, text_lower))

    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False

    return None


def extract_number(text):
    """
    Extract a numeric value from free text.

    Handles decimal numbers, integers, comma-separated numbers,
    and English words zero through twenty (from SpatialScore).

    Returns float or None.
    """
    text_lower = text.strip().lower()

    # Check for conclusion tags
    tag_match = re.search(r"<conclusion>\s*([\d.]+)\s*</conclusion>", text_lower)
    if tag_match:
        try:
            return float(tag_match.group(1))
        except ValueError:
            pass

    # Try numeric patterns: "3.5", "3,500", "3"
    num_match = re.search(r"(\d+(?:,\d{3})*(?:\.\d+)?)", text_lower)
    if num_match:
        try:
            return float(num_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Try English word numbers
    for word, val in WORD_TO_NUM.items():
        if re.search(rf"\b{word}\b", text_lower):
            return float(val)

    return None


def extract_numeric_with_unit(text):
    """
    Extract a numeric value paired with a unit from free text.

    Recognizes distance units: meters, centimeters, feet, inches, etc.
    Adapted from SpatialScore's extract_numeric_with_unit.

    Returns (value, unit_string) or None.
    """
    unit_pattern = "|".join(re.escape(u) for u in sorted(UNIT_TO_CM.keys(), key=len, reverse=True))

    # Pattern: "3.5 meters", "3.5m", "approximately 3.5 meters"
    match = re.search(
        rf"(\d+(?:\.\d+)?)\s*({unit_pattern})\b",
        text.lower()
    )
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2)
            return (value, unit)
        except ValueError:
            pass

    return None


def extract_option(text):
    """
    Extract a multi-choice option letter (A-F) from free text.

    Cascading regex strategy adapted from SpatialScore, MindCube, and SpaCE-10:
    1. "Answer: X" or "answer is X" patterns
    2. "(X)" or "[X]" patterns
    3. Standalone capital letter with punctuation
    4. Any letter A-F found in text

    Returns uppercase letter string or None.
    """
    text = text.strip()

    # 1. "Answer: X" pattern (OmniSpatial/MindCube)
    match = re.search(r"[Aa]nswer\s*:\s*([A-Fa-f])\b", text)
    if match:
        return match.group(1).upper()

    # 2. "answer is X" pattern (SpatialScore)
    match = re.search(r"answer\s+is\s+([A-Fa-f])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. Conclusion tags
    match = re.search(r"<conclusion>\s*\(?([A-Fa-f])\)?\s*</conclusion>", text)
    if match:
        return match.group(1).upper()

    # 4. Parenthesized: "(A)", "[A]"
    match = re.search(r"[\(\[]([A-Fa-f])[\)\]]", text)
    if match:
        return match.group(1).upper()

    # 5. Letter with period or at start: "A.", "A,"
    match = re.search(r"\b([A-Fa-f])[.,;:]\s", text)
    if match:
        return match.group(1).upper()

    # 6. Direct single letter response
    if re.match(r"^[A-Fa-f]\.?$", text.strip()):
        return text.strip()[0].upper()

    return None


def normalize_to_cm(value, unit):
    """
    Convert a distance value to centimeters.

    Conversion factors from SpatialScore.
    """
    unit_lower = unit.lower().strip()
    multiplier = UNIT_TO_CM.get(unit_lower)
    if multiplier is None:
        return None
    return value * multiplier


# ---------------------------------------------------------------------------
# Question Type Classifier
# ---------------------------------------------------------------------------

_DISTANCE_PATTERNS = re.compile(
    r"(distance between|how far|how close|how distant|measure the distance|"
    r"distance of the|distance from|distance measurement)", re.IGNORECASE
)
_VERTICAL_PATTERN = re.compile(r"vertic", re.IGNORECASE)
_HORIZONTAL_PATTERN = re.compile(r"horizont", re.IGNORECASE)
_MEASUREMENT_PATTERNS = re.compile(
    r"(width of|height of|how wide|how tall|measure the width|measure the height|"
    r"horizontal dimensions|vertical dimensions|radius of)", re.IGNORECASE
)
_CHOICE_PATTERNS = re.compile(
    r"(which is more to the|which one appears|who is positioned more|"
    r"which is above|which is below|which one is positioned|"
    r"who is taller|who is shorter|who is higher|who is lower)", re.IGNORECASE
)
_PREDICATE_PATTERNS = re.compile(
    r"(^is the |^does the |^can you confirm)", re.IGNORECASE
)


def classify_question(question):
    """
    Classify a VQASynth question into a scoring category.

    Returns one of: "distance", "vertical_distance", "horizontal_distance",
    "measurement", "comparison_yn", "comparison_choice", or "unknown".
    """
    q = question.strip()

    # Distance questions (check vertical/horizontal sub-types first)
    if _DISTANCE_PATTERNS.search(q):
        if _VERTICAL_PATTERN.search(q):
            return "vertical_distance"
        if _HORIZONTAL_PATTERN.search(q):
            return "horizontal_distance"
        return "distance"

    # Measurement (width/height of single object)
    if _MEASUREMENT_PATTERNS.search(q):
        return "measurement"

    # Choice questions ("Which is taller?", "Who is more to the left?")
    if _CHOICE_PATTERNS.search(q):
        return "comparison_choice"

    # Yes/no predicate questions ("Is the X to the left of Y?")
    if _PREDICATE_PATTERNS.search(q):
        return "comparison_yn"

    return "unknown"


# ---------------------------------------------------------------------------
# Scoring Functions
# ---------------------------------------------------------------------------

def score_distance(pred_text, gt_text, tolerance=2.0):
    """
    Score distance predictions using ratio tolerance.

    From SpatialScore: extract numeric+unit from both texts,
    normalize to cm, check max(pred/gt, gt/pred) < tolerance.

    Returns 1.0 (within tolerance), 0.0 (outside), or None (extraction failed).
    """
    pred_pair = extract_numeric_with_unit(pred_text)
    gt_pair = extract_numeric_with_unit(gt_text)

    if pred_pair is None or gt_pair is None:
        # Fallback: try extracting just the number
        pred_val = extract_number(pred_text)
        gt_val = extract_number(gt_text)
        if pred_val is None or gt_val is None:
            return None
        if pred_val == 0 and gt_val == 0:
            return 1.0
        if pred_val == 0 or gt_val == 0:
            return 0.0
        ratio = max(pred_val / gt_val, gt_val / pred_val)
        return 1.0 if ratio < tolerance else 0.0

    pred_cm = normalize_to_cm(*pred_pair)
    gt_cm = normalize_to_cm(*gt_pair)

    if pred_cm is None or gt_cm is None:
        return None
    if pred_cm == 0 and gt_cm == 0:
        return 1.0
    if pred_cm == 0 or gt_cm == 0:
        return 0.0

    ratio = max(pred_cm / gt_cm, gt_cm / pred_cm)
    return 1.0 if ratio < tolerance else 0.0


def score_distance_mra(pred_text, gt_text, start=0.5, end=0.95, interval=0.05):
    """
    Mean Relative Accuracy over confidence thresholds.

    From SpatialScore (VSI-Bench): for each threshold t in [start, end],
    check if |pred - gt| / gt <= 1 - t. Return mean of pass/fail across
    all thresholds. Provides continuous [0, 1] score.
    """
    pred_pair = extract_numeric_with_unit(pred_text)
    gt_pair = extract_numeric_with_unit(gt_text)

    if pred_pair is None or gt_pair is None:
        pred_val = extract_number(pred_text)
        gt_val = extract_number(gt_text)
    else:
        pred_val = normalize_to_cm(*pred_pair)
        gt_val = normalize_to_cm(*gt_pair)

    if pred_val is None or gt_val is None:
        return None
    if gt_val == 0:
        return 1.0 if pred_val == 0 else 0.0

    num_pts = int((end - start) / interval) + 2
    thresholds = np.linspace(start, end, num_pts)
    rel_error = abs(pred_val - gt_val) / abs(gt_val)
    accuracy = (rel_error <= (1 - thresholds)).astype(float)
    return float(accuracy.mean())


def score_yes_no(pred_text, gt_text):
    """
    Score yes/no predictions via extraction and exact match.

    Returns 1.0, 0.0, or None if extraction fails on either side.
    """
    pred = extract_yes_no(pred_text)
    gt = extract_yes_no(gt_text)

    if pred is None or gt is None:
        return None
    return 1.0 if pred == gt else 0.0


def score_choice(pred_text, gt_text, question=""):
    """
    Score comparison choice predictions.

    Extracts the chosen object name from both prediction and ground truth
    by matching against VQASynth answer templates. Falls back to checking
    if the ground truth object name appears in the prediction.
    """
    # GT answer templates contain the correct object name directly
    # e.g., "the chair is taller." or "Positioned to the left is the table."
    gt_lower = gt_text.strip().lower()
    pred_lower = pred_text.strip().lower()

    # Extract object names from the question (between "the X or the Y")
    names_match = re.search(
        r"the\s+(.+?)\s+or\s+the\s+(.+?)[\?\.]",
        question.lower()
    )

    if names_match:
        name_a = names_match.group(1).strip()
        name_b = names_match.group(2).strip()

        # Determine which name is in the GT answer
        gt_has_a = name_a in gt_lower
        gt_has_b = name_b in gt_lower

        if gt_has_a and not gt_has_b:
            correct_name = name_a
        elif gt_has_b and not gt_has_a:
            correct_name = name_b
        else:
            # Both or neither found; fall back to full text comparison
            return 1.0 if gt_lower == pred_lower else 0.0

        return 1.0 if correct_name in pred_lower else 0.0

    # Fallback: exact match
    return 1.0 if gt_lower == pred_lower else 0.0


# ---------------------------------------------------------------------------
# Benchmark Definitions
# ---------------------------------------------------------------------------

# Each benchmark defines how to score each question type and how to group results.
# "scorer" maps qa_type -> scoring function name
# "categories" maps display category -> list of qa_types that roll up into it

BENCHMARKS = {
    "spatialscore": {
        "name": "SpatialScore",
        "description": "Distance ratio tolerance, yes/no extraction, MRA for numeric (SpatialScore, CVPR 2026)",
        "scorer": {
            "distance": "ratio_tolerance",
            "vertical_distance": "ratio_tolerance",
            "horizontal_distance": "ratio_tolerance",
            "measurement": "ratio_tolerance",
            "comparison_yn": "yes_no",
            "comparison_choice": "choice",
            "unknown": "llm_judge",
        },
        "categories": {
            "Object Distance": ["distance", "vertical_distance", "horizontal_distance"],
            "Object Properties": ["measurement"],
            "Spatial Relations": ["comparison_yn", "comparison_choice"],
        },
    },
    "omnispatial": {
        "name": "OmniSpatial",
        "description": "Accuracy with LLM judge fallback, cognitive-psychology categories (OmniSpatial, arXiv 2506.03135)",
        "scorer": {
            "distance": "llm_judge",
            "vertical_distance": "llm_judge",
            "horizontal_distance": "llm_judge",
            "measurement": "llm_judge",
            "comparison_yn": "llm_judge",
            "comparison_choice": "llm_judge",
            "unknown": "llm_judge",
        },
        "categories": {
            "Spatial Interaction": ["distance", "vertical_distance", "horizontal_distance"],
            "Complex Logic": ["measurement"],
            "Perspective Taking": ["comparison_yn", "comparison_choice"],
        },
    },
    "space10": {
        "name": "SpaCE-10",
        "description": "Rule-based extraction with GPT fallback, compositional QA types (SpaCE-10)",
        "scorer": {
            "distance": "ratio_tolerance",
            "vertical_distance": "ratio_tolerance",
            "horizontal_distance": "ratio_tolerance",
            "measurement": "ratio_tolerance",
            "comparison_yn": "yes_no",
            "comparison_choice": "choice",
            "unknown": "llm_judge",
        },
        "categories": {
            "Size Assessment": ["measurement"],
            "Object-Object Spatial Relationship": ["comparison_yn", "comparison_choice"],
            "Spatial Planning": ["distance", "vertical_distance", "horizontal_distance"],
        },
    },
    "mindcube": {
        "name": "MindCube",
        "description": "Answer accuracy with MRA for distance, directional relation scoring (MindCube, arXiv 2506.21458)",
        "scorer": {
            "distance": "mra",
            "vertical_distance": "mra",
            "horizontal_distance": "mra",
            "measurement": "mra",
            "comparison_yn": "yes_no",
            "comparison_choice": "choice",
            "unknown": "llm_judge",
        },
        "categories": {
            "Rotation": ["vertical_distance", "horizontal_distance"],
            "Among": ["comparison_yn", "comparison_choice"],
            "Around": ["distance", "measurement"],
        },
    },
}

ALL_BENCHMARK_NAMES = list(BENCHMARKS.keys())


def get_benchmark(name):
    """Return a benchmark config by name, or raise ValueError."""
    name_lower = name.lower()
    if name_lower not in BENCHMARKS:
        valid = ", ".join(ALL_BENCHMARK_NAMES)
        raise ValueError(f"Unknown benchmark '{name}'. Valid benchmarks: {valid}")
    return BENCHMARKS[name_lower]


# ---------------------------------------------------------------------------
# LLM Judge (optional fallback)
# ---------------------------------------------------------------------------

LLM_JUDGE_SYSTEM_PROMPT = (
    "You are a judge for spatial reasoning QA tests.\n"
    "The user will provide:\n"
    "  Question: The original question.\n"
    "  Prediction: The model's predicted answer.\n"
    "  Ground Truth: The correct answer.\n"
    "Judge whether the prediction is correct. Respond with only 'True' or 'False'."
)


def llm_judge(client, model, question, prediction, ground_truth):
    """
    Use an LLM to judge if a prediction matches ground truth.

    Adapted from OmniSpatial and SpaCE-10 GPT fallback patterns.
    Returns 1.0 or 0.0.
    """
    prompt = (
        f"Question: {question}\n"
        f"Prediction: {prediction}\n"
        f"Ground Truth: {ground_truth}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0,
        )
        result = response.choices[0].message.content.strip().lower()
        return 1.0 if "true" in result else 0.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Evaluator Class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluation stage for VQASynth pipeline.

    Scores model predictions against ground truth using methods from
    SpatialScore, OmniSpatial, SpaCE-10, and MindCube.

    Supports running a single benchmark, a list of benchmarks, or "all"
    to generate a full comparative report.
    """

    def __init__(
        self,
        prediction_column="predictions",
        ground_truth_column="messages",
        use_llm_judge=False,
        llm_api_key=None,
        llm_model="gpt-4o",
        distance_tolerance=2.0,
        benchmark="all",
    ):
        """
        Args:
            prediction_column: Column name with model predictions.
            ground_truth_column: Column name with ground truth messages.
            use_llm_judge: Enable LLM judge fallback for ambiguous outputs.
            llm_api_key: OpenAI API key (required if use_llm_judge=True).
            llm_model: Model name for LLM judge calls.
            distance_tolerance: Ratio tolerance for distance scoring.
            benchmark: Benchmark name, list of names, or "all".
        """
        self.prediction_column = prediction_column
        self.ground_truth_column = ground_truth_column
        self.distance_tolerance = distance_tolerance
        self.use_llm_judge = use_llm_judge
        self.llm_model = llm_model
        self.client = None

        if use_llm_judge and llm_api_key:
            self.client = OpenAI(api_key=llm_api_key)

        # Resolve benchmark selection
        if benchmark == "all":
            self.benchmarks = ALL_BENCHMARK_NAMES
        elif isinstance(benchmark, str):
            self.benchmarks = [benchmark.lower()]
        else:
            self.benchmarks = [b.lower() for b in benchmark]

        # Validate
        for b in self.benchmarks:
            get_benchmark(b)

    def _extract_qa_pairs(self, messages):
        """
        Extract (question, answer) pairs from VQASynth message format.

        Reuses the same logic as R1Reasoner._extract_qa_pairs.
        """
        pairs = []
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]
            if current_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                question_texts = [
                    c.get("text", "")
                    for c in current_msg.get("content", [])
                    if c.get("type") == "text" and c.get("text") is not None
                ]
                answer_texts = [
                    c.get("text", "")
                    for c in next_msg.get("content", [])
                    if c.get("type") == "text" and c.get("text") is not None
                ]
                if question_texts and answer_texts:
                    pairs.append((question_texts[0], answer_texts[0]))
        return pairs

    def _apply_scorer(self, scorer_name, question, prediction, ground_truth):
        """
        Dispatch to a scoring function by name.

        Returns (score, scoring_method) tuple. Score may be None on failure.
        """
        if scorer_name == "ratio_tolerance":
            return (
                score_distance(prediction, ground_truth, self.distance_tolerance),
                "ratio_tolerance",
            )
        elif scorer_name == "mra":
            return (
                score_distance_mra(prediction, ground_truth),
                "mra",
            )
        elif scorer_name == "yes_no":
            return (
                score_yes_no(prediction, ground_truth),
                "yes_no_match",
            )
        elif scorer_name == "choice":
            return (
                score_choice(prediction, ground_truth, question),
                "choice_match",
            )
        elif scorer_name == "llm_judge":
            if self.use_llm_judge and self.client is not None:
                return (
                    llm_judge(self.client, self.llm_model, question, prediction, ground_truth),
                    "llm_judge",
                )
            return (None, "llm_judge_unavailable")
        return (None, "none")

    def score_single(self, question, prediction, ground_truth, benchmark_name=None):
        """
        Score a single prediction against ground truth.

        If benchmark_name is provided, uses that benchmark's scoring strategy.
        Otherwise uses the first configured benchmark.

        Returns dict with "qa_type", "score", and "scoring_method".
        """
        qa_type = classify_question(question)
        bname = benchmark_name or self.benchmarks[0]
        bench = get_benchmark(bname)
        scorer_name = bench["scorer"].get(qa_type, bench["scorer"].get("unknown", "llm_judge"))

        score, scoring_method = self._apply_scorer(scorer_name, question, prediction, ground_truth)

        # Fallback chain: try rule-based if LLM judge failed or was unavailable
        if score is None and scorer_name == "llm_judge":
            if qa_type in ("distance", "vertical_distance", "horizontal_distance", "measurement"):
                score, scoring_method = self._apply_scorer("ratio_tolerance", question, prediction, ground_truth)
            elif qa_type == "comparison_yn":
                score, scoring_method = self._apply_scorer("yes_no", question, prediction, ground_truth)
            elif qa_type == "comparison_choice":
                score, scoring_method = self._apply_scorer("choice", question, prediction, ground_truth)

        # Final fallback: try LLM judge if rule-based failed
        if score is None and self.use_llm_judge and self.client is not None and scorer_name != "llm_judge":
            score, scoring_method = self._apply_scorer("llm_judge", question, prediction, ground_truth)

        if score is None:
            score = 0.0
            scoring_method = "fallback_zero"

        return {
            "qa_type": qa_type,
            "score": score,
            "scoring_method": scoring_method,
        }

    def _aggregate_scores(self, per_sample, benchmark_name):
        """Build aggregate stats for a single benchmark from per-sample results."""
        bench = get_benchmark(benchmark_name)
        all_scores = [r["score"] for r in per_sample]

        # Per qa_type breakdown
        type_scores = defaultdict(list)
        for r in per_sample:
            type_scores[r["qa_type"]].append(r["score"])

        by_type = {
            t: {"accuracy": sum(s) / len(s), "count": len(s)}
            for t, s in type_scores.items()
        }

        # Per benchmark category breakdown
        by_category = {}
        for cat_name, cat_types in bench["categories"].items():
            cat_scores = []
            for t in cat_types:
                cat_scores.extend(type_scores.get(t, []))
            if cat_scores:
                by_category[cat_name] = {
                    "accuracy": sum(cat_scores) / len(cat_scores),
                    "count": len(cat_scores),
                }

        return {
            "benchmark": bench["name"],
            "description": bench["description"],
            "overall_accuracy": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "total_qa_pairs": len(all_scores),
            "by_type": by_type,
            "by_category": by_category,
        }

    def run(self, predictions, messages, benchmark_name=None):
        """
        Score predictions against ground truth for a single benchmark.

        Args:
            predictions: List of prediction strings (one per QA pair).
            messages: VQASynth messages list (alternating user/assistant).
            benchmark_name: Benchmark to use. Defaults to first configured.

        Returns dict with "per_sample" scores and "aggregate" summary.
        """
        bname = benchmark_name or self.benchmarks[0]
        qa_pairs = self._extract_qa_pairs(messages)

        per_sample = []
        for i, (question, gt_answer) in enumerate(qa_pairs):
            pred = predictions[i] if i < len(predictions) else ""
            result = self.score_single(question, pred, gt_answer, bname)
            per_sample.append(result)

        aggregate = self._aggregate_scores(per_sample, bname)
        return {"per_sample": per_sample, "aggregate": aggregate}

    def generate_report(self, predictions, messages):
        """
        Generate a full evaluation report across all configured benchmarks.

        Runs each benchmark's scoring strategy and produces a comparative
        report with per-benchmark and per-category breakdowns.

        Returns dict with structure:
        {
            "summary": { benchmark_name: overall_accuracy, ... },
            "benchmarks": {
                benchmark_name: {
                    "overall_accuracy": float,
                    "by_type": { qa_type: {accuracy, count}, ... },
                    "by_category": { category: {accuracy, count}, ... },
                },
                ...
            },
            "total_qa_pairs": int,
        }
        """
        qa_pairs = self._extract_qa_pairs(messages)
        if not qa_pairs:
            return {
                "summary": {},
                "benchmarks": {},
                "total_qa_pairs": 0,
            }

        report = {"benchmarks": {}, "summary": {}, "total_qa_pairs": len(qa_pairs)}

        for bname in self.benchmarks:
            per_sample = []
            for i, (question, gt_answer) in enumerate(qa_pairs):
                pred = predictions[i] if i < len(predictions) else ""
                result = self.score_single(question, pred, gt_answer, bname)
                per_sample.append(result)

            agg = self._aggregate_scores(per_sample, bname)
            bench_display = agg.pop("benchmark")
            report["benchmarks"][bench_display] = agg
            report["summary"][bench_display] = agg["overall_accuracy"]

        return report

    def apply_transform(self, example):
        """
        Pipeline-compatible transform for dataset.map().

        Reads messages and predictions columns. When a single benchmark is
        configured, adds eval_scores/eval_types/eval_mean_score. When
        multiple benchmarks are configured, generates a full report in
        eval_report and stores per-benchmark scores.
        """
        messages = example.get(self.ground_truth_column)
        predictions = example.get(self.prediction_column)

        empty = {
            "eval_scores": [],
            "eval_types": [],
            "eval_mean_score": None,
            "eval_report": None,
        }

        if not messages:
            return empty

        qa_pairs = self._extract_qa_pairs(messages)
        if not qa_pairs:
            return empty

        # Single benchmark mode: flat scores
        if len(self.benchmarks) == 1:
            bname = self.benchmarks[0]
            scores = []
            types = []
            for i, (question, gt_answer) in enumerate(qa_pairs):
                pred = predictions[i] if predictions and i < len(predictions) else ""
                result = self.score_single(question, pred, gt_answer, bname)
                scores.append(result["score"])
                types.append(result["qa_type"])

            mean_score = sum(scores) / len(scores) if scores else None
            return {
                "eval_scores": scores,
                "eval_types": types,
                "eval_mean_score": mean_score,
                "eval_report": None,
            }

        # Multi-benchmark mode: full report
        preds_list = [
            predictions[i] if predictions and i < len(predictions) else ""
            for i in range(len(qa_pairs))
        ]
        report = self.generate_report(preds_list, messages)

        # Use first benchmark's scores as the primary eval_scores
        first_bname = self.benchmarks[0]
        first_scores = []
        first_types = []
        for i, (question, gt_answer) in enumerate(qa_pairs):
            pred = preds_list[i]
            result = self.score_single(question, pred, gt_answer, first_bname)
            first_scores.append(result["score"])
            first_types.append(result["qa_type"])

        mean_score = sum(first_scores) / len(first_scores) if first_scores else None

        return {
            "eval_scores": first_scores,
            "eval_types": first_types,
            "eval_mean_score": mean_score,
            "eval_report": json.dumps(report),
        }
