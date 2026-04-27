"""
Scoring primitives for spatial reasoning benchmarks.

Implements answer extractors and scoring functions adapted from:
- SpatialScore: answer extraction, distance ratio tolerance, Mean Relative Accuracy
- OmniSpatial: multi-choice accuracy, LLM judge fallback, per-category breakdown
- SpaCE-10: cascading rule-based extraction with GPT fallback
- MindCube: multi-choice extraction with cascading regex

These primitives are consumed by the BenchmarkRunner in vqasynth.benchmarks,
which scores model predictions against external benchmark datasets.
"""

import re
import numpy as np


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
    # VQASynth dataset convention: "Actually, ..." precedes a contradiction of
    # the question's premise, i.e. the answer is "no".
    if first_word == "actually":
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


