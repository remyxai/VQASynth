"""
Procedurally generated fine-grained grounding tasks.

Adapted from "PGT: Procedurally Generated Tasks for improving visual grounding
in MLLMs" (arXiv:2605.23883). PGT overlays unambiguous geometric primitives on
images to produce dense supervision that disentangles a model's *visual
grounding* ability from its *semantic priors*, and doubles as a low-cost
diagnostic for locating perception failures.

This module ports the data-generation / diagnostic half of PGT (the part that
needs no trainer or checkpoint, only code). It procedurally renders scenes of
colored shapes with known geometry, phrases questions about their relative
positions / counts by reusing VQASynth's existing prompt templates, and emits
items in the same normalized schema as the external benchmark loaders in
``vqasynth.benchmarks`` -- so they score through the very same machinery.

Scoring reuses the answer extractors / scorers in ``vqasynth.evaluation``.
"""

import random
from dataclasses import dataclass

from vqasynth.evaluation import extract_number, score_choice, score_yes_no
from vqasynth.prompt_templates import (
    above_choice_questions,
    above_choice_responses,
    above_predicate_questions,
    below_choice_questions,
    below_choice_responses,
    below_predicate_questions,
    left_choice_questions,
    left_choice_responses,
    left_predicate_questions,
    right_choice_questions,
    right_choice_responses,
    right_predicate_questions,
)

COLORS = ["red", "green", "blue", "yellow", "purple", "orange"]
SHAPES = ["circle", "square", "triangle"]

# relation -> question template list reused from prompt_templates
_RELATION_QUESTIONS = {
    "left": left_predicate_questions,
    "right": right_predicate_questions,
    "above": above_predicate_questions,
    "below": below_predicate_questions,
}
_RELATION_CHOICE = {
    "left": (left_choice_questions, left_choice_responses),
    "right": (right_choice_questions, right_choice_responses),
    "above": (above_choice_questions, above_choice_responses),
    "below": (below_choice_questions, below_choice_responses),
}
# Phrasing used to build unambiguous yes/no ground-truth answers.
_RELATION_PHRASE = {
    "left": "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
}


@dataclass
class Primitive:
    """A single geometric primitive placed in image (pixel) coordinates."""

    shape: str
    color: str
    cx: float
    cy: float
    size: float

    @property
    def description(self):
        return f"{self.color} {self.shape}"


def render_primitives(primitives, image_size=384, background="white"):
    """Render primitives onto a blank canvas, returning a PIL ``Image``.

    PIL is imported lazily so the rest of this module (sampling, question
    generation, scoring) stays importable in environments without Pillow.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (image_size, image_size), background)
    draw = ImageDraw.Draw(img)
    for p in primitives:
        box = [p.cx - p.size, p.cy - p.size, p.cx + p.size, p.cy + p.size]
        if p.shape == "circle":
            draw.ellipse(box, fill=p.color, outline="black")
        elif p.shape == "square":
            draw.rectangle(box, fill=p.color, outline="black")
        elif p.shape == "triangle":
            draw.polygon(
                [(p.cx, box[1]), (box[0], box[3]), (box[2], box[3])],
                fill=p.color,
                outline="black",
            )
    return img


def _sample_scene(rng, image_size, min_objs, max_objs):
    """Sample non-overlapping primitives with distinct (color, shape) names.

    Each primitive lands in its own cell of a 3x3 grid, so any pair is clearly
    separated -- the geometry (and thus the ground truth) is unambiguous, which
    is the entire point of PGT.
    """
    grid = 3
    n = min(rng.randint(min_objs, max_objs), grid * grid)

    combos = [(c, s) for c in COLORS for s in SHAPES]
    rng.shuffle(combos)
    cells = list(range(grid * grid))
    rng.shuffle(cells)

    cell = image_size / grid
    size = cell * 0.28
    jitter = cell * 0.12

    prims = []
    for (color, shape), idx in zip(combos[:n], cells[:n]):
        row, col = divmod(idx, grid)
        cx = col * cell + cell / 2 + rng.uniform(-jitter, jitter)
        cy = row * cell + cell / 2 + rng.uniform(-jitter, jitter)
        prims.append(Primitive(shape, color, cx, cy, size))
    return prims


def _dominant_axis(a, b):
    """Return whether the pair is more separated horizontally than vertically."""
    return abs(a.cx - b.cx) >= abs(a.cy - b.cy)


def _make_relation_yn(rng, scene):
    """Yes/no predicate along the pair's most-separated axis (always decidable)."""
    a, b = rng.sample(scene, 2)
    if _dominant_axis(a, b):
        relation = rng.choice(["left", "right"])
        truth = a.cx < b.cx if relation == "left" else a.cx > b.cx
    else:
        relation = rng.choice(["above", "below"])
        truth = a.cy < b.cy if relation == "above" else a.cy > b.cy

    question = (
        rng.choice(_RELATION_QUESTIONS[relation])
        .replace("[A]", a.description)
        .replace("[B]", b.description)
    )
    phrase = _RELATION_PHRASE[relation]
    if truth:
        answer = f"Yes, the {a.description} is {phrase} the {b.description}."
    else:
        answer = f"No, the {a.description} is not {phrase} the {b.description}."
    return question, answer, relation, "judgment"


def _make_relation_choice(rng, scene):
    """'Which is more to the left/above ...' between two primitives."""
    a, b = rng.sample(scene, 2)
    if _dominant_axis(a, b):
        relation = rng.choice(["left", "right"])
        if relation == "left":
            correct = a if a.cx < b.cx else b
        else:
            correct = a if a.cx > b.cx else b
    else:
        relation = rng.choice(["above", "below"])
        if relation == "above":
            correct = a if a.cy < b.cy else b
        else:
            correct = a if a.cy > b.cy else b

    q_templates, a_templates = _RELATION_CHOICE[relation]
    question = (
        rng.choice(q_templates)
        .replace("[A]", a.description)
        .replace("[B]", b.description)
    )
    answer = rng.choice(a_templates).replace("[X]", correct.description)
    return question, answer, relation, "comparison"


def _make_count(rng, scene):
    """Count primitives of a sampled shape -- a quantitative grounding probe."""
    shape = rng.choice([p.shape for p in scene])
    count = sum(1 for p in scene if p.shape == shape)
    question = f"How many {shape}s are in the image?"
    answer = f"There are {count} {shape}s in the image."
    return question, answer, "count", "open-ended"


_TASK_BUILDERS = [
    ("relation_yn", _make_relation_yn, "Relational"),
    ("relation_choice", _make_relation_choice, "Relational"),
    ("count", _make_count, "Quantitative"),
]


def generate_pgt_items(
    num_items=60, seed=0, image_size=384, min_objs=2, max_objs=4, render=True
):
    """Generate PGT diagnostic items in the ``vqasynth.benchmarks`` schema.

    Items cycle through relational (yes/no), relational (choice) and
    quantitative (count) probes. Each carries a rendered overlay image and a
    ground-truth answer derived from the scene geometry.
    """
    rng = random.Random(seed)
    items = []
    for i in range(num_items):
        scene = _sample_scene(rng, image_size, min_objs, max_objs)
        qa_type, builder, category = _TASK_BUILDERS[i % len(_TASK_BUILDERS)]
        question, answer, subcategory, question_type = builder(rng, scene)

        images = []
        if render:
            try:
                images = [render_primitives(scene, image_size)]
            except Exception:
                # Pillow missing or render failure: keep the item, drop the image.
                images = []

        items.append(
            {
                "id": f"pgt_{i}",
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "category": category,
                "subcategory": subcategory,
                "options": [],
                "images": images,
                "source": "PGT",
                "qa_type": qa_type,
                "primitives": [p.description for p in scene],
            }
        )
    return items


def score_pgt_item(item, prediction):
    """Score a model prediction for one PGT item via ``vqasynth.evaluation``.

    Returns 1.0 / 0.0, or None when the prediction cannot be parsed (so the
    caller can route to its LLM-judge fallback, matching the other benchmarks).
    """
    qa_type = item.get("qa_type")
    gt = item["answer"]

    if qa_type == "relation_yn":
        return score_yes_no(prediction, gt)
    if qa_type == "relation_choice":
        return score_choice(prediction, gt, item.get("question", ""))
    if qa_type == "count":
        pred_val = extract_number(prediction)
        gt_val = extract_number(gt)
        if pred_val is None or gt_val is None:
            return None
        return 1.0 if pred_val == gt_val else 0.0
    return None
