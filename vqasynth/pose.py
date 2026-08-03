"""
Human pose estimation for VQASynth.

Adds a body-keypoint estimation stage and PoseText-style VQA generation so the
pipeline can produce instruction-tuning data for fine-tuning Molmo on body
keypoint estimation (issue #31, salma-remyx/PoseText).

Design
------
Molmo already emits ``<point x=.. y=.. alt="name">`` tokens for arbitrary
point localization (``vqasynth.localize.MolmoCaptionLocalizer``). Estimating a
human pose is a *constrained* instance of that: instead of free-form objects,
we ask Molmo for a fixed set of named body joints (the COCO-17 skeleton) and
map the names it returns onto canonical indices.

Reuse, not duplication:
  * ``parse_pose`` calls ``vqasynth.localize.extract_points_and_descriptions``
    (the same parser the object localizer uses). The import is lazy because
    ``vqasynth.localize`` pulls heavyweight model deps (sam2 / transformers /
    accelerate) at module top; in lightweight environments (tests, CPU) we fall
    back to an equivalent local parser so the module stays importable with the
    standard library alone. The model-backed ``MolmoPoseLocalizer`` likewise
    imports torch / transformers only when instantiated.
  * ``build_pose_qa_pairs`` consumes the existing spatial-predicate template
    corpus in ``vqasynth.prompt_templates`` (pure Python) for the relative
    keypoint questions, so pose QA reads exactly like the rest of the dataset.

No GPU, model download, or sam2 install is required to import this module or to
run its pure-Python parsing / QA logic — that path is what the tests cover. Real
end-to-end pose inference with Molmo belongs on a GPU host, mirroring how
``tests/test_vggt_speedups.py`` validates wrapper mechanics against fakes.
"""
from __future__ import annotations

import itertools
import random
import re

from vqasynth import prompt_templates


# ---------------------------------------------------------------------------
# COCO-17 person keypoint skeleton
# ---------------------------------------------------------------------------
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
"""Canonical 17 COCO person keypoints, in dataset order (0-indexed)."""

KEYPOINT_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# Bone edges as 0-indexed (joint_a, joint_b) pairs — the standard COCO-17
# skeleton (face, arms, torso, legs).
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # nose <-> eyes <-> ears
    (5, 7), (7, 9), (6, 8), (8, 10),           # shoulders -> elbows -> wrists
    (5, 6), (11, 12), (5, 11), (6, 12),        # shoulder girdle + hips + torso sides
    (11, 13), (13, 15), (12, 14), (14, 16),    # hips -> knees -> ankles
]


# ---------------------------------------------------------------------------
# Name normalization: free-form Molmo alt text -> canonical keypoint name
# ---------------------------------------------------------------------------
# (substring, canonical_left, canonical_right). ``None`` marks an unpaired
# joint (nose). Order matters: longer/more-specific tokens first.
_PARTS = [
    ("nose", None, None),
    ("eye", "left_eye", "right_eye"),
    ("ear", "left_ear", "right_ear"),
    ("shoulder", "left_shoulder", "right_shoulder"),
    ("elbow", "left_elbow", "right_elbow"),
    ("wrist", "left_wrist", "right_wrist"),
    ("hand", "left_wrist", "right_wrist"),     # Molmo often says "hand" for the wrist joint
    ("hip", "left_hip", "right_hip"),
    ("knee", "left_knee", "right_knee"),
    ("ankle", "left_ankle", "right_ankle"),
    ("foot", "left_ankle", "right_ankle"),     # foot -> ankle joint
]

_FILLER_RE = re.compile(
    r"\b(the|a|an|of|person|persons|person'?s|body|joint|joints|point|tip|location)\b"
)


def normalize_keypoint_name(raw):
    """Map free-form keypoint text (e.g. "person's left shoulder") to a
    canonical name in :data:`COCO_KEYPOINTS`, or ``None`` if it isn't a body
    joint. Handles case, punctuation, filler words, ``l``/``r`` shorthand, and
    common synonyms (hand->wrist, foot->ankle).
    """
    if not raw:
        return None
    text = re.sub(r"[^a-z0-9 ]", " ", raw.strip().lower())
    text = _FILLER_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    side = None
    if re.search(r"\bleft\b", text) or re.search(r"(^|\s)l(\s|$)", text):
        side = "left"
    elif re.search(r"\bright\b", text) or re.search(r"(^|\s)r(\s|$)", text):
        side = "right"

    for part, left, right in _PARTS:
        if part not in text:
            continue
        if left is None:                      # unpaired joint (nose)
            return part
        if side == "left":
            return left
        if side == "right":
            return right
        return None                           # part seen but side ambiguous
    return None


# ---------------------------------------------------------------------------
# Parsing Molmo <point> output into an ordered pose
# ---------------------------------------------------------------------------
_POINT_RE = re.compile(
    r'<point\s+x="\s*([0-9]+(?:\.[0-9]+)?)"\s+y="\s*([0-9]+(?:\.[0-9]+)?)"\s+alt="([^"]+)">'
)


def _local_extract_points(molmo_output, image_w, image_h):
    """Local copy of ``vqasynth.localize.extract_points_and_descriptions``.

    Used only when the shared parser can't be imported (sam2 / transformers not
    installed). Kept byte-for-byte equivalent in behaviour so both paths agree.
    """
    results = []
    for match in _POINT_RE.finditer(molmo_output):
        try:
            x_norm = float(match.group(1))
            y_norm = float(match.group(2))
            description = match.group(3)
        except ValueError:
            continue
        if max(x_norm, y_norm) > 100:
            continue
        x_pixel = (x_norm / 100.0) * image_w
        y_pixel = (y_norm / 100.0) * image_h
        results.append({"points": [x_pixel, y_pixel], "caption": description})
    return results


def _extract_points(molmo_output, image_w, image_h):
    """Prefer the shared object-localizer parser; fall back to the local copy."""
    try:
        from vqasynth.localize import extract_points_and_descriptions
    except Exception:
        extract_points_and_descriptions = _local_extract_points
    return extract_points_and_descriptions(molmo_output, image_w, image_h)


def parse_pose(molmo_output, image_w, image_h):
    """Parse Molmo ``<point>`` output into an ordered COCO-17 pose.

    Each joint is reported once (first detection wins). Coordinates are in
    image pixels; undetected joints are reported with ``xy=None`` /
    ``visible=False`` (the COCO convention for "not labeled").

    Returns::

        {
          "keypoints": [ {index, name, xy, visible}, ... x17 ],
          "image_size": (image_w, image_h),
          "num_detected": int,
        }
    """
    detected = {}  # canonical index -> [x_pixel, y_pixel]
    for entry in _extract_points(molmo_output, image_w, image_h):
        name = normalize_keypoint_name(entry.get("caption", ""))
        if name is None:
            continue
        idx = KEYPOINT_INDEX[name]
        if idx not in detected:               # first detection wins
            detected[idx] = list(entry["points"])

    keypoints = []
    for idx, name in enumerate(COCO_KEYPOINTS):
        xy = detected.get(idx)
        keypoints.append({
            "index": idx,
            "name": name,
            "xy": xy,
            "visible": xy is not None,
        })
    return {
        "keypoints": keypoints,
        "image_size": (image_w, image_h),
        "num_detected": sum(1 for k in keypoints if k["visible"]),
    }


# ---------------------------------------------------------------------------
# Pose QA generation (PoseText-style, for fine-tuning Molmo)
# ---------------------------------------------------------------------------
def build_pose_prompt(keypoint_names=None):
    """Prompt asking Molmo to emit one ``<point>`` per visible body joint,
    reusing the normalized 0-100 coordinate convention from
    :class:`vqasynth.localize.MolmoCaptionLocalizer`.
    """
    joints = [n.replace("_", " ") for n in (keypoint_names or COCO_KEYPOINTS)]
    listed = ", ".join(joints)
    return (
        "You are an AI assistant that estimates human body keypoints. "
        "For the person in the image, locate each visible body joint and output "
        'one <point> element per joint in this format: '
        '<point x="X" y="Y" alt="joint name"/>. '
        f"The joints are: {listed}. "
        'Use the joint name exactly as listed for the alt text. '
        "Use normalized coordinates from 0 to 100. "
        "Only output points for joints that are visible; omit any you cannot see. "
        "Only provide valid points in the specified format."
    )


_SINGLE_KP_QUESTIONS = [
    "Point to the {kp} of the person.",
    "Where is the person's {kp}?",
    "Locate the {kp} of the person.",
    "Identify the position of the {kp}.",
]
_SINGLE_KP_ANSWERS = [
    "{pt}",
    "The {kp} is at {pt}.",
    "The person's {kp} is located at {pt}.",
]
_ALL_KP_QUESTIONS = [
    "Locate all visible body keypoints of the person.",
    "Estimate the body keypoints of the person.",
    "Where are the person's joints? Give a point for each visible joint.",
]


def _disp(name):
    """Display form of a canonical keypoint name (``left_shoulder`` -> ``left shoulder``)."""
    return name.replace("_", " ")


def _point_token(xy, image_w, image_h, name=None):
    """Molmo ``<point>`` token (normalized 0-100) for a pixel coordinate."""
    x = round((xy[0] / image_w) * 100.0, 1)
    y = round((xy[1] / image_h) * 100.0, 1)
    alt = f' alt="{name}"' if name else ""
    return f'<point x="{x}" y="{y}"{alt}/>'


def _spatial_relation(a_xy, b_xy):
    """Coarse relation of joint A vs joint B in image space.

    Image y grows downward, so smaller y == higher. Returns one of
    ``left`` / ``right`` / ``above`` / ``below`` / ``same``.
    """
    dx = a_xy[0] - b_xy[0]
    dy = a_xy[1] - b_xy[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "same"
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "above" if dy < 0 else "below"


# relation -> (predicate questions, true responses, false responses, affirmative flag)
_RELATION_TEMPLATES = {
    "left": (prompt_templates.left_predicate_questions,
             prompt_templates.left_true_responses,
             prompt_templates.left_false_responses),
    "right": (prompt_templates.right_predicate_questions,
              prompt_templates.right_true_responses,
              prompt_templates.right_false_responses),
    "above": (prompt_templates.above_predicate_questions,
              prompt_templates.above_true_responses,
              prompt_templates.above_false_responses),
    "below": (prompt_templates.below_predicate_questions,
              prompt_templates.below_true_responses,
              prompt_templates.below_false_responses),
}


def _substitute(template, a, b):
    return template.replace("[A]", a).replace("[B]", b)


def build_pose_qa_pairs(pose, max_questions=12, seed=None):
    """Generate PoseText-style (question, answer) pairs from a parsed pose.

    Three families, all in forms directly usable for instruction-tuning Molmo:

      * per-keypoint point localization (answer is a Molmo ``<point>`` token);
      * whole-pose localization (one ``<point>`` per visible joint);
      * relative-position predicates between joint pairs, phrased with the
        existing :mod:`vqasynth.prompt_templates` corpus so pose QA is
        stylistically consistent with the rest of a VQASynth dataset.

    Args:
        pose: output of :func:`parse_pose`.
        max_questions: cap on the number of pairs returned.
        seed: optional int for deterministic output (tests).

    Returns a list of ``{"question": str, "answer": str}`` dicts.
    """
    rng = random.Random(seed) if seed is not None else random
    image_w, image_h = pose["image_size"]
    visible = [k for k in pose["keypoints"] if k["visible"]]
    qa = []

    def add(question, answer):
        if len(qa) < max_questions:
            qa.append({"question": question, "answer": answer})

    # 1) per-keypoint point localization
    for kp in visible:
        disp = _disp(kp["name"])
        token = _point_token(kp["xy"], image_w, image_h, disp)
        question = rng.choice(_SINGLE_KP_QUESTIONS).format(kp=disp)
        answer = rng.choice(_SINGLE_KP_ANSWERS).format(kp=disp, pt=token)
        add(question, answer)

    # 2) whole-pose localization
    if visible:
        tokens = " ".join(
            _point_token(kp["xy"], image_w, image_h, _disp(kp["name"]))
            for kp in visible
        )
        add(rng.choice(_ALL_KP_QUESTIONS), tokens)

    # 3) relative-position predicates between joint pairs
    if len(visible) >= 2:
        pairs = list(itertools.combinations(visible, 2))
        rng.shuffle(pairs)
        for kp_a, kp_b in pairs:
            relation = _spatial_relation(kp_a["xy"], kp_b["xy"])
            templates = _RELATION_TEMPLATES.get(relation)
            if templates is None:            # "same" — skip, no clean predicate
                continue
            questions, true_responses, false_responses = templates
            disp_a, disp_b = _disp(kp_a["name"]), _disp(kp_b["name"])
            question = _substitute(rng.choice(questions), disp_a, disp_b)
            answer = _substitute(rng.choice(true_responses), disp_a, disp_b)
            add(question, answer)
            if len(qa) >= max_questions:
                break

    return qa


# ---------------------------------------------------------------------------
# Model-backed estimator (heavy deps imported lazily on instantiation)
# ---------------------------------------------------------------------------
class MolmoPoseLocalizer:
    """Estimate COCO-17 body keypoints by prompting Molmo for named joints.

    Wraps :class:`vqasynth.localize.MolmoCaptionLocalizer` (same 4-bit Molmo
    load path used for object localization) and parses its output with
    :func:`parse_pose`. Torch / transformers / bitsandbytes are imported only
    when the estimator is constructed, so importing this module stays cheap.
    """

    def __init__(self, model_name="cyan2k/molmo-7B-O-bnb-4bit", device=None):
        from vqasynth.localize import MolmoCaptionLocalizer
        self.model_name = model_name
        self._localizer = MolmoCaptionLocalizer(model_name=model_name, device=device)

    @property
    def device(self):
        return self._localizer.device

    @property
    def processor(self):
        return self._localizer.processor

    def run(self, image):
        """Run Molmo with the pose prompt and return the parsed pose dict."""
        import torch
        prompt = build_pose_prompt()
        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {
            k: v.to(self._localizer.device).unsqueeze(0)
            for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        generated_ids = self._localizer.generate_ids(inputs)
        text = self.processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        image_w, image_h = image.size
        return parse_pose(text, image_w, image_h)
