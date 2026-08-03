"""
Human pose estimation for VQASynth — data-generation direction.

Produces SFT training samples that teach a VLM (Molmo) to emit
``<point x=.. y=.. alt=..>`` tags for body keypoints — the PoseText task
(issue #31, salma-remyx/PoseText). Molmo is the *target* (student) of this
distillation, **not** the source of the keypoints.

Design (keypoint-source-first)
------------------------------
Keypoints come from a lightweight pose model or an annotated dataset, then
the emitter renders them as Molmo-style answers. The pipeline:

  1. Run a pluggable backend over an image — default MediaPipe Pose
     (CPU-friendly, Apache-2.0, 33 body landmarks) — or read a
     pre-annotated dataset (COCO Keypoints / MPII).
  2. Extract 2D pixel coordinates per keypoint per detected person.
  3. Format each keypoint set as SFT training samples (user: image + a
     question; assistant: a Molmo-style ``<point>`` response carrying the
     ground-truth pixel coordinates).
  4. Emit the samples as a HF dataset shard (see ``docker/pose_stage``),
     appending a ``pose_messages`` column like the other stages.

Pluggable backend
-----------------
:class:`KeypointExtractor` accepts any backend exposing
``extract(image) -> (people, image_size)`` (see :class:`PoseBackend`), where
each entry in ``people`` is a per-person COCO-17 keypoint array. The default
is :class:`MediaPipePoseBackend`; :class:`StubPoseBackend` is a deterministic,
dependency-free implementation used by the tests. Adding a backend (e.g. a
YOLOv8-pose wrapper, or a COCO/MPII dataset reader) is a short subclass.

Reuse, not duplication
----------------------
  * The QA-pair emitter (:func:`build_pose_qa_pairs`) renders Molmo-style
    ``<point>`` answers and phrases relative-position questions through the
    existing :mod:`vqasynth.prompt_templates` spatial-predicate corpus, so
    pose QA reads exactly like the rest of a VQASynth dataset.
  * The chat-message shape (:func:`build_pose_messages`) mirrors the nested
    ``messages`` column convention :mod:`vqasynth.prompts` writes.

No GPU, model download, or mediapipe/ultralytics install is required to
import this module or to run its pure-Python QA / message logic — that is
what the tests cover. Real end-to-end pose inference belongs on a host with
the chosen backend installed.

Licensing note: MediaPipe is Apache-2.0 and is the default backend.
YOLOv8-pose (ultralytics) is AGPL-3.0 and is **opt-in only** — it is never
selected by name and is not installed by the pose stage; users who accept
that license shape can pass a custom backend instance to
:class:`KeypointExtractor`.
"""
from __future__ import annotations

import itertools
import random

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

# MediaPipe Pose emits 33 landmarks; this maps each COCO joint (in COCO-17
# dataset order) onto its corresponding MediaPipe landmark index. Used to
# project MediaPipe's 33-landmark output down to the 17-name COCO skeleton
# the QA emitter and downstream training expect.
MEDIAPIPE_TO_COCO = [
    0,   # nose          -> mp nose
    2,   # left_eye      -> mp left eye
    5,   # right_eye     -> mp right eye
    7,   # left_ear      -> mp left ear
    8,   # right_ear     -> mp right ear
    11,  # left_shoulder -> mp left shoulder
    12,  # right_shoulder-> mp right shoulder
    13,  # left_elbow    -> mp left elbow
    14,  # right_elbow   -> mp right elbow
    15,  # left_wrist    -> mp left wrist
    16,  # right_wrist   -> mp right wrist
    23,  # left_hip      -> mp left hip
    24,  # right_hip     -> mp right hip
    25,  # left_knee     -> mp left knee
    26,  # right_knee    -> mp right knee
    27,  # left_ankle    -> mp left ankle
    28,  # right_ankle   -> mp right ankle
]


# ---------------------------------------------------------------------------
# Pose dict construction (backend output -> canonical pose)
# ---------------------------------------------------------------------------
def pose_from_keypoints(keypoints, image_size):
    """Build the canonical COCO-17 pose dict from raw per-joint pixel coords.

    Args:
        keypoints: per-joint pixel coordinates in COCO-17 order. Either:

            * an iterable of 17 entries, each an ``(x, y)`` / ``[x, y]``
              pixel coordinate, or ``None`` if the joint was not detected; or
            * a mapping keyed by canonical name *or* 0-indexed COCO index,
              with missing joints treated as not visible.

        image_size: ``(width, height)`` of the source image, in pixels.

    Returns the pose dict shape :func:`build_pose_qa_pairs` consumes::

        {
          "keypoints": [ {index, name, xy, visible}, ... x17 ],
          "image_size": (w, h),
          "num_detected": int,
        }
    """
    if isinstance(keypoints, dict):
        resolved = [None] * len(COCO_KEYPOINTS)
        for idx, name in enumerate(COCO_KEYPOINTS):
            value = keypoints.get(idx, keypoints.get(name))
            if value is not None:
                resolved[idx] = list(value)
        keypoints = resolved

    out = []
    detected = 0
    for idx, name in enumerate(COCO_KEYPOINTS):
        xy = keypoints[idx] if idx < len(keypoints) else None
        if xy is not None:
            xy = [float(xy[0]), float(xy[1])]
            detected += 1
        out.append({"index": idx, "name": name, "xy": xy, "visible": xy is not None})
    return {"keypoints": out, "image_size": image_size, "num_detected": detected}


# ---------------------------------------------------------------------------
# Pluggable keypoint backends
# ---------------------------------------------------------------------------
class PoseBackend:
    """Interface for a keypoint-detection backend.

    Subclasses implement :meth:`extract` to return the raw per-person
    keypoint arrays for an image; :class:`KeypointExtractor` normalizes them
    into canonical pose dicts. A new backend only has to map its native
    keypoint format onto the 17-name COCO skeleton — see
    :class:`MediaPipePoseBackend` and :class:`StubPoseBackend`.
    """

    def extract(self, image):
        """Run the backend over ``image`` (a :class:`PIL.Image.Image`).

        Returns ``(people, image_size)`` where ``image_size`` is
        ``(width, height)`` in pixels and ``people`` is a list of per-person
        keypoint arrays in COCO-17 order — each an iterable of 17 ``(x, y)``
        pixel coordinates or ``None``.
        """
        raise NotImplementedError


class StubPoseBackend(PoseBackend):
    """Deterministic, dependency-free backend for tests and the CPU demo.

    Always returns ``keypoints`` for a single detected person regardless of
    the input image, so QA / message logic can be exercised without a pose
    model installed. ``keypoints`` follows the same format as
    :func:`pose_from_keypoints`.
    """

    def __init__(self, keypoints, image_size):
        # Normalize to the per-person list form the backend contract expects
        # (handles dict fixtures too, via pose_from_keypoints).
        pose = pose_from_keypoints(keypoints, image_size)
        self._person = [k["xy"] for k in pose["keypoints"]]  # 17 entries, [x,y] or None
        self._image_size = image_size

    def extract(self, image):
        # Return a shallow copy so callers can't mutate the fixture.
        return [list(self._person)], self._image_size


class MediaPipePoseBackend(PoseBackend):
    """Default backend: MediaPipe Pose (CPU-friendly, Apache-2.0).

    MediaPipe emits 33 body landmarks per person; we project them onto the
    17-name COCO skeleton via :data:`MEDIAPIPE_TO_COCO`. ``mediapipe`` is
    imported lazily on construction, so importing this module (and running
    the test suite) needs no pose-model install.
    """

    def __init__(self, model_complexity=1, min_detection_confidence=0.5,
                 min_visibility=0.5):
        import mediapipe as mp  # lazy: heavy, optional dep
        self._min_visibility = min_visibility
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )

    def extract(self, image):
        import numpy as np  # local import; numpy is a core vqasynth dep
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        result = self._pose.process(np.asarray(image))

        # The legacy solutions.pose API returns one pose (most-prominent);
        # newer Tasks builds expose pose_landmarks_list with N persons.
        if getattr(result, "pose_landmarks_list", None):
            landmark_sets = result.pose_landmarks_list
        elif result.pose_landmarks is not None:
            landmark_sets = [result.pose_landmarks]
        else:
            landmark_sets = []

        people = []
        for landmarks in landmark_sets:
            kps = [None] * len(COCO_KEYPOINTS)
            for coco_idx, mp_idx in enumerate(MEDIAPIPE_TO_COCO):
                if mp_idx >= len(landmarks.landmark):
                    continue
                lm = landmarks.landmark[mp_idx]
                # MediaPipe coords are normalized to [0, 1]; visibility is the
                # probability the joint is unoccluded — drop low ones.
                if getattr(lm, "visibility", 1.0) < self._min_visibility:
                    continue
                kps[coco_idx] = [lm.x * width, lm.y * height]
            people.append(kps)
        return people, (width, height)

    def close(self):
        self._pose.close()


def _resolve_backend(backend):
    """Resolve a backend argument to a :class:`PoseBackend` instance."""
    if isinstance(backend, str):
        name = backend.lower()
        if name == "mediapipe":
            return MediaPipePoseBackend()
        if name in ("yolov8", "ultralytics"):
            raise ValueError(
                "The ultralytics (YOLOv8-pose) backend is AGPL-3.0 and is "
                "opt-in only — it is never selected by name. Install "
                "ultralytics and pass a backend instance to KeypointExtractor "
                "directly if you accept that license."
            )
        raise ValueError(f"Unknown pose backend: {backend!r}")
    return backend


class KeypointExtractor:
    """Run a pluggable pose backend and emit per-person COCO-17 pose dicts.

    Args:
        backend: a :class:`PoseBackend` instance, or a backend name string.
            ``"mediapipe"`` (the default) selects :class:`MediaPipePoseBackend`.
            Pass a custom instance (e.g. :class:`StubPoseBackend`) for tests
            or to wrap an annotated-dataset reader.

    The backend's heavy dependencies are imported lazily when the named
    backend is constructed, so importing this module stays cheap and tests
    need no pose-model install.
    """

    def __init__(self, backend="mediapipe"):
        self.backend = _resolve_backend(backend)

    def extract(self, image):
        """Return one canonical pose dict (see :func:`pose_from_keypoints`)
        per person detected in ``image``."""
        people, image_size = self.backend.extract(image)
        return [pose_from_keypoints(person, image_size) for person in people]

    def extract_messages(self, image, max_questions=12, seed=None):
        """Convenience: keypoints -> SFT chat messages for every person.

        Returns a flat list of chat-message samples (see
        :func:`build_pose_messages`), spanning every detected person.
        """
        samples = []
        for pose in self.extract(image):
            samples.extend(
                build_pose_messages(pose, max_questions=max_questions, seed=seed)
            )
        return samples


# ---------------------------------------------------------------------------
# Pose QA generation (PoseText-style, for fine-tuning Molmo)
# ---------------------------------------------------------------------------
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


# relation -> (predicate questions, true responses, false responses)
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
    """Generate PoseText-style (question, answer) pairs from a pose dict.

    Three families, all directly usable for instruction-tuning a VLM to emit
    Molmo ``<point>`` tags for body keypoints:

      * per-keypoint point localization (answer is a Molmo ``<point>`` token);
      * whole-pose localization (one ``<point>`` per visible joint);
      * relative-position predicates between joint pairs, phrased with the
        existing :mod:`vqasynth.prompt_templates` corpus so pose QA is
        stylistically consistent with the rest of a VQASynth dataset.

    Args:
        pose: output of :func:`pose_from_keypoints` / :meth:`KeypointExtractor.extract`.
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
            questions, true_responses, _false_responses = templates
            disp_a, disp_b = _disp(kp_a["name"]), _disp(kp_b["name"])
            question = _substitute(rng.choice(questions), disp_a, disp_b)
            answer = _substitute(rng.choice(true_responses), disp_a, disp_b)
            add(question, answer)
            if len(qa) >= max_questions:
                break

    return qa


def build_pose_messages(pose, max_questions=12, seed=None):
    """Turn a pose dict into a list of SFT chat-message samples.

    Each sample is ``{"messages": [...]}`` in the nested role/content
    structure :mod:`vqasynth.prompts` writes to the ``messages`` column: the
    user turn carries an image placeholder (``index: 0``) followed by the
    question, and the assistant turn carries the Molmo-style ``<point>``
    answer. One sample per (question, answer) pair from
    :func:`build_pose_qa_pairs`.
    """
    samples = []
    for qa in build_pose_qa_pairs(pose, max_questions=max_questions, seed=seed):
        samples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"index": 0, "text": None, "type": "image"},
                        {"index": None, "text": qa["question"], "type": "text"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"index": None, "text": qa["answer"], "type": "text"},
                    ],
                },
            ]
        })
    return samples


# ---------------------------------------------------------------------------
# Dataset transform (column-append convention used by the docker stages)
# ---------------------------------------------------------------------------
class PoseAnnotator:
    """Dataset transform that appends a ``pose_messages`` column.

    Mirrors the column-append convention used by the other stages (e.g.
    :class:`vqasynth.embeddings.EmbeddingGenerator`): ``apply_transform`` is
    meant to be passed to ``dataset.map(..., batched=True)`` and writes one
    list of SFT chat-message samples (see :func:`build_pose_messages`) per
    input image. Rows where pose detection yields no samples are written as
    ``None`` and dropped downstream by :func:`vqasynth.utils.filter_null`.
    """

    def __init__(self, backend="mediapipe", max_questions=12, seed=None):
        self.extractor = KeypointExtractor(backend=backend)
        self.max_questions = max_questions
        self.seed = seed

    def _samples_for(self, image):
        samples = []
        for pose in self.extractor.extract(image):
            samples.extend(
                build_pose_messages(pose, max_questions=self.max_questions, seed=self.seed)
            )
        return samples or None

    def apply_transform(self, example, images):
        """Process a single example or a batch, adding a ``pose_messages`` column."""
        from PIL import Image  # lazy: keeps the module importable without PIL
        is_batched = isinstance(example[images], list)

        try:
            if is_batched:
                results = []
                for img_item in example[images]:
                    image = img_item[0] if isinstance(img_item, list) else img_item
                    if not isinstance(image, Image.Image):
                        raise ValueError(f"Expected a PIL image but got {type(image)}")
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    results.append(self._samples_for(image))
                example["pose_messages"] = results
            else:
                image = example[images][0] if isinstance(example[images], list) else example[images]
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected a PIL image but got {type(image)}")
                if image.mode != "RGB":
                    image = image.convert("RGB")
                example["pose_messages"] = self._samples_for(image)
        except Exception as e:  # mirror the embeddings stage: skip, don't abort the batch
            print(f"Error processing image, skipping: {e}")
            example["pose_messages"] = [None] * len(example[images]) if is_batched else None

        return example
