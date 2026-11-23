"""Camera-object relation QA-pair synthesis (Ultimate3D taxonomy).

Ultimate3D ("Teaching Multimodal LLMs 3D Camera-Object Relations by
Large-Scale Visual Instruction Data", arXiv:2611.24795) shows that MLLMs
systematically miss three camera-object relations: which side of an object
faces the camera (orientation), where the camera sits relative to the
object's height (viewpoint), and how tightly the object is framed (shot).
The paper's precise labels come from rendering 3D assets under controlled
cameras; this stage recovers the same 8 / 3 / 3 categorical labels from the
signals this pipeline already produces per object, so every fused scene can
carry camera-object relation QA without a rendering farm:

  orientation (8 classes)  azimuth from :mod:`vqasynth.orientation`
                           (Orient-Anything), binned into 45-degree sectors;
                           ``azimuth % 360 == 0`` means the object's front
                           faces the camera, and the sectors run
                           counterclockwise (azimuth 90 shows the object's
                           right side, 180 its back, 270 its left)
  viewpoint   (3 classes)  elevation of the object centroid in camera space
                           (VQASynth point clouds are camera-space: camera at
                           the origin, +Y up, +Z into the scene — the same
                           axes :mod:`vqasynth.prompts`' above / front
                           predicates use)
  shot        (3 classes)  camera-object distance over object size — a
                           scale-free framing ratio (a ~50-degree FOV frames
                           an object fully at distance ~= 1.1x its size, so
                           the close-up / medium / long-shot cuts are taken
                           at 1.5x and 4x)

Adaptations vs. the paper (the taxonomy is the ported core; the auxiliaries
are target-native):
  - precise labels come from monocular reconstruction + Orient-Anything
    instead of controlled 3D rendering / diffusion refinement;
  - question paraphrases come from fixed template banks in the
    :mod:`vqasynth.prompt_templates` ``[A]``/``[X]`` convention instead of
    an LLM prompt-generation pass.
Out of scope: the rendering + diffusion image pipeline, the 240K-scale
dataset, and the benchmark / fine-tuning harness.

Dependency model: the binning math and QA formatting are standard-library
only, so they are unit-testable without CUDA, numpy, or open3d. open3d is
imported lazily and only on the ``.pcd``-reading path, like
:func:`vqasynth.scene_fusion.restore_pointclouds`.

References:
  paper : Ultimate3D (https://arxiv.org/abs/2611.24795)
  wires : :meth:`vqasynth.prompts.PromptGenerator.run` (appends these pairs),
          :mod:`vqasynth.orientation` (azimuth source)
"""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# 1) Taxonomy constants
# ---------------------------------------------------------------------------

# Counterclockwise from front-facing-camera; see module docstring.
ORIENTATION_CLASSES = (
    "front",
    "front-right",
    "right",
    "back-right",
    "back",
    "back-left",
    "left",
    "front-left",
)
VIEWPOINT_CLASSES = ("low-angle", "eye-level", "high-angle")
SHOT_CLASSES = ("close-up", "medium shot", "long shot")

# Within +-15 degrees of the object's horizon the camera reads as eye level.
EYE_LEVEL_BAND_DEG = 15.0
# Framing cuts on distance / object-size; see module docstring for the FOV
# reasoning behind 1.5x and 4x.
CLOSE_UP_MAX_RATIO = 1.5
LONG_SHOT_MIN_RATIO = 4.0
# Orient-Anything's in/out-of-distribution head, softmax -> P(in-dist).
# Below this the azimuth on an in-the-wild crop is too unreliable to label.
MIN_ORIENTATION_CONFIDENCE = 0.5


def classify_orientation(azimuth_deg) -> str:
    """Bin an Orient-Anything azimuth (degrees) into the 8 orientation classes.

    Sectors are 45 degrees wide, with ``front`` centered on azimuth 0 (object
    facing the camera) and boundaries at 22.5 + k*45 degrees. Values outside
    0..359 are taken modulo 360, so ``-90`` and ``270`` both map to ``left``.
    """
    sector = int(((float(azimuth_deg) % 360.0) + 22.5) // 45.0) % len(
        ORIENTATION_CLASSES
    )
    return ORIENTATION_CLASSES[sector]


def classify_viewpoint(elevation_deg) -> str:
    """Bin a camera-to-object elevation (degrees) into the 3 viewpoint classes.

    Positive elevation means the object sits above the camera (the camera
    tilts up -> ``low-angle``); negative means the camera looks down
    (``high-angle``).
    """
    elevation_deg = float(elevation_deg)
    if elevation_deg > EYE_LEVEL_BAND_DEG:
        return "low-angle"
    if elevation_deg < -EYE_LEVEL_BAND_DEG:
        return "high-angle"
    return "eye-level"


def classify_shot(distance, object_size) -> str:
    """Bin a camera-object distance over object size into the 3 shot classes.

    Args:
        distance: camera-object distance, same unit as ``object_size``
            (the ratio is scale-free, so meters or scaled units both work).
        object_size: the object's largest axis-aligned extent.

    Raises:
        ValueError: if ``object_size`` is not positive.
    """
    if object_size <= 0:
        raise ValueError(f"object_size must be positive, got {object_size}")
    ratio = float(distance) / float(object_size)
    if ratio >= LONG_SHOT_MIN_RATIO:
        return "long shot"
    if ratio >= CLOSE_UP_MAX_RATIO:
        return "medium shot"
    return "close-up"


# ---------------------------------------------------------------------------
# 2) Camera-space geometry from per-object point clouds (stdlib only)
# ---------------------------------------------------------------------------

def _to_points(item) -> list:
    """Coerce one point-cloud-ish item to a list of (x, y, z) tuples.

    Accepts an open3d ``PointCloud`` (has a ``.points`` attribute), an iterable
    of ``(x, y, z)`` rows, or a ``.pcd`` filepath. open3d is imported lazily
    and only reached for the filepath case (mirroring
    :func:`vqasynth.scene_fusion.restore_pointclouds`).
    """
    points_attr = getattr(item, "points", None)
    if points_attr is not None and not isinstance(item, (str, bytes)):
        return [tuple(p) for p in points_attr]
    if isinstance(item, (str, bytes)):
        import open3d as o3d  # runtime-only dependency, like scene_fusion
        pcd = o3d.io.read_point_cloud(str(item))
        return [tuple(p) for p in pcd.points]
    return [tuple(p) for p in item]


def _is_point_row(el) -> bool:
    """True when ``el`` is one ``(x, y, z)`` row, not a cloud / path / list."""
    if isinstance(el, (str, bytes)) or hasattr(el, "points"):
        return False
    if not isinstance(el, (tuple, list)):
        return False
    return len(el) > 0 and all(isinstance(v, (int, float)) for v in el)


def load_object_points(pointclouds) -> list:
    """Load a list of point-cloud items, unwrapping the single-example wrap.

    Mirrors :func:`vqasynth.detection_3d._load_pointclouds` — the
    single-example ``dataset.map`` path stores the per-object list inside an
    extra list — with one refinement: ``[[point, point, ...]]`` (one object
    given as raw points) is kept as a single object rather than unwrapped
    into per-point pseudo-clouds.
    """
    items = list(pointclouds)
    if (
        len(items) == 1
        and isinstance(items[0], list)
        and not any(_is_point_row(el) for el in items[0])
    ):
        items = items[0]
    return [_to_points(it) for it in items]


def object_centroid(points):
    """Mean position of a ``(x, y, z)`` point list (camera-space)."""
    points = list(points)
    if not points:
        raise ValueError("cannot take the centroid of an empty point cloud")
    n = len(points)
    return tuple(sum(p[i] for p in points) / n for i in range(3))


def camera_elevation(points) -> float:
    """Elevation (degrees) of an object above the camera horizon.

    Positive means the object centroid is above the camera. The camera sits
    at the origin of VQASynth's camera-space clouds (see module docstring).
    """
    x, y, z = object_centroid(points)
    return math.degrees(math.atan2(y, math.hypot(x, z)))


def camera_distance(points) -> float:
    """Camera-object distance: norm of the centroid (camera at the origin)."""
    x, y, z = object_centroid(points)
    return math.sqrt(x * x + y * y + z * z)


def object_size(points) -> float:
    """Largest axis-aligned extent of a point list — the framing diameter."""
    points = list(points)
    if not points:
        raise ValueError("cannot size an empty point cloud")
    return max(
        max(p[i] for p in points) - min(p[i] for p in points) for i in range(3)
    )


# ---------------------------------------------------------------------------
# 3) QA template banks (prompt_templates' [A]/[X] placeholder convention)
# ---------------------------------------------------------------------------

camera_orientation_questions = [
    "Which side of the [A] is facing the camera?",
    "How is the [A] oriented with respect to the camera?",
    "From the camera's viewpoint, what side of the [A] is visible?",
    "Describe the orientation of the [A] relative to the viewer.",
    "What side of the [A] does the camera see?",
]

camera_orientation_answers = [
    "[X].",
    "The [A] is showing its [X] side to the camera.",
    "From the camera's perspective, the [X] of the [A] is visible.",
    "The [A] faces the viewer with its [X].",
]

camera_viewpoint_questions = [
    "From what viewpoint is the [A] photographed?",
    "Is the camera above, below, or at eye level with the [A]?",
    "Describe the camera angle used to capture the [A].",
    "What is the camera's viewpoint relative to the [A]?",
]

camera_viewpoint_answers = [
    "[X].",
    "The photo is taken from a [X] viewpoint.",
    "The camera is positioned [X] relative to the [A].",
    "This is a [X] view of the [A].",
]

camera_shot_questions = [
    "How is the [A] framed in this shot?",
    "Is this a close-up, medium shot, or long shot of the [A]?",
    "What type of camera shot is used for the [A]?",
    "Describe the shot scale of the [A] in the image.",
]

camera_shot_answers = [
    "[X].",
    "The [A] is captured in a [X].",
    "This is a [X] of the [A].",
    "The framing of the [A] is a [X].",
]


# ---------------------------------------------------------------------------
# 4) QA-pair synthesis
# ---------------------------------------------------------------------------

def _fill(question_template, answer_template, label, value):
    """Substitute [A]/[X] and join into the repo's ``question Answer: answer``."""
    question = question_template.replace("[A]", label)
    answer = answer_template.replace("[A]", label).replace("[X]", value)
    return f"{question} Answer: {answer}"


def make_camera_relation_qa(
    label: str,
    points: Optional[Sequence] = None,
    orientation: Optional[dict] = None,
    rng: Optional[random.Random] = None,
) -> list:
    """Turn one object's camera-relation signals into training QA pairs.

    Args:
        label: the object's caption (lowercased here, as
            :class:`vqasynth.prompts.PromptGenerator` does for its pairs).
        points: per-object ``(x, y, z)`` points (or open3d cloud / ``.pcd``
            path). Supplies the viewpoint + shot pairs; skipped when ``None``.
        orientation: an Orient-Anything dict from :mod:`vqasynth.orientation`
            (``azimuth`` in degrees, ``confidence`` in 0..1). Supplies the
            orientation pair; skipped when ``None`` or below
            ``MIN_ORIENTATION_CONFIDENCE``.
        rng: optional ``random.Random`` for deterministic template choice.

    Returns:
        A list of ``"question Answer: answer"`` strings — the same shape
        :meth:`vqasynth.prompts.PromptGenerator.run` emits downstream.
    """
    rng = rng if rng is not None else random
    label = (label or "object").lower()
    pairs = []

    if points is not None:
        points = _to_points(points)
        elevation = camera_elevation(points)
        if math.isfinite(elevation):
            pairs.append(
                _fill(
                    rng.choice(camera_viewpoint_questions),
                    rng.choice(camera_viewpoint_answers),
                    label,
                    classify_viewpoint(elevation),
                )
            )
        size = object_size(points)
        distance = camera_distance(points)
        if math.isfinite(size) and math.isfinite(distance) and size > 0:
            pairs.append(
                _fill(
                    rng.choice(camera_shot_questions),
                    rng.choice(camera_shot_answers),
                    label,
                    classify_shot(distance, size),
                )
            )

    if orientation is not None:
        azimuth = orientation.get("azimuth")
        confidence = orientation.get("confidence", 1.0)
        if (
            azimuth is not None
            and math.isfinite(float(azimuth))
            and float(confidence) >= MIN_ORIENTATION_CONFIDENCE
        ):
            pairs.append(
                _fill(
                    rng.choice(camera_orientation_questions),
                    rng.choice(camera_orientation_answers),
                    label,
                    classify_orientation(float(azimuth)),
                )
            )

    return pairs


def camera_relation_prompts(captions, pointclouds, orientations=None) -> list:
    """Camera-object relation QA for every object in one fused scene.

    This is the hook :meth:`vqasynth.prompts.PromptGenerator.run` appends to
    the spatial-predicate output. One bad object never sinks the row — errors
    are printed and that object's pairs are skipped, matching the defensive
    stance of :class:`vqasynth.detection_3d.Detection3DGenerator`.

    Args:
        captions: per-object captions, aligned with ``pointclouds``.
        pointclouds: per-object point clouds (open3d clouds, ``(x, y, z)``
            lists, or ``.pcd`` paths; the single-example list wrap is
            unwrapped like :func:`vqasynth.detection_3d._load_pointclouds`).
        orientations: optional per-object Orient-Anything dicts — the
            dataset's ``orientation`` column when the
            :mod:`vqasynth.orientation` stage ran, ``None`` otherwise (the
            viewpoint / shot pairs need only the point clouds).

    Returns:
        A list of ``"question Answer: answer"`` strings; empty when nothing
        could be derived.
    """
    clouds = load_object_points(pointclouds)
    orientations = list(orientations) if orientations is not None else []
    captions = list(captions)
    # Align defensively the same way upstream does — truncate to the shorter.
    n = min(len(clouds), len(captions))
    pairs = []
    for i in range(n):
        orientation = orientations[i] if i < len(orientations) else None
        try:
            pairs.extend(
                make_camera_relation_qa(captions[i], clouds[i], orientation)
            )
        except Exception as exc:  # one bad object shouldn't sink the row
            print(
                f"Error generating camera-relation QA for an object, "
                f"skipping: {exc}"
            )
    return pairs


# ---------------------------------------------------------------------------
# 5) CameraRelationGenerator — the standalone stage, mirroring
#    PromptGenerator's surface (as Detection3DGenerator does)
# ---------------------------------------------------------------------------

class CameraRelationGenerator:
    """Camera-object relation QA-pair synthesis stage.

    Consumes the per-object point clouds + captions produced upstream by
    :mod:`vqasynth.localize` + :mod:`vqasynth.scene_fusion` (and, when
    available, the ``orientation`` column from :mod:`vqasynth.orientation`)
    and emits Ultimate3D-taxonomy training pairs. ``run`` mirrors
    :meth:`vqasynth.prompts.PromptGenerator.run` and ``apply_transform``
    mirrors :meth:`vqasynth.prompts.PromptGenerator.apply_transform`, so
    these pairs land in the same dataset schema as the spatial pairs —
    use this stage directly when only camera-relation columns are wanted.
    """

    def run(self, captions, pointclouds, orientations=None) -> list:
        """Emit QA pairs for one scene (empty on error, like PromptGenerator)."""
        try:
            return camera_relation_prompts(captions, pointclouds, orientations)
        except Exception as e:  # noqa: BLE001 - match upstream's defensive stance
            print(f"[CameraRelationGenerator] skipping sample: {e}")
            return []

    def create_messages_from_prompts(self, prompts) -> list:
        """Same message schema ``PromptGenerator.create_messages_from_prompts`` emits.

        The first user message carries the image (``index: 0``); every prompt
        is split on ``"Answer: "`` into a user question + assistant answer.
        """
        messages = []
        first_prompt = True
        for prompt in prompts:
            if "Answer: " not in prompt:
                continue
            question, answer = prompt.split("Answer: ", 1)
            content = [{"index": None, "text": question.strip(), "type": "text"}]
            if first_prompt:
                content.insert(0, {"index": 0, "text": None, "type": "image"})
            messages.append({"content": content, "role": "user"})
            messages.append(
                {
                    "content": [
                        {"index": None, "text": answer.strip(), "type": "text"}
                    ],
                    "role": "assistant",
                }
            )
            first_prompt = False
        return messages

    def apply_transform(
        self,
        example,
        captions: str = "captions",
        pointclouds: str = "pointclouds",
        orientations: str = "orientation",
    ) -> dict:
        """``dataset.map(...)`` transform mirroring ``PromptGenerator.apply_transform``.

        Adds two columns:
          ``camera_relation_prompts``  -> ``"question Answer: answer"`` strings
          ``camera_relation_messages`` -> ``{role, content}`` message dicts

        Returns the empty-list form (never ``None``) so the column schema
        stays stable; a ``len(messages) > 0`` filter (like prompt_stage's)
        drops the empty rows downstream.
        """
        try:
            prompts = camera_relation_prompts(
                example[captions],
                example[pointclouds],
                example.get(orientations),
            )
            return {
                "camera_relation_prompts": prompts,
                "camera_relation_messages": self.create_messages_from_prompts(
                    prompts
                ),
            }
        except Exception as e:  # noqa: BLE001 - match upstream's defensive stance
            print(f"Error processing sample, skipping: {e}")
            return {
                "camera_relation_prompts": [],
                "camera_relation_messages": [],
            }
