"""3D object-detection QA-pair synthesis stage.

Builds on the per-object point clouds that :mod:`vqasynth.scene_fusion` already
emits (VGGT metric depth -> SAM2 masks -> one ``.pcd`` per detected object) and
on the open3d bounding-box helpers prototyped in
``tests/data_processing/clipseg_data_processing.py`` — specifically
``get_bounding_box_height`` / ``compare_bounding_box_height``, which call
``pcd.get_axis_aligned_bounding_box()``. This stage extends that path: instead
of reading only the box *height*, it emits the full axis-aligned 3D box (center
+ extent, with an optional oriented box for objects where the axis-aligned form
is a poor fit) and formats it as Molmo-style pointing training data.

The output shape is the 3D analog of the ``<point ...>`` tags emitted by
:mod:`vqasynth.localize`: a ``<point3d ...>`` tag carrying the box center,
extent, and the semantic label carried over from the segmentation/captioning
stage. A SpatialRGPT-style bracketed ``[center, extent]`` region descriptor is
also available (SpatialRGPT is the design-space anchor for region-level spatial
descriptors — see the design brief).

Scope discipline (issue #47):
  - This stage does NOT re-run depth estimation or segmentation. It consumes the
    per-object point clouds + captions produced upstream.
  - It does NOT port SpatialRGPT's model; the reference is for the OUTPUT-FORMAT
    design only.

Dependency model: the bounding-box math (axis-aligned + oriented) and the
QA-pair formatting are pure-Python standard library, so they are unit-testable
without CUDA, depth models, SAM, numpy, or open3d installed. open3d (and numpy,
if available) are imported lazily and only on the ``.pcd``-reading path — the
same role PySpark/Polars play for other dataframe-expectation code paths.

References:
  issue  : https://github.com/remyxai/VQASynth/issues/47
  anchor : tests/data_processing/clipseg_data_processing.py (get_axis_aligned_bounding_box)
  format : vqasynth/localize.py (<point> precedent), SpatialRGPT
           (https://www.anjiecheng.me/assets/SpatialRGPT/Spatial_RGPT.pdf)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# 1) Bounding-box data structures (pure data, no third-party deps)
# ---------------------------------------------------------------------------

@dataclass
class OrientedBoundingBox3D:
    """An oriented (rotation-aware) 3D bounding box.

    Attached to a :class:`BoundingBox3D` only when the axis-aligned form wastes
    volume (diagonal / elongated objects). Represented in SpatialRGPT's
    region-descriptor convention: ``center`` and ``extent`` in the box's local
    frame, plus ``rotation`` — the three principal axes (rows) that map the
    local frame onto the scene axes.
    """

    center: tuple
    extent: tuple
    rotation: tuple  # 3 unit principal axes (rows), each a 3-tuple

    @property
    def volume(self) -> float:
        return float(self.extent[0] * self.extent[1] * self.extent[2])


@dataclass
class BoundingBox3D:
    """An axis-aligned 3D bounding box, with an optional oriented refinement.

    Attributes:
        center: (cx, cy, cz) box center in metric (canonicalized) scene units.
        extent: (dx, dy, dz) full side lengths along each axis.
        label: semantic label carried from the segmentation/captioning stage.
        oriented: an :class:`OrientedBoundingBox3D` attached when the
            axis-aligned form is a poor fit; ``None`` otherwise.
    """

    center: tuple
    extent: tuple
    label: str = ""
    oriented: Optional[OrientedBoundingBox3D] = None

    @property
    def volume(self) -> float:
        return float(self.extent[0] * self.extent[1] * self.extent[2])

    @property
    def min_bound(self) -> tuple:
        return tuple(self.center[i] - self.extent[i] / 2.0 for i in range(3))

    @property
    def max_bound(self) -> tuple:
        return tuple(self.center[i] + self.extent[i] / 2.0 for i in range(3))

    def corners(self) -> list:
        """Return the 8 corner points of the axis-aligned box."""
        c = self.center
        half = tuple(self.extent[i] / 2.0 for i in range(3))
        signs = (
            (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
            (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
        )
        return [tuple(c[k] + s[k] * half[k] for k in range(3)) for s in signs]


# ---------------------------------------------------------------------------
# 2) Bounding-box computation (pure Python; numpy-free)
# ---------------------------------------------------------------------------

def compute_aabb(points: Iterable, label: str = "") -> BoundingBox3D:
    """Axis-aligned 3D bounding box from an iterable of (x, y, z) points.

    This is the pure-Python equivalent of open3d's
    ``pcd.get_axis_aligned_bounding_box()`` — the same primitive the clipseg
    prototype's ``get_bounding_box_height`` builds on — computed directly from
    the points so the box math is unit-testable without open3d installed.
    """
    pts = [tuple(p) for p in points]
    if not pts:
        raise ValueError("cannot compute a bounding box from zero points")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    extent = (maxx - minx, maxy - miny, maxz - minz)
    center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0, (minz + maxz) / 2.0)
    return BoundingBox3D(center=center, extent=extent, label=label)


def _jacobi_eigendecomp(matrix):
    """Diagonalize a 3x3 symmetric matrix via cyclic Jacobi rotations.

    Returns ``(eigenvalues, eigenvectors)`` where ``eigenvectors`` is a list of
    three principal axes (each a unit 3-vector in scene coordinates).
    Pure-Python Numerical-Recipes-style implementation so the oriented-box math
    is unit-testable without numpy.
    """
    a = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(50):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
        if off <= 1e-12:
            break
        for p in range(3):
            for q in range(p + 1, 3):
                apq = a[p][q]
                if abs(apq) <= 1e-15:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0)
                )
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                a[p][p] -= t * apq
                a[q][q] += t * apq
                a[p][q] = 0.0
                a[q][p] = 0.0
                for i in range(3):
                    if i == p or i == q:
                        continue
                    aip = a[i][p]
                    aiq = a[i][q]
                    a[i][p] = c * aip - s * aiq
                    a[p][i] = a[i][p]
                    a[i][q] = s * aip + c * aiq
                    a[q][i] = a[i][q]
                for i in range(3):
                    vip = v[i][p]
                    viq = v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq
    eigenvalues = [a[0][0], a[1][1], a[2][2]]
    # Columns of v are the eigenvectors.
    eigenvectors = [
        (v[0][0], v[1][0], v[2][0]),
        (v[0][1], v[1][1], v[2][1]),
        (v[0][2], v[1][2], v[2][2]),
    ]
    return eigenvalues, eigenvectors


def compute_obb(points: Iterable) -> OrientedBoundingBox3D:
    """Oriented bounding box via PCA (pure Python).

    Equivalent to open3d's ``pcd.get_oriented_bounding_box()`` for the principal
    axes, computed without numpy so it is unit-testable. The principal axes come
    from the eigendecomposition of the point-cloud covariance matrix.
    """
    pts = [tuple(p) for p in points]
    n = len(pts)
    if n < 2:
        raise ValueError("oriented box needs at least 2 points")

    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cz = sum(p[2] for p in pts) / n

    cov = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for x, y, z in pts:
        dx, dy, dz = x - cx, y - cy, z - cz
        cov[0][0] += dx * dx
        cov[0][1] += dx * dy
        cov[0][2] += dx * dz
        cov[1][1] += dy * dy
        cov[1][2] += dy * dz
        cov[2][2] += dz * dz
    for i in range(3):
        for j in range(i, 3):
            cov[i][j] /= n
            cov[j][i] = cov[i][j]

    _, axes = _jacobi_eigendecomp(cov)  # 3 unit principal axes (rows)

    # Project the centered points onto the principal axes and take per-axis span.
    proj = [
        (
            (x - cx) * axes[0][0] + (y - cy) * axes[0][1] + (z - cz) * axes[0][2],
            (x - cx) * axes[1][0] + (y - cy) * axes[1][1] + (z - cz) * axes[1][2],
            (x - cx) * axes[2][0] + (y - cy) * axes[2][1] + (z - cz) * axes[2][2],
        )
        for x, y, z in pts
    ]
    lo = (min(p[0] for p in proj), min(p[1] for p in proj), min(p[2] for p in proj))
    hi = (max(p[0] for p in proj), max(p[1] for p in proj), max(p[2] for p in proj))
    extent = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])

    # OBB center = centroid + sum of per-axis midpoints rotated back to scene.
    mid = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0)
    ox = cx + mid[0] * axes[0][0] + mid[1] * axes[1][0] + mid[2] * axes[2][0]
    oy = cy + mid[0] * axes[0][1] + mid[1] * axes[1][1] + mid[2] * axes[2][1]
    oz = cz + mid[0] * axes[0][2] + mid[1] * axes[1][2] + mid[2] * axes[2][2]

    return OrientedBoundingBox3D(
        center=(ox, oy, oz),
        extent=extent,
        rotation=tuple(axes),
    )


def compute_bounding_box(
    points: Iterable, label: str = "", oriented_threshold: Optional[float] = 0.35
) -> BoundingBox3D:
    """Compute the axis-aligned box; attach an oriented box when AABB is wasteful.

    ``oriented_threshold`` is the minimum relative volume the oriented box must
    save (``1 - obb.volume / aabb.volume``) before we bother attaching it. Pass
    ``0`` or ``None`` to disable the oriented refinement entirely.
    """
    box = compute_aabb(points, label=label)
    if oriented_threshold and oriented_threshold > 0 and box.volume > 0:
        try:
            obb = compute_obb(points)
        except ValueError:
            obb = None
        if obb is not None and obb.volume > 0:
            saving = 1.0 - (obb.volume / box.volume)
            if saving >= oriented_threshold:
                box.oriented = obb
    return box


# ---------------------------------------------------------------------------
# 3) Comparison helpers (generalize the clipseg prototype's height helpers)
# ---------------------------------------------------------------------------

def box_height(box: BoundingBox3D) -> float:
    """Vertical (Y) extent of a 3D box.

    Generalizes ``get_bounding_box_height`` in
    ``tests/data_processing/clipseg_data_processing.py`` (which returns
    ``aabb.get_extent()[1]``) to operate on a :class:`BoundingBox3D`.
    """
    return float(box.extent[1])


def compare_box_height(box_i: BoundingBox3D, box_j: BoundingBox3D) -> bool:
    """True if ``box_i`` is taller than ``box_j``.

    The :class:`BoundingBox3D` analog of the clipseg prototype's
    ``compare_bounding_box_height``.
    """
    return box_height(box_i) > box_height(box_j)


def box_volume(box: BoundingBox3D) -> float:
    return float(box.extent[0] * box.extent[1] * box.extent[2])


def compare_box_volume(box_i: BoundingBox3D, box_j: BoundingBox3D) -> bool:
    """True if ``box_i`` encloses more 3D volume than ``box_j``."""
    return box_volume(box_i) > box_volume(box_j)


# ---------------------------------------------------------------------------
# 4) QA-pair formatting (Molmo <point3d> + SpatialRGPT bracketed descriptor)
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    """Format a coordinate to a stable 2-decimal string for parseable answers."""
    return f"{float(value):.2f}"


def format_point3d(box: BoundingBox3D) -> str:
    """Molmo-style 3D pointing tag — the 3D analog of localize.py's ``<point>``.

    ``<point x=".." y=".." z=".." alt=".."/>`` becomes
    ``<point3d x=".." y=".." z=".." extent="dx,dy,dz" alt=".."/>``.
    """
    cx, cy, cz = box.center
    dx, dy, dz = box.extent
    return (
        f'<point3d x="{_fmt(cx)}" y="{_fmt(cy)}" z="{_fmt(cz)}" '
        f'extent="{_fmt(dx)},{_fmt(dy)},{_fmt(dz)}" alt="{box.label}"/>'
    )


def format_box_coordinates(box: BoundingBox3D) -> str:
    """SpatialRGPT-style bracketed region descriptor: ``[center, extent]``."""
    center = "[" + ",".join(_fmt(v) for v in box.center) + "]"
    extent = "[" + ",".join(_fmt(v) for v in box.extent) + "]"
    return f"[{center},{extent}]"


# Uses the same [A] / [X] placeholder convention as vqasynth.prompt_templates so
# 3D-detection pairs slot into the existing answer-string pipeline.
detection_3d_questions = [
    "Where is the [A] located in 3D space?",
    "What is the 3D bounding box of the [A]?",
    "Give the 3D coordinates of the [A].",
    "Locate the [A] in the scene and report its 3D box.",
    "Provide the 3D position and dimensions of the [A].",
    "Identify the 3D bounding box for the [A].",
]

detection_3d_answers = [
    "[X]",
    "The [A] is at [X].",
    "The 3D bounding box of the [A] is [X].",
    "[A]: [X].",
]


def make_qa_pairs(
    box: BoundingBox3D,
    n_questions: Optional[int] = None,
    fmt: str = "point3d",
    rng: Optional[random.Random] = None,
) -> list:
    """Turn one 3D box into training QA pairs.

    Args:
        box: a :class:`BoundingBox3D`.
        n_questions: how many question paraphrases to emit (``None`` => all).
        fmt: ``"point3d"`` (Molmo-style ``<point3d>`` tag, the default — matches
            :mod:`vqasynth.localize`'s ``<point>`` output convention) or
            ``"bracket"`` (SpatialRGPT-style ``[center, extent]`` descriptor).
        rng: optional ``random.Random`` for deterministic answer selection.

    Returns:
        A list of ``"question Answer: answer"`` strings — the same shape
        :meth:`vqasynth.prompts.PromptGenerator.run` /
        ``create_messages_from_prompts`` consume downstream.
    """
    if fmt == "point3d":
        value = format_point3d(box)
    elif fmt == "bracket":
        value = format_box_coordinates(box)
    else:
        raise ValueError(f"unknown fmt {fmt!r}; use 'point3d' or 'bracket'")

    rng = rng if rng is not None else random
    label = box.label or "object"

    questions = list(detection_3d_questions)
    if n_questions is not None:
        questions = questions[: int(n_questions)]

    pairs = []
    for q in questions:
        answer_template = rng.choice(detection_3d_answers)
        question = q.replace("[A]", label)
        answer = answer_template.replace("[A]", label).replace("[X]", value)
        pairs.append(f"{question} Answer: {answer}")
    return pairs


# ---------------------------------------------------------------------------
# 5) Point-cloud loading (open3d/numpy are runtime-only, imported lazily)
# ---------------------------------------------------------------------------

def _to_points(item) -> list:
    """Coerce one point-cloud-ish item to a list of (x, y, z) tuples.

    Accepts an open3d ``PointCloud`` (has a ``.points`` attribute), an iterable
    of ``(x, y, z)`` rows, or a ``.pcd`` filepath. open3d is imported lazily and
    only reached for the filepath case (mirroring
    :func:`vqasynth.scene_fusion.restore_pointclouds`).
    """
    points_attr = getattr(item, "points", None)
    if points_attr is not None and not isinstance(item, (str, bytes)):
        return [tuple(p) for p in points_attr]
    if isinstance(item, (str, bytes)):
        import open3d as o3d  # runtime-only dependency, like scene_fusion.restore_pointclouds
        pcd = o3d.io.read_point_cloud(str(item))
        return [tuple(p) for p in pcd.points]
    return [tuple(p) for p in item]


def _load_pointclouds(pointclouds) -> list:
    """Load a list of point-cloud items, unwrapping the single-example wrap.

    Mirrors :func:`vqasynth.scene_fusion.restore_pointclouds`: the single-example
    ``dataset.map`` path stores the per-object list inside an extra list.
    """
    items = list(pointclouds)
    if len(items) == 1 and isinstance(items[0], list):
        items = items[0]
    return [_to_points(it) for it in items]


def _serialize_box(box: BoundingBox3D) -> dict:
    """JSON-friendly dict form of a box for storing in a HuggingFace dataset row."""
    data = {
        "center": [float(c) for c in box.center],
        "extent": [float(e) for e in box.extent],
        "label": box.label,
        "oriented": None,
    }
    if box.oriented is not None:
        data["oriented"] = {
            "center": [float(c) for c in box.oriented.center],
            "extent": [float(e) for e in box.oriented.extent],
            "rotation": [[float(x) for x in row] for row in box.oriented.rotation],
        }
    return data


# ---------------------------------------------------------------------------
# 6) Detection3DGenerator — the stage, mirroring PromptGenerator's surface
# ---------------------------------------------------------------------------

class Detection3DGenerator:
    """Per-object 3D bounding-box QA-pair synthesis.

    Consumes the per-object point clouds + captions produced upstream by
    :mod:`vqasynth.localize` + :mod:`vqasynth.scene_fusion`, computes a 3D box
    per object, and emits Molmo-style ``<point3d>`` pointing training pairs (the
    3D analog of :mod:`vqasynth.localize`'s ``<point>`` outputs).

    ``run`` mirrors :meth:`vqasynth.prompts.PromptGenerator.run` and
    ``apply_transform`` mirrors :meth:`vqasynth.prompts.PromptGenerator.apply_transform`,
    so 3D-detection pairs land in the same dataset schema as the 2D spatial pairs.
    """

    def __init__(
        self,
        fmt: str = "point3d",
        questions_per_object: int = 3,
        oriented_threshold: Optional[float] = 0.35,
    ):
        self.fmt = fmt
        self.questions_per_object = questions_per_object
        self.oriented_threshold = oriented_threshold

    def compute_boxes(self, captions: Sequence[str], pointclouds) -> list:
        """Compute one :class:`BoundingBox3D` per object (label from captions)."""
        clouds = _load_pointclouds(pointclouds)
        captions = list(captions)
        # Align defensively the same way upstream does — truncate to the shorter.
        n = min(len(clouds), len(captions))
        boxes = []
        for label, pts in zip(captions[:n], clouds[:n]):
            if not pts:
                continue
            boxes.append(
                compute_bounding_box(
                    pts, label=label, oriented_threshold=self.oriented_threshold
                )
            )
        return boxes

    def run(self, captions, pointclouds) -> list:
        """Compute boxes and emit QA pairs.

        Mirrors :meth:`vqasynth.prompts.PromptGenerator.run(captions, pointclouds, ...)`.
        Returns a list of ``"question Answer: answer"`` strings (empty on error).
        """
        try:
            prompts = []
            for box in self.compute_boxes(captions, pointclouds):
                prompts.extend(
                    make_qa_pairs(
                        box,
                        n_questions=self.questions_per_object,
                        fmt=self.fmt,
                    )
                )
            return prompts
        except Exception as e:  # noqa: BLE001 - match upstream's defensive stance
            print(f"[Detection3DGenerator] skipping sample: {e}")
            return []

    def create_messages_from_prompts(self, prompts) -> list:
        """Same message schema ``PromptGenerator.create_messages_from_prompts`` emits.

        The first user message carries the image (``index: 0``); every prompt is
        split on ``"Answer: "`` into a user question + assistant answer.
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
    ) -> dict:
        """``dataset.map(...)`` transform mirroring ``PromptGenerator.apply_transform``.

        Adds three columns:
          ``detection_3d_boxes``    -> serialized per-object boxes (list[dict])
          ``detection_3d_prompts``  -> ``"question Answer: answer"`` strings
          ``detection_3d_messages`` -> ``{role, content}`` message dicts

        Returns the empty-list form (never ``None``) so the column schema stays
        stable; a ``len(messages) > 0`` filter (like prompt_stage's) drops the
        empty rows downstream.
        """
        try:
            boxes = self.compute_boxes(example[captions], example[pointclouds])
            prompts = []
            for box in boxes:
                prompts.extend(
                    make_qa_pairs(
                        box,
                        n_questions=self.questions_per_object,
                        fmt=self.fmt,
                    )
                )
            return {
                "detection_3d_boxes": [_serialize_box(b) for b in boxes],
                "detection_3d_prompts": prompts,
                "detection_3d_messages": self.create_messages_from_prompts(prompts),
            }
        except Exception as e:  # noqa: BLE001 - match upstream's defensive stance
            print(f"Error processing sample, skipping: {e}")
            return {
                "detection_3d_boxes": [],
                "detection_3d_prompts": [],
                "detection_3d_messages": [],
            }
