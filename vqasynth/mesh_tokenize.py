"""LLaMA-Mesh style 3D mesh tokenization for VQASynth (text-to-3D).

Ports the mesh-tokenization pipeline from the maintainer's LLaMA-Mesh Colab into
the vqasynth package, so Objaverse OBJ meshes can be structured into the text
``v x y z`` / ``f a b c`` token format used to fine-tune text-to-3D VLMs.

Pipeline (mirrors the notebook): load OBJ -> filter to <= ``max_faces`` faces ->
quantize vertices into ``bins`` per axis -> random 90 deg z-axis rotation (data
augmentation) -> (optional) simplify to a target vertex count -> sort vertices
by z (faces canonically) -> emit as OBJ-style text tokens.

Two deviations from the reference notebook, both pure correctness hardening
(neither changes the emitted token format on the Objaverse path the brief
targets):

1. ``load_obj`` parses face indices to 0-based internally and ``mesh_to_text``
   emits them back as 1-based OBJ tokens. The notebook carried 1-based indices
   through NumPy array math, which makes the sort step silently point faces at
   the wrong vertices. Internal 0-based / external 1-based is the standard OBJ
   convention and keeps the sort remap correct.
2. ``sort_vertices_faces`` remaps face indices through the sort permutation.
   The notebook reordered vertices by z but left face indices untouched, so
   every emitted face referenced the wrong vertex. The canonical face ordering
   (``sorted(faces, key=sorted)``) is preserved.

The optional ``simplify_mesh`` step is ported verbatim (scipy Delaunay). It is
off the Objaverse path (cell 5 of the notebook omits it) and carries the same
caveat as the original: 3D Delaunay ``.simplices`` are tetrahedra, not triangle
faces, so it is only appropriate for the coarse cow-style demo. It is kept for
faithfulness and left disabled by default.

Objaverse XL download is supported via ``download_and_process_objaverse``, which
lazily imports the optional ``objaverse`` package so importing this module never
requires it.

Reference: LLaMA-Mesh (arXiv:2411.09595) · issue
https://github.com/remyxai/VQASynth/issues/30
"""
from __future__ import annotations

import os
import random
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import Delaunay

__all__ = [
    "load_obj",
    "filter_faces",
    "quantize_vertices",
    "random_rotate",
    "simplify_mesh",
    "sort_vertices_faces",
    "mesh_to_text",
    "write_mesh_text",
    "process_mesh_file",
    "process_directory",
    "download_and_process_objaverse",
]


def load_obj(filename):
    """Parse a Wavefront OBJ file into ``(vertices, faces)`` NumPy arrays.

    Only ``v`` (vertex) and ``f`` (face) lines are consumed; normals, texture
    coords, and objects/groups are ignored. Face vertex specs of the form
    ``v``, ``v/vt``, ``v//vn`` or ``v/vt/vn`` are all accepted by taking the
    first component. Indices are returned 0-based (see module docstring).

    Args:
        filename: path to a ``.obj`` file.

    Returns:
        ``(vertices, faces)`` where ``vertices`` is a ``(V, 3)`` float array and
        ``faces`` is an ``(F, k)`` int array (``k`` is the face arity, typically
        3 for triangle meshes).
    """
    vertices = []
    faces = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif tag == "f":
                # Take the first '/'-separated component of each vertex spec,
                # converted from 1-based OBJ to 0-based internal indexing.
                face = [int(spec.split("/")[0]) - 1 for spec in parts[1:]]
                faces.append(face)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def filter_faces(vertices, faces, max_faces=500):
    """Pass through meshes with at most ``max_faces`` faces.

    Raises:
        ValueError: if the mesh exceeds the face budget. Matches the notebook's
            gate so oversized meshes are skipped during bulk processing.
    """
    if len(faces) > max_faces:
        raise ValueError(
            f"Mesh has {len(faces)} faces, more than the {max_faces} limit."
        )
    return vertices, faces


def quantize_vertices(vertices, bins=64):
    """Min-max normalize each axis to ``[0, 1]`` then floor into ``bins`` levels.

    Degenerate axes (zero extent) are left at 0 to avoid division by zero — the
    same guard the rest of the pipeline uses (see ``vqasynth.utils.colorize``).

    Returns:
        ``(V, 3)`` int array with values in ``[0, bins - 1]``.
    """
    vertices = np.asarray(vertices, dtype=float)
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    extent = max_coords - min_coords
    # Avoid 0-division on flat (degenerate) axes.
    safe = np.where(extent != 0, extent, 1.0)
    normalized = (vertices - min_coords) / safe
    quantized = np.floor(normalized * (bins - 1)).astype(int)
    return quantized


# 90-degree rotations about the z-axis: identity, 90, 180, 270.
_Z_AXIS_ROTATIONS = [
    np.eye(3),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
]


def random_rotate(vertices, rng: Optional[random.Random] = None):
    """Apply a random 0/90/180/270-degree rotation about the z-axis.

    Args:
        vertices: ``(V, 3)`` array.
        rng: optional ``random.Random`` for deterministic augmentation. If
            ``None``, the module-level ``random`` is used (notebook behavior).
    """
    choice = (rng or random).choice(range(len(_Z_AXIS_ROTATIONS)))
    rotation = _Z_AXIS_ROTATIONS[choice]
    return vertices @ rotation.T


def simplify_mesh(vertices, faces, target_vertices_count, rng=None):
    """Coarsen a mesh down to ``target_vertices_count`` vertices.

    Faithful port of the notebook: sub-sample ``target_vertices_count`` vertices
    at random and re-triangulate via a scipy 3D ``Delaunay`` triangulation. The
    input ``faces`` are discarded and replaced by the triangulation's simplices.

    Note:
        3D Delaunay ``.simplices`` are tetrahedra (4 vertices each), not triangle
        faces, and the call requires ``>= 4`` non-coplanar points. This step is
        therefore only appropriate for the coarse cow-style demo (notebook cell
        2); the Objaverse path (cell 5) does not use it. Kept verbatim for
        faithfulness and disabled by default in ``process_mesh_file``.

    Args:
        rng: optional ``numpy.random.Generator`` for deterministic sampling. If
            ``None``, the global ``numpy.random`` state is used (notebook
            behavior).
    """
    vertices = np.asarray(vertices)
    num_vertices = len(vertices)
    if num_vertices <= target_vertices_count:
        return vertices, faces

    if rng is None:
        selected_indices = np.random.choice(num_vertices, target_vertices_count, replace=False)
    else:
        selected_indices = rng.choice(num_vertices, target_vertices_count, replace=False)
    new_vertices = vertices[selected_indices]
    new_faces = Delaunay(new_vertices).simplices
    return new_vertices, new_faces


def sort_vertices_faces(vertices, faces):
    """Sort vertices by z (ascending) and reorder faces to match.

    Vertices are reordered by a stable argsort on their z coordinate, and face
    indices are remapped through the inverse permutation so each face still
    references the same (now relocated) vertex. Faces are then sorted into a
    canonical order. (The reference notebook reordered vertices but left face
    indices stale; see module docstring.)
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if len(vertices) == 0:
        return vertices, faces

    order = np.argsort(vertices[:, 2], kind="stable")  # old indices, z-sorted
    inv = np.empty(order.shape, dtype=np.int64)        # old index -> new position
    inv[order] = np.arange(order.shape[0])

    vertices = vertices[order]
    if faces.size:
        faces = inv[faces]
    # Canonical face ordering (matches the notebook's tie-break).
    faces = sorted(faces.tolist(), key=lambda f: tuple(sorted(f)))
    return vertices, faces


def mesh_to_text(vertices, faces) -> str:
    """Serialize a mesh to ``v x y z`` / ``f a b c`` text tokens.

    Vertex coordinates are emitted as integers (they are whole numbers once
    quantized). Face indices are converted from internal 0-based back to the
    1-based OBJ convention that the VLM token format expects.
    """
    lines = []
    for vertex in np.asarray(vertices):
        lines.append("v " + " ".join(str(int(c)) for c in vertex))
    for face in np.asarray(faces):
        lines.append("f " + " ".join(str(int(i) + 1) for i in face))
    return "\n".join(lines) + ("\n" if lines else "")


def write_mesh_text(vertices, faces, output_path) -> str:
    """Write ``mesh_to_text`` output to ``output_path`` and return the path."""
    with open(output_path, "w") as f:
        f.write(mesh_to_text(vertices, faces))
    return output_path


def process_mesh_file(
    filename,
    max_faces: int = 500,
    bins: int = 64,
    simplify: bool = False,
    target_vertices_count: Optional[int] = None,
    rng: Optional[random.Random] = None,
):
    """Run the full tokenization pipeline on a single OBJ file.

    Mirrors the notebook's Objaverse path (``max_faces=500`` default, no
    simplification). Pass ``simplify=True`` with a ``target_vertices_count`` to
    additionally coarsen the mesh (notebook cell 2 / cow path).

    Returns:
        ``(vertices, faces)`` NumPy arrays ready for ``mesh_to_text`` /
        ``write_mesh_text``.
    """
    vertices, faces = load_obj(filename)

    vertices, faces = filter_faces(vertices, faces, max_faces)
    vertices = quantize_vertices(vertices, bins)
    # NOTE: follows the notebook's quantize-then-rotate order. A 90 deg z-axis
    # rotation permutes and may negate the (already quantized) x/y coordinates,
    # so emitted vertex tokens can be negative. z is invariant.
    vertices = random_rotate(vertices, rng=rng)

    if simplify:
        if target_vertices_count is None:
            raise ValueError("target_vertices_count must be set when simplify=True")
        vertices, faces = simplify_mesh(vertices, faces, target_vertices_count)

    vertices, faces = sort_vertices_faces(vertices, faces)
    return vertices, faces


def process_directory(
    input_dir,
    output_dir,
    max_faces: int = 500,
    bins: int = 64,
    rng: Optional[random.Random] = None,
    skip_errors: bool = True,
):
    """Tokenize every ``.obj`` in ``input_dir`` to a per-mesh ``.txt`` in ``output_dir``.

    Returns a list of per-mesh record dicts shaped for the rest of the VQASynth
    pipeline. Rows that failed to tokenize (raised, or produced no tokens) are
    dropped via ``vqasynth.utils.filter_null`` — the same null filter the
    image-derived rows pass through — so mesh records are drop-in compatible with
    ``vqasynth.datasets``.
    """
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(".obj"):
            continue
        path = os.path.join(input_dir, name)
        mesh_id = os.path.splitext(name)[0]
        out_path = os.path.join(output_dir, f"{mesh_id}.txt")
        # All data keys default to None so failed rows are dropped by filter_null.
        record = {
            "id": mesh_id,
            "source": path,
            "output": None,
            "text": None,
            "n_vertices": None,
            "n_faces": None,
        }
        try:
            vertices, faces = process_mesh_file(
                path, max_faces=max_faces, bins=bins, rng=rng
            )
            text = mesh_to_text(vertices, faces)
            with open(out_path, "w") as f:
                f.write(text)
            record.update(
                output=out_path,
                text=text,
                n_vertices=int(len(vertices)),
                n_faces=int(len(faces)),
            )
        except Exception as exc:
            if not skip_errors:
                raise
            record["error"] = str(exc)
        records.append(record)

    # Integration with the rest of the pipeline: drop incomplete rows. Lazy
    # import keeps the core tokenization path depending only on numpy + scipy.
    from vqasynth.utils import filter_null

    return [record for record in records if filter_null(record)]


def download_and_process_objaverse(
    output_dir,
    objects_to_sample: int = 10,
    max_faces: int = 500,
    bins: int = 64,
    download_dir: str = "~/.objaverse",
    rng: Optional[random.Random] = None,
):
    """Download a sample of Objaverse XL meshes and tokenize each to ``output_dir``.

    Mirrors notebook cell 5. Requires the optional ``objaverse`` package
    (``pip install objaverse``); importing this module does not. Each mesh that
    survives the face-budget filter is written to ``<output_dir>/<sha256>.txt``.
    """
    import objaverse.xl as oxl  # lazy: optional dependency

    os.makedirs(output_dir, exist_ok=True)
    annotations = oxl.get_annotations(download_dir=download_dir)
    sampled_objects = annotations.sample(objects_to_sample)

    def handle_found_object(local_path, file_identifier, sha256, metadata):
        try:
            vertices, faces = process_mesh_file(
                local_path, max_faces=max_faces, bins=bins, rng=rng
            )
            out_path = os.path.join(output_dir, f"{sha256}.txt")
            write_mesh_text(vertices, faces, out_path)
        except Exception as exc:  # mirror the notebook: skip, don't abort the batch
            print(f"Skipped {file_identifier}: {exc}")

    oxl.download_objects(objects=sampled_objects, handle_found_object=handle_found_object)
