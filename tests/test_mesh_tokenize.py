"""Smoke tests for vqasynth.mesh_tokenize.

Verifies the LLaMA-Mesh OBJ -> text tokenization pipeline against tiny synthetic
meshes written to tmp files — no Objaverse download, no network. Also exercises
the integration with the existing pipeline by routing mesh records through
``vqasynth.utils.filter_null`` (the same null filter image-derived rows use).
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from vqasynth.mesh_tokenize import (
    filter_faces,
    load_obj,
    mesh_to_text,
    process_directory,
    process_mesh_file,
    quantize_vertices,
    random_rotate,
    simplify_mesh,
    sort_vertices_faces,
)
# Integration gate: exercise a pre-existing module in the package, not only the
# new one. Mesh records must survive the same null filter the rest of the
# pipeline applies to its rows.
from vqasynth.utils import filter_null


TETRA_OBJ = """\
v 0 0 0
v 2 0 1
v 0 2 2
v 1 1 3
f 1 2 3
f 1 3 4
f 1 2 4
f 2 3 4
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_load_obj_parses_vertices_and_zero_based_faces(tmp_path):
    path = _write(tmp_path, "tri.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    vertices, faces = load_obj(path)
    assert vertices.shape == (3, 3)
    # Face indices are converted to 0-based internally (module docstring).
    assert faces.tolist() == [[0, 1, 2]]


def test_load_obj_strips_uv_and_normal_components(tmp_path):
    path = _write(tmp_path, "uv.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1 2/2 3/3\n")
    _, faces = load_obj(path)
    assert faces.tolist() == [[0, 1, 2]]


def test_filter_faces_rejects_over_budget():
    vertices = np.zeros((2, 3))
    faces = np.zeros((501, 3), dtype=int)
    with pytest.raises(ValueError, match="501"):
        filter_faces(vertices, faces, max_faces=500)


def test_filter_faces_passes_under_budget():
    vertices = np.zeros((2, 3))
    faces = np.zeros((10, 3), dtype=int)
    out_v, out_f = filter_faces(vertices, faces, max_faces=500)
    assert len(out_f) == 10


def test_quantize_vertices_in_range_and_at_extents():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    q = quantize_vertices(vertices, bins=64)
    assert q[0].tolist() == [0, 0, 0]
    assert q[1].tolist() == [63, 63, 63]
    assert q.min() >= 0 and q.max() <= 63


def test_quantize_vertices_handles_degenerate_axis():
    # Flat mesh: z is constant -> must not divide by zero.
    vertices = np.array([[0.0, 0.0, 5.0], [1.0, 2.0, 5.0]])
    q = quantize_vertices(vertices, bins=64)
    assert np.isfinite(q).all()
    assert q[:, 2].tolist() == [0, 0]


def test_random_rotate_only_uses_four_z_axis_rotations():
    vertices = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 9.0]])
    identity = np.eye(3)
    r90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    r270 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    expected = {
        tuple((vertices @ r.T).ravel().astype(int).tolist())
        for r in (identity, r90, r180, r270)
    }

    rng = random.Random(0)
    seen = set()
    for _ in range(60):
        out = random_rotate(vertices, rng=rng)
        seen.add(tuple(out.ravel().astype(int).tolist()))
        # z-axis rotation preserves the z coordinate.
        assert np.array_equal(out[:, 2], vertices[:, 2])

    assert seen, "rng never produced a rotation"
    assert seen <= expected  # every observed rotation is one of the four


def test_sort_vertices_faces_remaps_face_indices():
    """Regression for the notebook bug: sorting vertices by z must carry face
    indices along, so each face still references its original geometry."""
    # Distinct z values (5, 1, 9, 3) force a non-trivial sort permutation.
    vertices = np.array(
        [[0.0, 0.0, 5.0], [0.0, 0.0, 1.0], [0.0, 0.0, 9.0], [0.0, 0.0, 3.0]]
    )
    # Face references original vertices 0, 1, 2 -> z = {5, 1, 9}.
    faces = np.array([[0, 1, 2]])
    sorted_v, sorted_f = sort_vertices_faces(vertices, faces)

    # Vertices are now z-ascending.
    assert sorted_v[:, 2].tolist() == [1, 3, 5, 9]
    # And the face still points at the same three vertices (z = {5, 1, 9}).
    referenced = sorted_v[np.asarray(sorted_f[0])]
    assert sorted(referenced[:, 2].tolist()) == [1, 5, 9]


def test_mesh_to_text_emits_obj_tokens_with_one_based_faces():
    vertices = np.array([[0, 0, 0], [63, 63, 63]])
    faces = np.array([[0, 1, 0]])
    text = mesh_to_text(vertices, faces)
    lines = text.strip().split("\n")
    assert lines[0] == "v 0 0 0"
    assert lines[1] == "v 63 63 63"
    # Internal 0-based indices emitted as 1-based OBJ tokens.
    assert lines[2] == "f 1 2 1"


def test_process_mesh_file_round_trip(tmp_path):
    path = _write(tmp_path, "tet.obj", TETRA_OBJ)
    vertices, faces = process_mesh_file(path, bins=64, rng=random.Random(0))
    text = mesh_to_text(vertices, faces)

    n_v = len(vertices)
    bound = 64 - 1
    for line in text.strip().split("\n"):
        parts = line.split()
        assert parts[0] in ("v", "f")
        if parts[0] == "v":
            x, y, z = (int(v) for v in parts[1:4])
            # Rotate-then-quantize: every emitted axis lands in the
            # LLaMA-Mesh token vocabulary ``[0, bins-1]`` regardless of
            # which 90 deg rotation was drawn (min-max quantization
            # re-centers the rotated bounding box).
            assert 0 <= x <= bound
            assert 0 <= y <= bound
            assert 0 <= z <= bound
        else:
            idx = [int(x) for x in parts[1:]]
            assert all(1 <= i <= n_v for i in idx)  # 1-based, in range

    # Vertices are z-ascending after the pipeline.
    assert np.all(np.diff(vertices[:, 2]) >= 0)


def test_process_mesh_file_all_rotations_stay_in_range(tmp_path):
    """The LLaMA-Mesh token vocabulary is ``[0, bins-1]`` on every axis.
    Rotate-before-quantize must keep every emitted vertex inside that
    range regardless of which of the four 90 deg rotations was drawn —
    an earlier revision quantized first and then rotated, which
    produced negative x/y tokens outside the vocabulary."""
    path = _write(tmp_path, "tet.obj", TETRA_OBJ)
    bound = 64 - 1
    # Force each rotation index by seeding the module-level ``random``
    # so ``random.choice(range(4))`` picks each in turn.
    for rot_index in range(4):
        rng = random.Random()
        rng.seed(0)
        # Pre-consume RNG state until choice(range(4)) yields rot_index.
        # Simpler: rebind random_rotate to test each rotation explicitly.
        import vqasynth.mesh_tokenize as mt
        original = mt._Z_AXIS_ROTATIONS
        try:
            mt._Z_AXIS_ROTATIONS = [original[rot_index]]
            vertices, _ = process_mesh_file(
                path, bins=64, rng=random.Random(0),
            )
            for v in vertices:
                for axis_val in v:
                    assert 0 <= int(axis_val) <= bound, (
                        f"rotation {rot_index}: axis value {int(axis_val)} "
                        f"outside [0, {bound}]"
                    )
        finally:
            mt._Z_AXIS_ROTATIONS = original


def test_process_mesh_file_over_budget_is_rejected(tmp_path):
    over = "v 0 0 0\nv 1 0 0\nv 0 1 0\n" + ("f 1 2 3\n" * 600)
    path = _write(tmp_path, "big.obj", over)
    with pytest.raises(ValueError, match="600"):
        process_mesh_file(path, max_faces=500)


def test_process_directory_drops_failed_meshes_via_filter_null(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "good.obj").write_text(TETRA_OBJ)
    # A second mesh over the 500-face budget -> should be skipped, not abort.
    (in_dir / "toobig.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n" + ("f 1 2 3\n" * 600)
    )

    records = process_directory(str(in_dir), str(out_dir), rng=random.Random(0))

    ids = {r["id"] for r in records}
    assert ids == {"good"}  # the over-budget mesh was dropped

    good = next(r for r in records if r["id"] == "good")
    assert good["n_faces"] == 4
    assert good["text"].startswith("v ")
    assert os.path.exists(os.path.join(str(out_dir), "good.txt"))

    # Integration: every returned record passes the pipeline's null filter, and
    # a deliberately None-poisoned record does not.
    for record in records:
        assert filter_null(record)
    assert not filter_null({"id": "x", "text": None})


def test_simplify_mesh_noop_when_already_small():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    out_v, out_f = simplify_mesh(vertices, faces, target_vertices_count=10)
    assert np.array_equal(out_v, vertices)
    assert out_f is faces


def test_simplify_mesh_reduces_vertex_count():
    rng = np.random.default_rng(0)
    # 8 points in general position -> 3D Delaunay is well defined.
    points = rng.random((8, 3))
    out_v, out_f = simplify_mesh(points, np.array([]), target_vertices_count=5)
    assert len(out_v) == 5
    assert out_f is not None
