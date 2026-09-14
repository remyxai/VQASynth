"""Structural tests for the NOOA 3D scene-graph tool.

Mirrors ``test_boxes3d_tool.py``'s philosophy: no CUDA, no SAM, no open3d —
synthetic box geometry feeds the real tools. The integration anchors are the
pre-existing modules this tool composes: ``experiments.nooa_agent.tools.boxes3d``
(:class:`Box3D` nodes, and the real masks+depth → ``detect_3d_boxes`` chain)
and ``experiments.nooa_agent.spatial_annotator`` (the tool-registry wiring the
``scene_graph`` / ``bev_layout`` methods hang off). If the Box3D contract or
the annotator wiring changes, these tests move with it — this is not a
self-test of the new module alone.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from experiments.nooa_agent.tools.boxes3d import Box3D, detect_3d_boxes
from experiments.nooa_agent.tools.depth import DepthResult
from experiments.nooa_agent.tools.scene_graph import (
    SceneGraphResult,
    bev_layout,
    build_scene_graph,
    describe_scene_graph,
)


def _depth_result(point_cloud_xyz: np.ndarray) -> DepthResult:
    """Build a minimal DepthResult carrying the given (H, W, 3) point cloud.

    Same fixture discipline as ``test_boxes3d_tool.py`` — the point-cloud path
    reads ``point_cloud_xyz`` directly; depth_m / intrinsics only make the
    dataclass shape honest.
    """
    H, W = point_cloud_xyz.shape[:2]
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = 100.0
    return DepthResult(
        depth_m=np.zeros((H, W), dtype=np.float32),
        focal_px=100.0,
        intrinsics_3x3=K,
        point_cloud_xyz=point_cloud_xyz.astype(np.float32),
        backend="test",
    )


def _box(label, center, extent):
    return Box3D(center=center, extent=extent, label=label)


def _relations(graph):
    return {(e.source, e.relation, e.target) for e in graph.edges}


# ---------------------------------------------------------------------------
# Integration: masks + depth -> pre-existing detect_3d_boxes -> scene graph
# ---------------------------------------------------------------------------
def test_scene_graph_built_from_real_boxes3d_output():
    """Full chain: synthetic masks + metric depth through the PRE-EXISTING
    ``detect_3d_boxes`` tool, then ``build_scene_graph`` over its real Box3D
    outputs. Object A ends up left of, above, and nearer than object B."""
    H, W = 4, 4
    pcd = np.zeros((H, W, 3), dtype=np.float32)
    # A: upper-left 2x2 -> axis-aligned 2x2x2 box, center (1, 1, 1).
    pcd[0, 0] = (0, 0, 0)
    pcd[0, 1] = (2, 0, 0)
    pcd[1, 0] = (0, 2, 0)
    pcd[1, 1] = (2, 2, 2)
    # B: lower-right 2x2 -> 1x1x1 box, center (3.5, 3.5, 3.5).
    pcd[2, 2] = (3, 3, 3)
    pcd[2, 3] = (4, 3, 3)
    pcd[3, 2] = (3, 4, 3)
    pcd[3, 3] = (4, 4, 4)

    mask_a = np.zeros((H, W), dtype=bool)
    mask_a[0:2, 0:2] = True
    mask_b = np.zeros((H, W), dtype=bool)
    mask_b[2:4, 2:4] = True

    boxes = detect_3d_boxes(
        Image.new("RGB", (W, H)), [mask_a, mask_b], _depth_result(pcd),
        labels=["crate", "ball"],
    )
    graph = build_scene_graph(boxes)

    assert graph.nodes == boxes  # nodes reference the real tool outputs
    relations = _relations(graph)
    # Camera frame is x→right, y→down, z→away: A(dx=-2.5, dy=-2.5, dz=-2.5)
    # vs the 0.10 m deadband → one canonical edge per axis.
    assert (0, "left_of", 1) in relations
    assert (0, "above", 1) in relations
    assert (0, "in_front_of", 1) in relations
    # Only two objects → the pair is trivially mutual-nearest.
    assert (0, "nearest", 1) in relations
    # Backend carries the boxes' provenance + the symbolic suffix.
    assert graph.backend == "open3d_aabb+symbolic_relations"
    # Every edge carries a deterministic, quotable measurement.
    assert all("m" in e.evidence for e in graph.edges)


def test_axis_deadband_suppresses_marginal_relations():
    # dx = 0.05 m is inside the 0.10 m deadband → no lateral edge, the objects
    # are laterally aligned. Mirrors relative_position_2d's 10-px deadband.
    a = _box("a", (0.0, 0.0, 2.0), (0.3, 0.3, 0.3))
    b = _box("b", (0.05, 0.0, 2.0), (0.3, 0.3, 0.3))
    graph = build_scene_graph([a, b])
    assert not [e for e in graph.edges if e.relation in ("left_of", "right_of")]
    # Proximity is not axis-deadbanded — the pair is still mutual-nearest.
    assert (0, "nearest", 1) in _relations(graph)


# ---------------------------------------------------------------------------
# on_top_of — the support relation distance_3d cannot express
# ---------------------------------------------------------------------------
def _table_and_mug():
    # y is down: table top face at y = 0.95, mug bottom at y = 0.93 → contact.
    table = _box("table", (0.0, 1.0, 2.0), (1.0, 0.1, 1.0))
    mug = _box("mug", (0.0, 0.88, 2.0), (0.1, 0.1, 0.1))
    return table, mug


def test_on_top_of_requires_contact_and_footprint_overlap():
    table, mug = _table_and_mug()
    graph = build_scene_graph([table, mug])
    relations = _relations(graph)
    assert (1, "on_top_of", 0) in relations  # mug rests on the table
    evidence = next(
        e.evidence for e in graph.edges if e.relation == "on_top_of"
    )
    assert "footprint overlap" in evidence

    # Same vertical contact but a disjoint ground footprint (mug floating
    # beside the table) → NOT on top of.
    aside = _box("mug2", (5.0, 0.88, 2.0), (0.1, 0.1, 0.1))
    graph2 = build_scene_graph([table, aside])
    assert not [e for e in graph2.edges if e.relation == "on_top_of"]


def test_on_top_of_rejects_vertical_gap_beyond_tolerance():
    table, _ = _table_and_mug()
    floating = _box("mug", (0.0, 0.5, 2.0), (0.1, 0.1, 0.1))  # 0.4 m above
    graph = build_scene_graph([table, floating])
    assert not [e for e in graph.edges if e.relation == "on_top_of"]


# ---------------------------------------------------------------------------
# nearest — mutual pairs only
# ---------------------------------------------------------------------------
def test_nearest_edge_only_for_mutual_pairs():
    a = _box("a", (0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    b = _box("b", (0.3, 0.0, 0.0), (0.1, 0.1, 0.1))  # a ↔ b mutual nearest
    c = _box("c", (5.0, 0.0, 0.0), (0.1, 0.1, 0.1))  # c's nearest is b, but b's is a
    graph = build_scene_graph([a, b, c])
    nearest = {(e.source, e.target) for e in graph.edges if e.relation == "nearest"}
    assert (0, 1) in nearest
    assert (1, 2) not in nearest
    assert (0, 2) not in nearest


# ---------------------------------------------------------------------------
# bev_layout — the allocentric view
# ---------------------------------------------------------------------------
def test_bev_layout_grid_legend_and_overlap():
    table, mug = _table_and_mug()
    graph = build_scene_graph([table, mug])
    out = bev_layout(graph, grid_size=12)

    assert len(out["grid"]) == 12
    assert all(len(row) == 12 for row in out["grid"])
    # Legend keys nodes to their grid symbol.
    assert out["legend"][0].startswith("a: table")
    assert out["legend"][1].startswith("b: mug")
    # The mug sits on the table → their ground footprints overlap → '*' cells.
    assert any("*" in row for row in out["grid"])
    # Reading guide states the allocentric orientation (row 0 = nearest).
    assert "row 0" in out["view"]
    assert out["cell_size_m"] > 0


def test_bev_layout_row_zero_is_nearest_the_camera():
    # Two non-overlapping objects at different depths: the nearer one (smaller
    # z) must occupy the earlier row — the egocentric→allocentric transform.
    near = _box("near obj", (0.0, 0.0, 1.0), (0.2, 0.2, 0.2))
    far = _box("far obj", (0.0, 0.0, 4.0), (0.2, 0.2, 0.2))
    out = bev_layout(build_scene_graph([near, far]), grid_size=12)
    rows_of = {
        sym: next(r for r, row in enumerate(out["grid"]) if sym in row)
        for sym in ("a", "b")
    }
    assert rows_of["a"] < rows_of["b"]


def test_bev_layout_empty_graph_degrades_gracefully():
    out = bev_layout(build_scene_graph([]))
    assert out["grid"] == []
    assert out["legend"] == []
    assert "no objects" in out["view"]


# ---------------------------------------------------------------------------
# describe_scene_graph — CoT-quotable prose with evidence attached
# ---------------------------------------------------------------------------
def test_describe_scene_graph_quotes_deterministic_evidence():
    table, mug = _table_and_mug()
    lines = describe_scene_graph(build_scene_graph([table, mug]))
    assert lines[0].startswith("scene graph: 2 object(s)")
    on_top = [line for line in lines if "is on top of" in line]
    assert on_top and "footprint overlap" in on_top[0]
    # Labels (not indices) carry into the prose the CoT would quote.
    assert "the mug is on top of the table" in on_top[0]


def test_describe_empty_scene_graph():
    assert describe_scene_graph(build_scene_graph([])) == [
        "empty scene graph — no objects detected"
    ]


# ---------------------------------------------------------------------------
# SceneGraphResult.__repr__ — compact, pinned so NOOA traces don't bloat
# ---------------------------------------------------------------------------
def test_scene_graph_result_repr_is_compact():
    graph = build_scene_graph([_box("x", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    text = repr(graph)
    assert "SceneGraphResult" in text
    assert "objects=1" in text
    assert "\n" not in text
    assert len(text) < 120  # no edge/evidence dump in the summary


# ---------------------------------------------------------------------------
# Tool-registry wiring — spatial_annotator exposes scene_graph / bev_layout
# ---------------------------------------------------------------------------
def _make_bare_annotator():
    """Build the agent class without NOOA or model backends.

    ``_make_agent_class`` only CREATES the class (backends are injected via
    ``__init__``), so with nooa absent we stub the four NOOA symbols it needs
    and instantiate with None backends — the tool methods under test are pure
    delegations that never touch them. On a Py 3.12 host with nooa installed,
    the real class is used as-is.
    """
    import experiments.nooa_agent.spatial_annotator as sa

    saved = (sa.Agent, sa.strategy, sa.CodeActStrategy, sa.CodeActConfig)
    if sa.Agent is None:  # pragma: no cover — CPU dev env has no nooa
        sa.Agent = object
        sa.CodeActStrategy = lambda config: None

        class _StubConfig:  # accepts (and ignores) any kwargs
            def __init__(self, **kwargs):
                pass

        sa.CodeActConfig = _StubConfig

        def _identity_strategy(_s):
            def deco(fn):
                return fn
            return deco

        sa.strategy = _identity_strategy
    try:
        cls = sa._make_agent_class("cpu")
        return cls(detector=None, segmenter=None, depth=None)
    finally:
        sa.Agent, sa.strategy, sa.CodeActStrategy, sa.CodeActConfig = saved


def test_annotator_exposes_scene_graph_and_bev_layout_tools():
    agent = _make_bare_annotator()
    table, mug = _table_and_mug()

    graph = agent.scene_graph([table, mug])  # the wired tool method
    assert isinstance(graph, SceneGraphResult)
    assert (1, "on_top_of", 0) in _relations(graph)

    bev = agent.bev_layout(graph)  # the wired tool method
    assert bev == bev_layout(graph, grid_size=16)  # faithful delegation
    assert len(bev["grid"]) == 16

    # NOOA derives the tool schema from signature + docstring — both must
    # survive on the wired methods, like the other tool methods.
    assert agent.scene_graph.__doc__ and "scene graph" in agent.scene_graph.__doc__
    assert agent.bev_layout.__doc__ and "bird" in agent.bev_layout.__doc__.lower()
