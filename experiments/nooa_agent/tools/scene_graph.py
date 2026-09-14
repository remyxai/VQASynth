"""3D scene graph with deterministic spatial relations + BEV layout.

GraFT (arXiv:2609.03892) showed a compact, easily maintained 3D scene graph
(3DSG) is enough to hand a *frozen* MLLM the 3D structure it lacks: the graph
supplies (1) deterministic geometry through symbolic tools, (2) allocentric
layout through a bird's-eye-view (BEV) rendering, and (3) visual-attribute
grounding through task-relevant egocentric frames — training-free, no 3D
encoder bolted onto the backbone.

**Adapted port (Mode 2).** The graph itself — nodes for objects, edges for
symbolically-computed relations, a top-down layout for allocentric viewing —
is delivered at full fidelity on the surface the repo already has. The
substituted auxiliaries, explicitly: GraFT's ScanQA / VSI-Bench benchmark
harness → cut (evaluation belongs to the downstream multi-benchmark stage);
GraFT's BEV *image* rendering consumed by the MLLM's vision encoder → a text
occupancy grid with per-cell object tags (deterministic, quotable in
``supporting_evidence`` / CoT, no renderer needed); GraFT's egocentric-frame
selection module → nothing new — the annotator already has ``describe_region``
/ ``dense_region_captions``, and every node keeps its ``mask_id`` so the agent
can pull the matching egocentric caption on demand.

What this adds over the existing tool surface: the annotator could already
answer "how far is A from B" (``distance_3d``) and "which of A/B is left in
pixel space" (``pixel_relative_position``), but had no scene-level structure —
no on-top-of / support relation, no nearest-object edges, and no single
allocentric view of the whole layout. That structure is what GraFT's 3DSG
contributed to a frozen MLLM, and what this tool hands the agent.

Nodes are the :class:`Box3D` outputs of
:func:`experiments.nooa_agent.tools.boxes3d.detect_3d_boxes` (SAM2/Florence
masks + DepthPro/VGGT metric depth). Pure stdlib — importing this module pulls
in neither torch nor numpy, so it stays importable on the NOOA test host
without CUDA or weights (same discipline as the other tool modules).

Frame convention (matches :mod:`experiments.nooa_agent.tools.depth`): the
camera frame is x→right, y→down, z→away-from-camera. "above" therefore means
smaller y; "in front of" means smaller z; ``on_top_of`` means vertical contact
(y) plus ground-footprint overlap (x/z).
"""
from __future__ import annotations

import math
import string
from dataclasses import dataclass, field
from typing import Sequence

from experiments.nooa_agent.tools.boxes3d import Box3D

# Deadband for axis relations: a center offset smaller than this emits no edge
# (the objects are aligned on that axis) — the metric analog of
# ``relative_position_2d``'s 10-px deadband. Thin-object depth noise easily
# exceeds a few cm, so 10 cm keeps relation edges from flipping on noise.
DEFAULT_MARGIN_M = 0.10

# Vertical-contact tolerance for on_top_of: the gap between the upper box's
# bottom face and the lower box's top face must fall within ±this many meters.
DEFAULT_CONTACT_TOL_M = 0.15

# BEV grid symbols: '.' empty, '*' two-or-more overlapping footprints.
_EMPTY_CELL = "."
_OVERLAP_CELL = "*"
_NODE_SYMBOLS = string.ascii_lowercase + string.digits


@dataclass
class SceneEdge:
    """One deterministic spatial relation between two scene-graph nodes.

    ``source`` / ``target`` are indices into ``SceneGraphResult.nodes`` —
    labels can duplicate ("chair" ×2), so the index is the stable identity.

    ``relation`` is one of: left_of, right_of, above, below, in_front_of,
    behind, on_top_of, nearest.

    ``evidence`` is the deterministic measurement that justifies the edge
    (e.g. ``"dx=-0.82 m (margin 0.10 m)"``) — quotable in a NOOA trace or CoT
    so a spatial claim cites real geometry, per GraFT's symbolic-tools
    capability.
    """

    source: int
    target: int
    relation: str
    evidence: str


@dataclass
class SceneGraphResult:
    """Scene-level 3D structure: object nodes + deterministic relation edges.

    Fields:
        nodes: the input :class:`Box3D` list (non-finite boxes dropped) — the
            graph references geometry, it does not copy it; node identity is
            the list index.
        edges: :class:`SceneEdge` list (see there for the vocabulary).
        margin_m: axis deadband the relations were computed with.
        contact_tol_m: vertical-contact tolerance used for on_top_of.
        backend: the nodes' box backend suffixed ``+symbolic_relations`` so
            consumers can tell relation extraction was deterministic, not
            learned (e.g. ``"open3d_aabb+symbolic_relations"``).
    """

    nodes: list[Box3D]
    edges: list[SceneEdge] = field(default_factory=list)
    margin_m: float = DEFAULT_MARGIN_M
    contact_tol_m: float = DEFAULT_CONTACT_TOL_M
    backend: str = "symbolic_relations"

    def __repr__(self) -> str:
        # Compact one-liner — same trace-bloat guard as DepthResult/Box3D (a
        # NOOA trace event fires per tool call). Full edges stay on `.edges`.
        return (
            f"SceneGraphResult(objects={len(self.nodes)}, "
            f"edges={len(self.edges)}, backend={self.backend!r})"
        )


def _interval_overlap(a_center: float, a_extent: float,
                      b_center: float, b_extent: float) -> float:
    """1-D overlap length of the two center±extent/2 intervals (<=0: disjoint)."""
    return (
        min(a_center + a_extent / 2, b_center + b_extent / 2)
        - max(a_center - a_extent / 2, b_center - b_extent / 2)
    )


def _axis_edges(i: int, j: int, a: Box3D, b: Box3D, margin: float) -> list[SceneEdge]:
    """Canonical axis relations for one unordered pair — at most one per axis.

    y is down (camera frame): smaller y = higher, so ``a`` is above ``b`` when
    ``a.center[1] < b.center[1] - margin``. z is away from camera: smaller
    z = nearer, so ``a`` is in front of ``b`` when ``a.center[2]`` is the
    smaller one by more than the margin.
    """
    dx = a.center[0] - b.center[0]
    dy = a.center[1] - b.center[1]
    dz = a.center[2] - b.center[2]
    edges = []
    if dx < -margin:
        edges.append(SceneEdge(i, j, "left_of", f"dx={dx:+.2f} m (margin {margin:.2f} m)"))
    elif dx > margin:
        edges.append(SceneEdge(i, j, "right_of", f"dx={dx:+.2f} m (margin {margin:.2f} m)"))
    if dy < -margin:
        edges.append(SceneEdge(i, j, "above", f"dy={dy:+.2f} m (margin {margin:.2f} m)"))
    elif dy > margin:
        edges.append(SceneEdge(i, j, "below", f"dy={dy:+.2f} m (margin {margin:.2f} m)"))
    if dz < -margin:
        edges.append(SceneEdge(i, j, "in_front_of", f"dz={dz:+.2f} m (margin {margin:.2f} m)"))
    elif dz > margin:
        edges.append(SceneEdge(i, j, "behind", f"dz={dz:+.2f} m (margin {margin:.2f} m)"))
    return edges


def _on_top_of(upper: Box3D, lower: Box3D, contact_tol: float) -> tuple[bool, str]:
    """Support predicate: does box ``upper`` rest on box ``lower``?

    Deterministic geometry, no contact sensor: the upper box's bottom face
    must sit within ±``contact_tol`` of the lower box's top face (y is down),
    AND their ground footprints (x/z extents) must overlap — a mug floating
    beside a table is not on the table.
    """
    gap = (upper.center[1] + upper.extent[1] / 2) - (
        lower.center[1] - lower.extent[1] / 2
    )
    overlap_x = _interval_overlap(
        upper.center[0], upper.extent[0], lower.center[0], lower.extent[0]
    )
    overlap_z = _interval_overlap(
        upper.center[2], upper.extent[2], lower.center[2], lower.extent[2]
    )
    if upper.center[1] < lower.center[1] and abs(gap) <= contact_tol \
            and overlap_x > 0 and overlap_z > 0:
        return True, (
            f"footprint overlap {overlap_x:.2f} m x {overlap_z:.2f} m, "
            f"vertical gap {gap:+.2f} m (tol {contact_tol:.2f} m)"
        )
    return False, ""


def build_scene_graph(
    boxes: Sequence[Box3D],
    *,
    margin_m: float = DEFAULT_MARGIN_M,
    contact_tol_m: float = DEFAULT_CONTACT_TOL_M,
) -> SceneGraphResult:
    """Build a 3D scene graph from 3D bounding boxes (agent tool).

    One node per :class:`Box3D`; edges are computed symbolically from the box
    geometry — GraFT's deterministic-geometry capability over the annotator's
    own depth/mask outputs:

    - left_of / right_of / above / below / in_front_of / behind — pairwise
      axis relations with a deadband: center offsets within ``margin_m`` emit
      no edge (the objects are aligned on that axis).
    - on_top_of — vertical contact + ground-footprint overlap; the support
      relation ``distance_3d`` alone cannot express ("resting on", not just
      "close to").
    - nearest — mutual-nearest pairs by center distance, emitted once per
      pair.

    Boxes with non-finite centers/extents are skipped (dropped-depth
    artifacts), not silently turned into NaN-relation nodes.

    Args:
        boxes: :class:`Box3D` list from the boxes3d tool.
        margin_m: axis deadband in meters (see :data:`DEFAULT_MARGIN_M`).
        contact_tol_m: on_top_of vertical-contact tolerance.

    Returns:
        :class:`SceneGraphResult` — feed to :func:`bev_layout` for the
        allocentric view or :func:`describe_scene_graph` for prose.
    """
    nodes = [
        b for b in boxes
        if all(math.isfinite(v) for v in tuple(b.center) + tuple(b.extent))
    ]
    backend = f"{nodes[0].backend}+symbolic_relations" if nodes else "symbolic_relations"

    edges: list[SceneEdge] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            edges.extend(_axis_edges(i, j, a, b, margin_m))
            # Support relation — check both orders; given distinct centers at
            # most one holds, but a symmetric interpenetration honestly emits
            # both rather than guessing.
            for src, tgt, upper, lower in ((i, j, a, b), (j, i, b, a)):
                ok, evidence = _on_top_of(upper, lower, contact_tol_m)
                if ok:
                    edges.append(SceneEdge(src, tgt, "on_top_of", evidence))

    if len(nodes) >= 2:
        nearest_of = {}
        for i, a in enumerate(nodes):
            best_j, best_d = None, math.inf
            for j, b in enumerate(nodes):
                if i == j:
                    continue
                d = math.dist(tuple(a.center), tuple(b.center))
                if d < best_d:
                    best_j, best_d = j, d
            nearest_of[i] = (best_j, best_d)
        for i, (j, d) in nearest_of.items():
            if i < j and nearest_of[j][0] == i:  # mutual — emit once per pair
                edges.append(
                    SceneEdge(i, j, "nearest", f"mutual nearest, center distance {d:.2f} m")
                )

    return SceneGraphResult(
        nodes=list(nodes),
        edges=edges,
        margin_m=margin_m,
        contact_tol_m=contact_tol_m,
        backend=backend,
    )


def bev_layout(graph: SceneGraphResult, *, grid_size: int = 16) -> dict:
    """Render the scene graph as an allocentric bird's-eye-view layout.

    GraFT's BEV capability, as a text occupancy grid instead of a rendered
    image: rows are depth (row 0 = nearest the camera) and columns are x
    (left → right, matching the camera frame) — the egocentric→allocentric
    transform the pixel-space tools can't provide. Each object's ground
    footprint (the x/z extent of its 3D box) is rasterized into shared square
    cells; cells covered by 2+ footprints show ``*`` so stacking / occlusion
    is visible from above.

    Returns a dict with:
        grid: ``grid_size`` row strings (``.`` empty, per-node letters,
            ``*`` overlap).
        legend: one entry per node — ``"a: table (x=0.00 m, z=2.00 m)"``.
        cell_size_m: meters per grid cell.
        x_range_m / z_range_m: the ground extents covered.
        view: one-line reading guide for the agent.
    """
    if not graph.nodes:
        return {
            "grid": [],
            "legend": [],
            "cell_size_m": 0.0,
            "x_range_m": [0.0, 0.0],
            "z_range_m": [0.0, 0.0],
            "view": "bird's-eye layout: no objects — empty scene graph",
        }

    min_x = min(b.center[0] - b.extent[0] / 2 for b in graph.nodes)
    max_x = max(b.center[0] + b.extent[0] / 2 for b in graph.nodes)
    min_z = min(b.center[2] - b.extent[2] / 2 for b in graph.nodes)
    max_z = max(b.center[2] + b.extent[2] / 2 for b in graph.nodes)

    pad = 1  # one empty ring so footprints don't touch the border
    cell = max(max_x - min_x, max_z - min_z, 1e-6) / max(1, grid_size - 2 * pad)

    def _clamp(v: int) -> int:
        return max(0, min(grid_size - 1, v))

    def _col(x: float) -> int:
        return _clamp(pad + int((x - min_x) / cell))

    def _row(z: float) -> int:
        return _clamp(pad + int((z - min_z) / cell))

    grid = [[_EMPTY_CELL] * grid_size for _ in range(grid_size)]
    legend = []
    for idx, b in enumerate(graph.nodes):
        symbol = _NODE_SYMBOLS[idx % len(_NODE_SYMBOLS)]
        for r in range(_row(b.center[2] - b.extent[2] / 2),
                       _row(b.center[2] + b.extent[2] / 2) + 1):
            for c in range(_col(b.center[0] - b.extent[0] / 2),
                           _col(b.center[0] + b.extent[0] / 2) + 1):
                if grid[r][c] == _EMPTY_CELL:
                    grid[r][c] = symbol
                elif grid[r][c] != symbol:
                    grid[r][c] = _OVERLAP_CELL
        legend.append(f"{symbol}: {b.label} (x={b.center[0]:.2f} m, z={b.center[2]:.2f} m)")

    return {
        "grid": ["".join(row) for row in grid],
        "legend": legend,
        "cell_size_m": round(cell, 3),
        "x_range_m": [round(min_x, 2), round(max_x, 2)],
        "z_range_m": [round(min_z, 2), round(max_z, 2)],
        "view": (
            "bird's-eye (allocentric): rows = depth (row 0 nearest the camera), "
            f"columns = x left→right, '*' = overlapping footprints, "
            f"cell = {cell:.2f} m"
        ),
    }


# Prose phrasing per relation — the quotable form GraFT-style 3DSG
# descriptions feed a frozen MLLM / downstream CoT.
_RELATION_PROSE = {
    "left_of": "is left of",
    "right_of": "is right of",
    "above": "is above",
    "below": "is below",
    "in_front_of": "is in front of",
    "behind": "is behind",
    "on_top_of": "is on top of",
    "nearest": "is nearest to",
}


def describe_scene_graph(graph: SceneGraphResult) -> list[str]:
    """Natural-language scene-graph summary (pure function).

    One line per edge, phrased for prompt / chain-of-thought augmentation —
    the 3DSG-grounded description style GraFT feeds the frozen MLLM, e.g.
    ``"the mug is on top of the table (footprint overlap 0.10 m x 0.10 m,
    vertical gap +0.02 m …)"``. The deterministic geometry stays attached to
    every claim so a downstream CoT can cite it.
    """
    if not graph.nodes:
        return ["empty scene graph — no objects detected"]
    lines = [
        f"scene graph: {len(graph.nodes)} object(s) — "
        + ", ".join(f"{i}:{b.label}" for i, b in enumerate(graph.nodes))
    ]
    for e in graph.edges:
        src = graph.nodes[e.source].label
        tgt = graph.nodes[e.target].label
        lines.append(f"the {src} {_RELATION_PROSE[e.relation]} the {tgt} ({e.evidence})")
    return lines


__all__ = [
    "SceneEdge",
    "SceneGraphResult",
    "build_scene_graph",
    "bev_layout",
    "describe_scene_graph",
    "DEFAULT_MARGIN_M",
    "DEFAULT_CONTACT_TOL_M",
]
