"""SpatialAnnotator — a NOOA-based agent for open-ended spatial annotation.

Wraps VQASynth's tool inventory (Florence-2 detection/OCR/captioning + metric
depth via DepthPro or VGGT) as an object-oriented agent. The agent's methods
are the tool surface; the LLM selects and sequences them per-sample to answer
arbitrary spatial questions about an image.

Usage:
    from PIL import Image
    from experiments.nooa_agent.spatial_annotator import SpatialAnnotator

    agent = SpatialAnnotator()  # picks CPU or GPU tier automatically
    result = await agent.annotate(
        Image.open("warehouse.jpg"),
        "How far apart are the two workers in the foreground?"
    )

Design references:
- Design conversation in VQASynth Issue #106
- Notes at ~/ecot_spacethinker_notes.md (2026-07-19 estimate-then-verify probe
  + Qwen3-VL family probe + native tool-call emission probe)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

try:
    from nooa import Agent, strategy  # NOOA is the runtime scaffolding
    from nooa.strategies import CodeActStrategy
    from nooa.config import CodeActConfig
except ImportError as _e:  # pragma: no cover — CPU dev env may not have Py 3.12
    Agent = None
    strategy = None
    CodeActStrategy = None
    CodeActConfig = None
    _NOOA_IMPORT_ERROR = _e
else:
    _NOOA_IMPORT_ERROR = None

# Default iteration budget for the annotate() loop. For offline labeling of a
# single spatial question, the worst-case tool chain is roughly:
#   detect_all_objects → describe_region × N → metric_depth → distance_3d × M
# 20 gives comfortable headroom without letting a runaway loop burn budget.
DEFAULT_MAX_ITERATIONS = 20

from experiments.nooa_agent.tools import detect_tier, Tier
from experiments.nooa_agent.tools.florence import (
    Box,
    FlorenceDetector,
    FlorenceSegmenter,
    relative_position_2d,
)
from experiments.nooa_agent.tools.depth import (
    DepthProEstimator,
    DepthResult,
    VggtEstimator,
    depth_at_point,
    distance_3d_meters,
)


@dataclass
class SpatialAnswer:
    """Structured output returned by ``SpatialAnnotator.annotate``."""
    answer: str
    """Natural-language answer to the input question."""

    confidence: str = "medium"
    """One of 'low' | 'medium' | 'high'. Model self-report."""

    supporting_evidence: list[str] = field(default_factory=list)
    """Tool-call summaries the model relied on."""

    tool_calls_used: int = 0
    """Number of tool invocations across the reasoning trace."""


def _make_agent_class(tier: Tier, max_iterations: int = DEFAULT_MAX_ITERATIONS, **class_kwargs):
    """Build the NOOA Agent subclass with tier-appropriate tool backends.

    Constructed at call time so NOOA's schema inspection sees the tool methods.
    ``class_kwargs`` (e.g. ``llm=...``) are forwarded to NOOA's class-level
    configuration hook (``class Agent(Base, llm=llm):``), matching the NOOA
    quickstart pattern.

    Args:
        tier: 'cpu' or 'gpu'.
        max_iterations: cap on the CodeAct reasoning loop for ``annotate()``.
            See :data:`DEFAULT_MAX_ITERATIONS` for the rationale.
        **class_kwargs: forwarded to ``class Agent(Base, **kw):``.
    """
    if Agent is None:
        raise ImportError(
            "nooa is required — install via `pip install "
            "'nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main'` "
            f"(requires Python 3.12+). Original error: {_NOOA_IMPORT_ERROR}"
        )

    _annotate_strategy = CodeActStrategy(
        config=CodeActConfig(max_iterations=max_iterations)
    )

    class _SpatialAnnotator(Agent, **class_kwargs):
        """You are a spatial annotation agent doing OFFLINE LABELING.

        There is no user to ask clarifying questions. Your job is to produce
        a fully-grounded answer using the tools you have. Iterate:

        - When a tool returns ambiguous results, disambiguate with FOLLOW-UP
          tools (describe_region, dense_region_captions, metric_depth for
          foreground selection) rather than stopping.
        - When one phrasing of detect_objects returns nothing, try a
          rephrasing before falling back to detect_all_objects.
        - For any physical-distance claim, USE metric_depth + distance_3d.
          Never estimate from pixel positions alone.
        - For composition claims, USE pixel_relative_position on detected
          boxes. Do not eyeball the image.

        Cite the tool call(s) that support each factual claim. Ambiguity is
        resolved by picking a canonical interpretation (largest bbox for
        'foreground', leftmost/rightmost for lateral questions, etc.) and
        noting the choice in supporting_evidence — NOT by asking the user.

        Only return low confidence after you have exhausted the tool surface.
        """

        tier_name: str = tier

        def __init__(
            self,
            detector: FlorenceDetector,
            segmenter: FlorenceSegmenter,
            depth: Any,
        ):
            super().__init__()
            self.detector = detector
            self.segmenter = segmenter
            self.depth = depth

        # --- Tool methods (auto-derived by NOOA into an OpenAI-tool-schema
        # list; the model sees signature + docstring + return-type annotation).

        def detect_objects(self, image: Image.Image, phrase: str) -> list[Box]:
            """Localize objects matching a natural-language phrase."""
            return self.detector.detect_objects(image, phrase)

        def detect_all_objects(self, image: Image.Image) -> list[Box]:
            """Return all objects Florence-2 finds — full scene inventory."""
            return self.detector.detect_all_objects(image)

        def describe_region(self, image: Image.Image, box: Box) -> str:
            """Caption a specific rectangular region of the image."""
            return self.detector.describe_region(image, box)

        def caption_scene(self, image: Image.Image, detail: str = "short") -> str:
            """Caption the whole image. `detail`: 'short' | 'detailed'."""
            return self.detector.caption(image, detail=detail)

        def dense_region_captions(self, image: Image.Image) -> list[dict]:
            """Cover the scene with per-region captions + bboxes."""
            return self.detector.dense_region_captions(image)

        def read_text(self, image: Image.Image, with_regions: bool = False) -> dict:
            """OCR the image. Set `with_regions=True` for per-text-region boxes."""
            return self.detector.read_text(image, with_regions=with_regions)

        def segment(self, image: Image.Image, referring_expression: str) -> list[dict]:
            """Return polygon regions for the object matching a phrase."""
            return self.segmenter.segment(image, referring_expression)

        def pixel_relative_position(self, box_a: Box, box_b: Box) -> dict:
            """2D pixel-space direction + distance between two boxes."""
            return relative_position_2d(box_a, box_b)

        # --- metric-3D tools (both tiers expose the same interface)

        def metric_depth(self, image: Image.Image) -> DepthResult:
            """Compute metric depth map + focal length + point cloud for the image."""
            return self.depth.metric_depth(image)

        def depth_at_pixel(self, depth: DepthResult, x: float, y: float) -> float:
            """Sample metric depth (meters) at a specific pixel coordinate."""
            return depth_at_point(depth, x, y)

        def distance_3d(self, depth: DepthResult, box_a: Box, box_b: Box) -> dict:
            """Compute 3D metric distance (meters) between two detected objects."""
            return distance_3d_meters(depth, box_a, box_b)

        # --- LLM-driven entry point

        @strategy(_annotate_strategy)
        async def annotate(self, image: Image.Image, question: str) -> SpatialAnswer:
            """Answer a spatial question about the image using available tools.

            This is offline labeling — there is no human on the other end to
            answer clarifying questions. Your job is to produce the best
            possible grounded answer using the tools available. Iterate.

            Recommended reasoning loop:

            1. Start broad if the target isn't obvious: ``caption_scene`` or
               ``detect_all_objects`` to survey what's in the image.
            2. If the question names specific objects, use ``detect_objects``
               with a natural-language phrase. If the result is empty, retry
               with a rephrased query ('the person walking' vs 'walker').
            3. If a detection returns MORE than the question implies (e.g.,
               'the two workers' → 4 boxes), DO NOT stop and ask for
               clarification. Pick a canonical interpretation:
                 - 'foreground' / 'nearest' → boxes with the largest area, or
                   the closest metric depth
                 - 'left'/'right' → sort by center x-coordinate
                 - 'the two X' with no qualifier → the two most prominent
                   instances (largest bboxes)
               Note the choice in ``supporting_evidence``.
            4. For physical-distance questions, ALWAYS call ``metric_depth``
               and then ``distance_3d``. Do not estimate distances from pixel
               coordinates alone.
            5. For pixel-space / composition questions, use
               ``pixel_relative_position`` on detected boxes. No depth needed.
            6. Verify uncertain detections with ``describe_region`` on the
               specific bbox — cheap sanity check.

            Return a SpatialAnswer with:
              - ``answer``: natural-language answer to the question. Always
                include units for metric answers ('3.2 m', not 'about 3').
              - ``confidence``: 'high' if you grounded every claim in a tool
                result; 'medium' if you made a canonical-interpretation choice
                for ambiguity; 'low' only if no tool chain could resolve the
                question at all (rare — try more tools before returning low).
              - ``supporting_evidence``: one short line per claim, citing
                which tool(s) supported it. Include the disambiguation choice
                if you made one.
              - ``tool_calls_used``: total number of tool invocations.
            """
            ...  # NOOA-driven — model implements this at runtime

    return _SpatialAnnotator


def SpatialAnnotator(
    tier: Tier | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    **class_kwargs,
):
    """Construct a SpatialAnnotator for the given (or auto-detected) tier.

    Args:
        tier: 'cpu' or 'gpu'. If None, uses :func:`detect_tier` heuristic.
        max_iterations: cap on the CodeAct reasoning loop for ``annotate()``.
            Bump if you see the agent hitting the limit on complex scenes.
        **class_kwargs: forwarded to NOOA's class-level configuration hook
            (e.g. ``llm=...``, matching NOOA's ``class Agent(Base, llm=llm):``
            quickstart pattern).
    """
    if tier is None:
        tier = detect_tier()

    detector = FlorenceDetector(device="cpu" if tier == "cpu" else "cuda")
    segmenter = FlorenceSegmenter(detector=detector)
    depth = VggtEstimator() if tier == "gpu" else DepthProEstimator(device="cpu")

    AgentCls = _make_agent_class(tier, max_iterations=max_iterations, **class_kwargs)
    return AgentCls(detector=detector, segmenter=segmenter, depth=depth)


__all__ = ["SpatialAnnotator", "SpatialAnswer", "Tier"]
