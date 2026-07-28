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
    from nooa import Agent  # NOOA is the runtime scaffolding
except ImportError as _e:  # pragma: no cover — CPU dev env may not have Py 3.12
    Agent = None
    _NOOA_IMPORT_ERROR = _e
else:
    _NOOA_IMPORT_ERROR = None

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


def _make_agent_class(tier: Tier, **class_kwargs):
    """Build the NOOA Agent subclass with tier-appropriate tool backends.

    Constructed at call time so NOOA's schema inspection sees the tool methods.
    ``class_kwargs`` (e.g. ``llm=...``) are forwarded to NOOA's class-level
    configuration hook (``class Agent(Base, llm=llm):``), matching the NOOA
    quickstart pattern.
    """
    if Agent is None:
        raise ImportError(
            "nooa is required — install via `pip install "
            "'nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main'` "
            f"(requires Python 3.12+). Original error: {_NOOA_IMPORT_ERROR}"
        )

    class _SpatialAnnotator(Agent, **class_kwargs):
        """You are a spatial annotation agent. Given an image and a question,
        use the available tools to ground your answer in real geometry rather
        than guessing from the image. Prefer metric-3D tools when the question
        involves distances or physical relationships; use 2D tools for
        composition questions.

        Always cite which tool call(s) supported each claim. If a tool returns
        an ambiguous or degenerate result, either try a different phrasing or
        return low confidence — do not fabricate.
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

        async def annotate(self, image: Image.Image, question: str) -> SpatialAnswer:
            """Answer a spatial question about the image using available tools.

            Reason about what tools to call, dispatch them, and synthesize the
            answer. Use metric-3D tools when the question involves physical
            distances or relationships in the world; 2D tools when the
            question is about composition or pixel-space geometry.

            Return a SpatialAnswer with the natural-language answer, a
            self-reported confidence tier, tool-call summaries that support
            each claim, and the total tool-call count.
            """
            ...  # NOOA-driven — model implements this at runtime

    return _SpatialAnnotator


def SpatialAnnotator(tier: Tier | None = None, **class_kwargs):
    """Construct a SpatialAnnotator for the given (or auto-detected) tier.

    Args:
        tier: 'cpu' or 'gpu'. If None, uses :func:`detect_tier` heuristic.
        **class_kwargs: forwarded to NOOA's class-level configuration hook
            (e.g. ``llm=...``, matching NOOA's ``class Agent(Base, llm=llm):``
            quickstart pattern).
    """
    if tier is None:
        tier = detect_tier()

    detector = FlorenceDetector(device="cpu" if tier == "cpu" else "cuda")
    segmenter = FlorenceSegmenter(detector=detector)
    depth = VggtEstimator() if tier == "gpu" else DepthProEstimator(device="cpu")

    AgentCls = _make_agent_class(tier, **class_kwargs)
    return AgentCls(detector=detector, segmenter=segmenter, depth=depth)


__all__ = ["SpatialAnnotator", "SpatialAnswer", "Tier"]
