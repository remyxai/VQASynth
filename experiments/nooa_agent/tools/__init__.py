"""Resource-tier-aware tool wrappers for the SpatialAnnotator agent.

Two tiers, picked automatically by :func:`detect_tier`:

- ``cpu`` — Florence-2-base + monocular depth (Depth Anything V2 Small).
  Runs anywhere; ~2-3 GB VRAM equivalent; slower.
- ``gpu`` — Molmo captioner + SAM2 masks + VGGT-1B metric depth.
  Requires ≥12 GB VRAM; matches VQASynth's production pipeline.

The tier selection is a heuristic. Override with the ``VQASYNTH_AGENT_TIER``
env var (``cpu`` / ``gpu``) to force a tier during testing.
"""
from __future__ import annotations

import os
from typing import Literal

Tier = Literal["cpu", "gpu"]


def detect_tier() -> Tier:
    """Pick a resource tier based on CUDA availability + VRAM.

    Env override: ``VQASYNTH_AGENT_TIER=cpu|gpu``.
    """
    override = os.environ.get("VQASYNTH_AGENT_TIER")
    if override in ("cpu", "gpu"):
        return override  # type: ignore[return-value]

    try:
        import torch  # local import so the module works without torch installed
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return "gpu" if vram_gb >= 12 else "cpu"


__all__ = ["Tier", "detect_tier"]
