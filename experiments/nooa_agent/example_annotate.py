"""End-to-end example: annotate a spatial question about an image.

Requires Python 3.12+ (NOOA dependency), and one of the LiteLLM-supported
model providers configured via env var.

Usage:
    export OPENAI_API_KEY=...    # or ANTHROPIC_API_KEY, or GOOGLE_API_KEY
    python -m experiments.nooa_agent.example_annotate --image warehouse.jpg \\
        --question "How far apart are the two workers in the foreground?"
"""
from __future__ import annotations

import argparse
import asyncio
import os

from PIL import Image


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument(
        "--question", required=True,
        help="Free-form spatial question about the image",
    )
    parser.add_argument(
        "--tier", choices=["cpu", "gpu"], default=None,
        help="Force resource tier (default: auto-detect via CUDA availability)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-5",
        help="LiteLLM-compatible model id (default: claude-sonnet-4-5)",
    )
    args = parser.parse_args()

    if args.tier:
        os.environ["VQASYNTH_AGENT_TIER"] = args.tier

    # Lazy imports so --help works without nooa installed
    from nooa.unifiedllm.registry import get_llm_client
    from experiments.nooa_agent.spatial_annotator import SpatialAnnotator

    llm = get_llm_client(args.model)
    agent = SpatialAnnotator(llm=llm)

    image = Image.open(args.image).convert("RGB")
    result = await agent.annotate(image, args.question)

    print("── Answer ──")
    print(result.answer)
    print()
    print(f"Confidence: {result.confidence}")
    print(f"Tool calls used: {result.tool_calls_used}")
    if result.supporting_evidence:
        print("Supporting evidence:")
        for e in result.supporting_evidence:
            print(f"  - {e}")


if __name__ == "__main__":
    asyncio.run(main())
