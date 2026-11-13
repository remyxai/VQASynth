"""VQASynth++ Phase-5 training glue (CPU-testable; notebooks import these).

- data: gated pipeline rows -> chat SFT examples (Stage-1 cold-start).
- reward: GRPO reward adapter wrapping the SpatialThinker dense reward (Stage-2).
"""
from vqasynth.training.data import (
    extract_question,
    format_sft_dataset,
    to_sft_example,
)
from vqasynth.training.reward import grpo_reward, make_grpo_reward

__all__ = [
    "extract_question",
    "to_sft_example",
    "format_sft_dataset",
    "grpo_reward",
    "make_grpo_reward",
]
