"""Smoke tests for vqasynth.vggt_speedups.

Verifies wrapper mechanics against a minimal fake VGGT-shaped module — no CUDA,
no real VGGT install, no download. Real end-to-end perf validation belongs on
a GPU host.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from vqasynth.vggt_speedups import apply_avggt_step1


class _FakeBlock(nn.Module):
    """Mimics a VGGT Block: takes (B, N, C) + optional pos, returns (B, N, C)."""

    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.calls = []  # record shape of every forward for inspection

    def forward(self, x, pos=None):
        self.calls.append(tuple(x.shape))
        return self.linear(x)


class _FakeAggregator(nn.Module):
    """Mimics VGGT's aggregator surface: global_blocks + _process_global_attention."""

    def __init__(self, n_global=12, dim=32):
        super().__init__()
        self.global_blocks = nn.ModuleList([_FakeBlock(dim) for _ in range(n_global)])

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        # Call the block at global_idx with the layout (B, S*P, C)
        return self.global_blocks[global_idx](tokens, pos=pos)


class _FakeVGGT(nn.Module):
    def __init__(self, n_global=12, dim=32):
        super().__init__()
        self.aggregator = _FakeAggregator(n_global=n_global, dim=dim)


def test_apply_avggt_step1_wraps_only_early_blocks():
    model = _FakeVGGT(n_global=12, dim=32)
    apply_avggt_step1(model, early_g2f=9)
    assert model._avggt_step1 == {"early_g2f": 9, "n_global": 12}


def test_wrapped_block_reshapes_to_per_frame():
    """When S>1, wrapped block should see (B*S, P, C), not (B, S*P, C)."""
    B, S, P, C = 2, 4, 5, 32
    model = _FakeVGGT(n_global=12, dim=C)
    apply_avggt_step1(model, early_g2f=9)

    x = torch.randn(B, S * P, C)
    # Simulate the aggregator's dispatch that plumbs S, P onto blocks.
    out = model.aggregator._process_global_attention(x, B, S, P, C, 0)

    # Output shape must round-trip back to (B, S*P, C)
    assert out.shape == (B, S * P, C)
    # And the wrapped block should have seen the per-frame layout
    assert model.aggregator.global_blocks[0].calls[-1] == (B * S, P, C)


def test_unwrapped_block_sees_original_layout():
    """Blocks past early_g2f should be untouched — they see (B, S*P, C)."""
    B, S, P, C = 2, 4, 5, 32
    model = _FakeVGGT(n_global=12, dim=C)
    apply_avggt_step1(model, early_g2f=9)

    x = torch.randn(B, S * P, C)
    out = model.aggregator._process_global_attention(x, B, S, P, C, 10)  # unwrapped block
    assert model.aggregator.global_blocks[10].calls[-1] == (B, S * P, C)
    assert out.shape == (B, S * P, C)


def test_s1_short_circuits_wrapper():
    """When S=1 (VQASynth's current call pattern), wrapper should fall through unchanged."""
    B, S, P, C = 1, 1, 20, 32
    model = _FakeVGGT(n_global=12, dim=C)
    apply_avggt_step1(model, early_g2f=9)

    x = torch.randn(B, S * P, C)
    out = model.aggregator._process_global_attention(x, B, S, P, C, 0)
    # At S=1 the fall-through path preserves the original (B, S*P, C) layout
    assert model.aggregator.global_blocks[0].calls[-1] == (B, S * P, C)
    assert out.shape == (B, S * P, C)


def test_reshape_correctness_via_permutation_invariant():
    """
    The wrapped block treats each frame independently. If we permute the frame
    ordering in the input and re-run, per-frame outputs should permute the
    same way (attention doesn't see across frames in the wrapped layer).
    """
    B, S, P, C = 1, 3, 4, 32
    model = _FakeVGGT(n_global=6, dim=C)
    apply_avggt_step1(model, early_g2f=3)
    torch.manual_seed(0)

    frames = [torch.randn(B, P, C) for _ in range(S)]
    orig_input = torch.cat(frames, dim=1)  # (B, S*P, C)
    perm = [2, 0, 1]
    perm_input = torch.cat([frames[i] for i in perm], dim=1)

    orig_out = model.aggregator._process_global_attention(orig_input, B, S, P, C, 0)
    perm_out = model.aggregator._process_global_attention(perm_input, B, S, P, C, 0)

    # Split outputs into per-frame chunks and check they permute correctly.
    orig_chunks = orig_out.reshape(B, S, P, C)
    perm_chunks = perm_out.reshape(B, S, P, C)
    for new_pos, old_pos in enumerate(perm):
        assert torch.allclose(
            perm_chunks[:, new_pos], orig_chunks[:, old_pos], atol=1e-5
        ), f"Frame {new_pos} should equal original frame {old_pos}"


def test_rejects_non_vggt_model():
    class BadModel(nn.Module):
        pass

    with pytest.raises(ValueError, match="aggregator"):
        apply_avggt_step1(BadModel(), early_g2f=9)


def test_rejects_early_g2f_out_of_range():
    model = _FakeVGGT(n_global=6, dim=32)
    with pytest.raises(ValueError, match="early_g2f=9"):
        apply_avggt_step1(model, early_g2f=9)
