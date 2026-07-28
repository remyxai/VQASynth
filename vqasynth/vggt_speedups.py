"""AVGGT Step 1 (early-global-as-frame) for VGGT-1B — training-free memory + latency win.

Issue #89 evaluated the full AVGGT patch (arXiv 2512.02541) and deferred integration
because the paper's 8× speedup only materializes at S≥30 frames per scene, while
VQASynth's typical scene sizes (3-15 images) put us in the MLP-dominated regime
where the compute win is bounded by ~1.26×.

This module ships only Step 1 — converting early global attention blocks to operate
per-frame — because it delivers two independently useful benefits at ANY scale:

1. Memory: early-layer attention drops from O(S²·P²) to O(S·P²), letting us fit
   larger scenes on the same VRAM. Compounds with scene batching.
2. Modest latency win (~5-10% at typical S), growing meaningfully as scenes grow.
3. Weight-preserving: no retraining, no state_dict changes, no accuracy risk on
   frame-only tokens (the paper's analysis shows early global layers don't form
   meaningful cross-view correspondence in VGGT-1B anyway).

Step 2 (subsampled-K/V) is deliberately NOT included here — its 370-LOC complexity
only pays off when attention dominates MLP (S ≥ 30). Revisit if VQASynth scene
sizes grow, or when a shared VGGT run batches many scenes together.

Reference: arXiv:2512.02541 · issue https://github.com/remyxai/VQASynth/issues/89
"""
from __future__ import annotations

import torch
import torch.nn as nn


def apply_avggt_step1(model: nn.Module, early_g2f: int = 9) -> nn.Module:
    """Convert the first ``early_g2f`` global attention blocks to per-frame attention.

    Args:
        model: a VGGT model (loaded from facebookresearch/vggt).
        early_g2f: number of early global blocks to convert. Paper default: 9
            (out of VGGT-1B's 24 global blocks).

    Returns:
        The same model instance, mutated in place.
    """
    agg = getattr(model, "aggregator", None)
    if agg is None or not hasattr(agg, "global_blocks"):
        raise ValueError("model.aggregator.global_blocks not found — is this a VGGT model?")

    n_global = len(agg.global_blocks)
    if early_g2f > n_global:
        raise ValueError(f"early_g2f={early_g2f} > {n_global} available global blocks")

    for i in range(early_g2f):
        _wrap_global_block_as_frame(agg.global_blocks[i])

    _hook_S_P_on_global_call(agg)
    model._avggt_step1 = {"early_g2f": early_g2f, "n_global": n_global}
    return model


def _wrap_global_block_as_frame(block: nn.Module) -> None:
    """Wrap a VGGT global Block to operate as frame attention.

    Reshapes (B, S*P, C) → (B*S, P, C) before calling the block, then reshapes
    back. The block's weights are reused; attention naturally becomes intra-frame.
    """
    orig_forward = block.forward

    def forward(x, pos=None):
        S = getattr(block, "_avggt_S", None)
        P = getattr(block, "_avggt_P", None)
        if S is None or P is None or S == 1:
            # Fall through if per-frame layout isn't set, or if S=1 (nothing to gain).
            return orig_forward(x, pos=pos)
        B, SP, C = x.shape
        if SP != S * P:
            # Layout mismatch — safest to defer to original.
            return orig_forward(x, pos=pos)
        x_pf = x.reshape(B * S, P, C)
        pos_pf = None if pos is None else pos.reshape(B * S, P, -1)
        out = orig_forward(x_pf, pos=pos_pf)
        return out.reshape(B, S * P, C)

    block.forward = forward


def _hook_S_P_on_global_call(agg: nn.Module) -> None:
    """Pre-hook the aggregator's _process_global_attention to plumb S, P per call."""
    orig = agg._process_global_attention

    def wrapped(tokens, B, S, P, C, global_idx, pos=None):
        for blk in agg.global_blocks:
            blk._avggt_S = S
            blk._avggt_P = P
        return orig(tokens, B, S, P, C, global_idx, pos=pos)

    agg._process_global_attention = wrapped
