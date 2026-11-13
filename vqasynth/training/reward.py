"""GRPO reward adapter: wrap the SpatialThinker dense reward for the Stage-2 RL loop.

``make_grpo_reward`` returns a reward function with the signature a TRL
``GRPOTrainer`` expects — ``reward_func(completions, **columns) -> list[float]`` —
where the per-example ``grounded_state`` and ``answer`` dataset columns are
forwarded as reward kwargs. The GT scene graph is built per example from
``grounded_state`` via :func:`vqasynth.spatial_reward.grounded_state_to_scene`.
"""
from __future__ import annotations

from vqasynth.spatial_reward import dense_reward, grounded_state_to_scene


def _completion_text(completion):
    """Accept a plain string or a chat-message list (TRL passes either) and return
    the assistant text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)):
        texts = []
        for m in completion:
            if isinstance(m, dict):
                content = m.get("content", m)
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, (list, tuple)):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            texts.append(c.get("text", ""))
                        elif isinstance(c, str):
                            texts.append(c)
            elif isinstance(m, str):
                texts.append(m)
        return "\n".join(t for t in texts if t)
    return ""


def grpo_reward(completions, grounded_state, answer, weights=None, return_breakdown=False, **kwargs):
    """Score a batch of GRPO completions with the dense reward.

    Args:
        completions: list of model outputs (str or chat-message list).
        grounded_state: per-example grounded_state records (dict or JSON str),
            aligned with ``completions``; the GT scene is built from each.
        answer: per-example GT answers, aligned with ``completions``.
        weights: optional dense-reward weight overrides.
        return_breakdown: if True, return per-example component dicts instead of
            the scalar totals.

    Returns:
        list[float] of total rewards (or list[dict] breakdowns).
    """
    out = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        gs = grounded_state[i] if grounded_state is not None and i < len(grounded_state) else None
        ans = answer[i] if answer is not None and i < len(answer) else None
        gt_scene = grounded_state_to_scene(gs)
        r = dense_reward(text, gt_scene, ans, weights=weights)
        out.append(r if return_breakdown else r["total"])
    return out


def make_grpo_reward(weights=None):
    """Return a reward function bound to ``weights`` for a TRL ``GRPOTrainer``.

    The returned callable takes ``(completions, grounded_state, answer, **kwargs)``
    — the trainer forwards the same-named dataset columns as reward kwargs — and
    returns a list of scalar total rewards.
    """

    def reward_func(completions, grounded_state=None, answer=None, **kwargs):
        return grpo_reward(completions, grounded_state, answer, weights=weights)

    reward_func.__name__ = "vqasynth_pp_dense_reward"
    return reward_func
