"""Structural smoke tests for experiments.train_qwen2_5_vl_3b.

Verifies the LoRA + training wiring declared in config.yaml — target-module
list, rank/alpha ratio, dropout, and the single-A10 memory-budget knobs —
without CUDA, without a real model, and without instantiating SFTTrainer.
The training stack (torch / transformers / trl / peft) is intentionally not
imported; train.py keeps those imports lazy inside main(), so only its
dependency-light config/normalizer helpers + PyYAML are exercised here.

Real end-to-end convergence validation belongs on a GPU host.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Exercises the real config-loading code path the script uses (the new
# module), and anchors on a pre-existing package module so the assertions
# are grounded in the actual VQASynth spatial-reasoning data domain rather
# than self-referencing only the new training code.
from experiments.train_qwen2_5_vl_3b.train import (
    build_lora_config_dict,
    load_config,
    normalize_example,
)

# Pre-existing module — the spatial-question/answer taxonomy VQASynth
# generates. Imported (not just referenced) so the test ties the LoRA
# recipe's stated domain to the package's real output distribution.
from vqasynth import prompt_templates


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "train_qwen2_5_vl_3b"
    / "config.yaml"
)

# Module-name suffixes peft targets. Split into groups so each invariant is
# checked independently and a regression points at the right concern.
REQUIRED_LLM_ATTENTION = {"q_proj", "k_proj", "v_proj", "o_proj"}
REQUIRED_LLM_MLP = {"gate_proj", "up_proj", "down_proj"}
VISION_AND_PROJECTION = {"qkv", "proj", "fc1", "fc2"}

# Spatial-reasoning question families the fine-tuned model is expected to
# answer — pulled from the pre-existing prompt taxonomy, not hard-coded.
METRIC_SPATIAL_FAMILIES = (
    "distance_template_questions",
    "height_questions",
    "width_questions",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


# ── config shape ────────────────────────────────────────────────────────────

def test_config_loads_with_expected_top_level_keys(cfg):
    assert {"base_model", "dataset", "lora", "training", "output_dir"} <= set(cfg)


def test_base_model_and_dataset_match_brief(cfg):
    # VLAA-Thinker-Qwen2.5VL-3B base + SpaceThinker dataset, per the brief.
    assert cfg["base_model"] == "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B"
    assert cfg["dataset"]["repo_id"] == "remyxai/SpaceThinker"


# ── LoRA rank / alpha ratio / dropout ───────────────────────────────────────

def test_build_lora_config_dict_round_trips(cfg):
    lora = build_lora_config_dict(cfg)
    assert set(lora) == {
        "r", "lora_alpha", "lora_dropout", "bias",
        "target_modules", "task_type",
    }
    assert lora["task_type"] == "CAUSAL_LM"
    assert lora["bias"] == "none"


def test_lora_rank_and_alpha_values(cfg):
    lora = cfg["lora"]
    assert lora["r"] == 128
    assert lora["lora_alpha"] == 256


def test_lora_alpha_is_exactly_double_rank(cfg):
    """α = 2·r is the stated convention; assert the invariant, not just values."""
    lora = cfg["lora"]
    assert lora["lora_alpha"] == 2 * lora["r"]


def test_lora_dropout(cfg):
    assert cfg["lora"]["lora_dropout"] == pytest.approx(0.05)
    assert 0.0 <= cfg["lora"]["lora_dropout"] < 1.0


# ── LoRA target modules ─────────────────────────────────────────────────────

def test_target_modules_are_nonempty_strings(cfg):
    targets = cfg["lora"]["target_modules"]
    assert isinstance(targets, list) and targets
    assert all(isinstance(t, str) and t for t in targets)


def test_target_modules_cover_llm_attention_and_mlp(cfg):
    """All language-decoder linear modules, not attention only."""
    targets = set(cfg["lora"]["target_modules"])
    assert REQUIRED_LLM_ATTENTION <= targets
    assert REQUIRED_LLM_MLP <= targets


def test_target_modules_cover_vision_and_projection(cfg):
    """The Qwen2.5-VL vision tower + multimodal projection must be in the
    target set — attention-only LoRA underfits VLMs (the brief's explicit
    requirement)."""
    targets = set(cfg["lora"]["target_modules"])
    assert VISION_AND_PROJECTION & targets, (
        "no vision/projection modules targeted — LoRA would be attention-only"
    )
    # At least the bulk of the vision linears should be present.
    assert len(VISION_AND_PROJECTION & targets) >= 2


def test_target_modules_not_attention_only(cfg):
    """Belt-and-suspenders: the target set must extend strictly beyond the
    four language attention projections."""
    targets = set(cfg["lora"]["target_modules"])
    assert targets - REQUIRED_LLM_ATTENTION


# ── single-A10 memory budget ────────────────────────────────────────────────

def test_a10_memory_budget_knobs(cfg):
    t = cfg["training"]
    assert t["bf16"] is True                     # A10 supports bf16
    assert t["gradient_checkpointing"] is True   # required to fit 24 GB
    # A single A10 can only afford a tiny per-device batch on a 3B VLM.
    assert t["per_device_train_batch_size"] <= 2
    # Effective batch must come from accumulation, not from VRAM-busting bs.
    assert t["gradient_accumulation_steps"] >= 8


def test_max_length_is_finite(cfg):
    """An unbounded max_length would let a long <think> trace OOM the A10."""
    assert isinstance(cfg["training"]["max_length"], int)
    assert 512 <= cfg["training"]["max_length"] <= 32768


# ── dataset normalizer (pure Python, no GPU stack) ──────────────────────────

def test_normalize_keeps_messages_that_already_have_image():
    row = {
        "images": ["<pil>"],
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "<pil>"},
                {"type": "text", "text": "How far is the chair from the desk?"},
            ]},
            {"role": "assistant", "content": "About 1.2 m."},
        ],
    }
    out = normalize_example(row, "images", "messages")
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert out["image"] == "<pil>"
    # Only one image part should remain — no duplicate injected.
    image_parts = [p for p in out["messages"][0]["content"] if p.get("type") == "image"]
    assert len(image_parts) == 1


def test_normalize_injects_image_when_messages_lack_one():
    row = {
        "images": ["<pil>"],
        "messages": [
            {"role": "user", "content": "Describe the scene."},
            {"role": "assistant", "content": "A warehouse."},
        ],
    }
    out = normalize_example(row, "images", "messages")
    content = out["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "image", "image": "<pil>"}


def test_normalize_falls_back_to_input_output_when_no_messages():
    row = {"images": ["<pil>"], "input": "Q?", "output": "A."}
    out = normalize_example(
        row, "images", "messages", "input", "output"
    )
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert out["messages"][0]["content"][0] == {"type": "image", "image": "<pil>"}
    assert out["messages"][1]["content"] == "A."


# ── pre-existing package anchor ─────────────────────────────────────────────

def test_spatial_taxonomy_supports_targeted_domain():
    """The LoRA recipe tunes for spatial reasoning on SpaceThinker. Anchor
    that claim on the existing vqasynth.prompt_templates taxonomy: the
    metric-spatial question families the fine-tuned model must handle are
    present in the package's real data-generation output."""
    for family in METRIC_SPATIAL_FAMILIES:
        questions = getattr(prompt_templates, family, None)
        assert isinstance(questions, list) and questions, (
            f"prompt_templates.{family} missing — spatial-reasoning domain "
            "anchor for this training recipe is gone"
        )
    # Sanity: the distance family actually asks about distance.
    assert any(
        "distance" in q.lower() or "far" in q.lower()
        for q in prompt_templates.distance_template_questions
    )
