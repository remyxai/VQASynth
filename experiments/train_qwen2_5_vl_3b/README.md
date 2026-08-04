# SpaceQwen2.5-VL-3B-Instruct — LoRA fine-tuning recipe

**Status:** experimental. A runnable, single-GPU LoRA SFT recipe that
reproduces the SpaceQwen2.5-VL-3B-Instruct training setup from a small,
**single NVIDIA A10 (24 GB)** configuration. It consumes existing Hugging
Face assets — no new dataset generation, no base-model pretraining:

| Asset | Source |
|---|---|
| Base model | [`UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B`](https://huggingface.co/UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B) — a Qwen2.5-VL-3B derivative already tuned for chain-of-thought reasoning, a strong starting point for spatial-VQA tuning |
| Dataset | [`remyxai/SpaceThinker`](https://huggingface.co/datasets/remyxai/SpaceThinker) — the OpenSpace successor with spatial-reasoning `<think>` traces (OpenAI/Qwen chat format) |
| Method | LoRA via [`peft`](https://github.com/huggingface/peft) — rank 128, alpha 256, dropout 0.05, **all linear modules** targeted |
| Trainer | TRL [`SFTTrainer`](https://github.com/huggingface/trl) |

This lands the customer's two asks from
[VQASynth issue #48](https://github.com/remyxai/VQASynth/issues/48): a
**training script** (this directory) and a **resource-requirements**
breakdown (below). SpaceLLaVA is intentionally out of scope — Qwen2.5-VL is
the roadmap direction.

## Resource requirements (single A10, 24 GB)

| Component | Approx. footprint |
|---|---|
| Base model, bf16 | ~6.0 GB (3B params × 2 bytes) |
| LoRA trainable params | ~1.5 % of the base (~40-50 M params at r=128 across all linears) |
| Optimizer state + adapter (AdamW fp32 master + 2 moments) | ~0.6-0.8 GB |
| Activations (bs=1, `max_length=4096`, gradient checkpointing on) | ~6-10 GB |
| **Peak VRAM** | **~14-18 GB → fits comfortably in 24 GB** |

- **Adapter size on disk:** ~100 MB (bf16) / ~180 MB (fp32) — only the
  adapter + processor are saved, never the base weights.
- **Wall-clock:** roughly **6-10 GPU-hours for 2 epochs** over the full
  SpaceThinker `train` split (~11.4k rows) at `per_device_train_batch_size=1`
  × `gradient_accumulation_steps=16` (effective batch 16). A 1k-row
  `max_samples` smoke run finishes in well under an hour.
- **Disk:** ~6 GB to cache the base model, ~12 GB for the cached dataset;
  checkpoints are adapter-only so each is ~0.2 GB.

> Numbers are approximate and were estimated, not measured on a benchmark
> host. Real validation belongs on a GPU machine — the structural test
> (below) does not launch a trainer.

## Install

This recipe needs the VLM training stack on top of the base VQASynth
environment (the base `requirements.txt` is unchanged):

```bash
pip install -e .    # VQASynth itself
pip install "trl>=0.13" "peft>=0.12" "transformers==4.48.0" \
            "qwen-vl-utils[decord]" accelerate
# Optional, faster attention on Ampere+ (A10):
pip install flash-attn --no-build-isolation
```

## Quick start

```bash
python -m experiments.train_qwen2_5_vl_3b.train \
    --config experiments/train_qwen2_5_vl_3b/config.yaml
```

Smoke run (cap the split, skip the full ~8-hour training):

```bash
python -m experiments.train_qwen2_5_vl_3b.train \
    --config experiments/train_qwen2_5_vl_3b/config.yaml \
    --max-samples 64
```

The adapter + processor are written to `output_dir`
(`outputs/spacethinker-qwen2_5_vl_3b_lora/` by default). Set
`hub.push_to_hub: true` + `hub.repo_id` in `config.yaml` to publish.

## What the config controls

`config.yaml` is the single source of truth for everything reviewable:
the base model id, the dataset columns, the LoRA target-module list +
rank/alpha/dropout, and the A10 training hyperparameters. The most
important knob is `lora.target_modules` — it targets **all linear
modules**, not attention only:

- Language decoder self-attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Language decoder SwiGLU MLP: `gate_proj`, `up_proj`, `down_proj`
- Qwen2.5-VL vision tower + multimodal projection: `qkv`, `proj`, `fc1`, `fc2`

Attention-only LoRA underfits VLMs; adapting the visual side is what lets
the spatial-reasoning traces move the needle. `peft` matches these by
module-name suffix, so the same list hits both the language decoder and the
vision tower wherever the names coincide.

## Scaling to VLAA-Thinker-7B

The customer asked about scaling to a 7B model. The recipe is model-size
agnostic — to switch, point `base_model` at a 7B Qwen2.5-VL derivative:

```yaml
base_model: <7b-qwen2_5_vl-id>     # e.g. a VLAA-Thinker-7B release
training:
  per_device_train_batch_size: 1   # keep 1; lower max_length if VRAM is tight
  gradient_accumulation_steps: 16
  max_length: 2048                 # shorter ctx to fit 7B activations
```

A 7B model is ~14 GB of bf16 weights alone, so it **does not fit a single
A10** alongside activations — target an **A100 40/80 GB** (single GPU, same
recipe) or move to multi-GPU FSDP. Multi-GPU / DeepSpeed / FSDP scaffolding
is deliberately out of scope for this PR (single-A10 target); the LoRA config
and collator here carry over unchanged when you add it.

## Data handling

`Qwen2_5_VLMultimodalCollator` (in `train.py`) does the Qwen2.5-VL-specific
plumbing the brief requires:

- applies the processor's chat template to each row's `messages`;
- runs the processor over `(text, images)` to build `pixel_values` +
  `image_grid_thw` + `input_ids` (image placeholders expand to the right
  number of `<|image_pad|>` tokens);
- masks the **prompt tokens** (up to and including the final
  `<|im_start|>assistant` header) and pad / image-pad tokens with `-100`, so
  the loss trains only on the assistant response (the `<think>` trace + answer).

Prompt length is measured from a prompt-only processor pass *with images*, so
the image-token expansion is accounted for when masking.

## Testing

Structural test (no GPU, no trainer instantiation, no heavy deps — parses
`config.yaml` and validates the LoRA target-module list + rank/alpha ratio +
dropout):

```bash
pytest tests/test_train_qwen2_5_vl_3b_config.py
```

## Not in scope for this branch

- SpaceLLaVA training plumbing (older LLaVA stack; Qwen2.5-VL is the roadmap).
- Multi-GPU / DeepSpeed / accelerate multi-node configs (single-A10 target).
- Full-parameter fine-tuning (LoRA only) and the RL/GRPO stage the VLAA
  paper describes (SFT adapter only).
- Real end-to-end training in CI (the test validates wiring, not convergence).
- The Qwen2.5-VL multi-modal merger projector (`visual.merger.mlp.{0,2}`) is
  not caught by suffix-name LoRA targeting (its module names are numeric).
  The vast majority of linear params are covered by the listed targets; for
  full coverage switch `lora.target_modules` to a single regex string.
