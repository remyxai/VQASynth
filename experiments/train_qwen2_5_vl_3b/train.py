"""LoRA SFT entry point for SpaceQwen2.5-VL-3B-Instruct.

Reproduces the SpaceQwen2.5-VL-3B-Instruct training setup from a small,
single-A10 (24 GB) configuration. Consumes existing Hugging Face assets —
no new dataset generation, no base-model pretraining:

  - Base model: ``UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B`` (a Qwen2.5-VL-3B
    derivative already tuned for chain-of-thought reasoning).
  - Dataset:     ``remyxai/SpaceThinker`` (the OpenSpace successor with
    spatial-reasoning traces, OpenAI/Qwen chat format).
  - Method:      LoRA via ``peft`` targeting all linear modules
    (attention + MLP of the language decoder + Qwen2.5-VL vision tower and
    multimodal projection linears), rank 128 / alpha 256 / dropout 0.05.
  - Trainer:     TRL ``SFTTrainer`` with bf16 + gradient checkpointing.

The heavy ML stack (torch / transformers / trl / peft / qwen_vl_utils) is
imported lazily inside ``main()`` so the dependency-light helpers in this
module (``load_config``, ``build_lora_config_dict``, the dataset normalizer
and the multimodal collator) stay importable in CI / test environments that
have neither a GPU nor the training stack installed — the same deferred-
import pattern used in ``experiments.nooa_agent.example_lerobot_aloha_ecot``.

Usage::

    # from repo root, after `pip install -e .` + the training extras below
    python -m experiments.train_qwen2_5_vl_3b.train \\
        --config experiments/train_qwen2_5_vl_3b/config.yaml

Training extras (not part of the base pipeline requirements)::

    pip install "trl>=0.13" "peft>=0.12" "transformers==4.48.0" \\
                "qwen-vl-utils[decord]" "accelerate" "flash-attn" --no-build-isolation

See ``experiments/train_qwen2_5_vl_3b/README.md`` for resource requirements
and how to scale the recipe up to VLAA-Thinker-7B on a larger GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


# ── dependency-light config helpers (importable without the GPU stack) ───────


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a training-config YAML file and return it as a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config file {path!r} did not parse to a mapping")
    return cfg


def build_lora_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return ``peft.LoraConfig`` kwargs derived from a parsed config dict.

    Returned as a plain dict (not a constructed ``LoraConfig``) so it is
    usable in environments where ``peft`` is not installed — e.g. for
    structural config tests. ``main()`` passes the result straight into
    ``peft.LoraConfig(**...)``.
    """
    lora = cfg["lora"]
    return {
        "r": int(lora["r"]),
        "lora_alpha": int(lora["lora_alpha"]),
        "lora_dropout": float(lora["lora_dropout"]),
        "bias": lora.get("bias", "none"),
        "target_modules": list(lora["target_modules"]),
        "task_type": lora.get("task_type", "CAUSAL_LM"),
    }


# ── dataset normalization ────────────────────────────────────────────────────


def _messages_have_image(messages: list[dict]) -> bool:
    """True if any message content already carries a usable image part."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image" and (
                    part.get("image") is not None
                    or part.get("base64") is not None
                ):
                    return True
    return False


def _ensure_image_in_messages(
    messages: list[dict], image: Any
) -> list[dict]:
    """Splice a decoded image into the first user turn in Qwen2.5-VL format.

    The SpaceThinker ``messages`` field normally already carries image content;
    this fills it in for snapshots that store the image only in a side column
    (``images``). Existing image placeholders are replaced in-place (no
    duplicates); a missing image part is prepended to the first user message.
    """
    image_part = {"type": "image", "image": image}
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [image_part, {"type": "text", "text": content}]
        elif isinstance(content, list):
            replaced = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    part.clear()
                    part.update(image_part)
                    replaced = True
                    break
            if not replaced:
                content.insert(0, image_part)
        else:
            msg["content"] = [image_part]
        return messages
    # No user turn at all — prepend one carrying the image.
    messages.insert(0, {"role": "user", "content": [image_part]})
    return messages


def normalize_example(
    example: dict,
    image_column: str,
    messages_column: str,
    fallback_input_column: str | None = None,
    fallback_output_column: str | None = None,
) -> dict:
    """Map a SpaceThinker-style row to ``{messages, image}`` for the collator.

    SpaceThinker ships an OpenAI/Quan ``messages`` field plus a side
    ``images`` column. We keep ``messages`` verbatim when it already carries
    image content; otherwise we splice the decoded image from ``images`` in.
    If a snapshot lacks ``messages`` entirely, we rebuild it from
    ``input``/``output`` + the image.
    """
    image = example.get(image_column)
    if isinstance(image, list):
        image = image[0] if image else None

    messages = example.get(messages_column)
    if messages:
        messages = [dict(m) for m in messages]
        if isinstance(messages, list) and image is not None and not _messages_have_image(messages):
            messages = _ensure_image_in_messages(messages, image)
        return {"messages": messages, "image": image}

    # Fallback: reconstruct messages from input/output + image.
    if fallback_input_column and fallback_output_column:
        user_text = example.get(fallback_input_column, "")
        assistant_text = example.get(fallback_output_column, "")
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": image}] if image is not None else []
                ) + [{"type": "text", "text": str(user_text)}],
            },
            {"role": "assistant", "content": str(assistant_text)},
        ]
        return {"messages": messages, "image": image}

    raise ValueError(
        f"example has no {messages_column!r} and no "
        f"({fallback_input_column!r}, {fallback_output_column!r}) fallback pair"
    )


# ── multimodal data collator ─────────────────────────────────────────────────


class Qwen2_5_VLMultimodalCollator:
    """Build the multi-modal input dict Qwen2.5-VL expects, with -100 masking.

    Per example:
      * applies the processor's chat template to ``messages``;
      * runs the processor over (text, images) to materialize ``pixel_values``,
        ``image_grid_thw``, ``input_ids`` and ``attention_mask`` — the image
        placeholders expand to the correct number of ``<|image_pad|>`` tokens;
      * masks the prompt tokens (everything up to and including the final
        ``<|im_start|>assistant`` header) and pad / image-pad tokens with
        ``-100`` so the loss is computed only on the assistant response.

    Prompt length is measured from a prompt-only processor pass (with images,
    so image-token expansion is accounted for). Assumes the tokenizer pads on
    the right — ``main()`` sets ``processor.tokenizer.padding_side = "right"``
    before constructing the collator.
    """

    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info  # deferred — GPU-stack dep

        messages_batch = [ex["messages"] for ex in examples]
        images_batch = []
        for messages in messages_batch:
            imgs, _ = process_vision_info(messages)
            # SpaceThinker rows carry one image each; text-only rows yield
            # imgs=None, which the processor accepts.
            images_batch.append(imgs)

        full_texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in messages_batch
        ]
        batch = self.processor(
            text=full_texts,
            images=images_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Mask prompt tokens per example. The prompt-only text ends right after
        # the ``<|im_start|>assistant\n`` header, so its token length is the
        # boundary at which the (learnable) assistant content begins.
        for i, (messages, imgs) in enumerate(zip(messages_batch, images_batch)):
            prompt_text = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            prompt = self.processor(
                text=[prompt_text], images=[imgs], return_tensors="pt"
            )
            prompt_len = prompt["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        labels[labels == pad_id] = -100
        labels[labels == image_pad_id] = -100

        batch["labels"] = labels
        return batch


# ── entry point ──────────────────────────────────────────────────────────────


def _select_max_samples(n, cfg_max):
    if n is not None:
        return n
    if cfg_max is not None:
        return cfg_max
    return None


def main() -> None:
    # Heavy ML stack imported here so importing this module's helpers never
    # requires CUDA / GPU libraries.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import SFTConfig, SFTTrainer

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the training split to N rows (smoke runs). Overrides config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg["dataset"]
    tcfg = cfg["training"]

    # ── processor + model ──────────────────────────────────────────────────
    torch_dtype = getattr(torch, str(model_cfg.get("torch_dtype", "bfloat16")))
    processor = AutoProcessor.from_pretrained(
        cfg["base_model"], trust_remote_code=model_cfg.get("trust_remote_code", True)
    )
    # Right padding so per-example prompt/assistant boundaries line up across
    # the padded batch (see Qwen2_5_VLMultimodalCollator).
    processor.tokenizer.padding_side = "right"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch_dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model.config.use_cache = False

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_config = LoraConfig(**build_lora_config_dict(cfg))
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── dataset ────────────────────────────────────────────────────────────
    dataset = load_dataset(
        ds_cfg["repo_id"],
        split=ds_cfg.get("split", "train"),
        trust_remote_code=True,
    )
    cap = _select_max_samples(args.max_samples, ds_cfg.get("max_samples"))
    if cap is not None:
        dataset = dataset.select(range(min(cap, len(dataset))))

    dataset = dataset.map(
        normalize_example,
        fn_kwargs={
            "image_column": ds_cfg.get("image_column", "images"),
            "messages_column": ds_cfg.get("messages_column", "messages"),
            "fallback_input_column": ds_cfg.get("fallback_input_column"),
            "fallback_output_column": ds_cfg.get("fallback_output_column"),
        },
        remove_columns=dataset.column_names,
        desc="Normalizing SpaceThinker rows",
    )

    # ── collator ───────────────────────────────────────────────────────────
    collator = Qwen2_5_VLMultimodalCollator(
        processor, max_length=tcfg.get("max_length", 4096)
    )

    # ── trainer ────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        weight_decay=tcfg.get("weight_decay", 0.0),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tcfg.get("logging_steps", 10),
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_steps=tcfg.get("save_steps", 500),
        save_total_limit=tcfg.get("save_total_limit", 2),
        seed=tcfg.get("seed", 42),
        report_to=tcfg.get("report_to", "none"),
        # We supply our own multimodal collator + pre-normalized rows; keep the
        # image/messages columns and disable TRL's text-field auto-formatting.
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])
    print(f"LoRA adapter + processor saved to {cfg['output_dir']}")

    hub_cfg = cfg.get("hub", {})
    if hub_cfg.get("push_to_hub") and hub_cfg.get("repo_id"):
        model.push_to_hub(hub_cfg["repo_id"])
        processor.push_to_hub(hub_cfg["repo_id"])
        print(f"Pushed LoRA adapter + processor to {hub_cfg['repo_id']}")


if __name__ == "__main__":
    main()
