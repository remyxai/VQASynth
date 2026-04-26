"""
Lightweight VLM inference wrapper for the evaluation stage.

Loads a HuggingFace vision-language model and runs inference on
(image, question) pairs to produce predictions for scoring.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from vqasynth.utils import pick_dtype


class VLMInference:
    """
    Run inference with a HuggingFace vision-language model.

    Supports models that follow the transformers AutoModelForVision2Seq
    or chat-template-based VLM pattern (Qwen2-VL, LLaVA-Next, InternVL, etc.).
    """

    def __init__(self, model_name, device=None, max_new_tokens=256):
        """
        Args:
            model_name: HuggingFace model slug (e.g., "Qwen/Qwen2.5-VL-7B-Instruct").
            device: torch device. Auto-detected if None.
            max_new_tokens: Max tokens to generate per response.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.dtype = pick_dtype()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image, question):
        """
        Run inference on a single (image, question) pair.

        Args:
            image: PIL Image or file path.
            question: Question string.

        Returns prediction string.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Build chat messages in the standard VLM format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Apply chat template if available
        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=prompt, images=[image], return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=question, images=image, return_tensors="pt"
            ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only the generated tokens (skip the prompt)
        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        generated_ids = output_ids[0][input_len:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def predict_batch(self, items):
        """
        Run inference on a list of (image, question) dicts.

        Args:
            items: List of dicts with "image" and "question" keys.

        Returns list of prediction strings.
        """
        predictions = []
        for item in items:
            pred = self.predict(item["image"], item["question"])
            predictions.append(pred)
        return predictions


def run_inference_on_dataset(model_name, dataset, ground_truth_column="messages",
                             image_column="image", prediction_column="predictions",
                             max_new_tokens=256, device=None):
    """
    Run VLM inference on a VQASynth dataset to populate the predictions column.

    Extracts (image, question) pairs from the messages column, runs the model,
    and returns the dataset with a new predictions column.

    Args:
        model_name: HuggingFace model slug.
        dataset: HuggingFace Dataset object.
        ground_truth_column: Column with VQASynth messages.
        image_column: Column with images.
        prediction_column: Column name for predictions output.
        max_new_tokens: Max generation length.
        device: torch device.

    Returns the dataset with the predictions column added.
    """
    vlm = VLMInference(model_name, device=device, max_new_tokens=max_new_tokens)

    def _infer(example):
        messages = example.get(ground_truth_column, [])
        image = example.get(image_column)

        if isinstance(image, list) and len(image) > 0:
            image = image[0]

        predictions = []
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]
            if current_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                question_texts = [
                    c.get("text", "")
                    for c in current_msg.get("content", [])
                    if c.get("type") == "text" and c.get("text") is not None
                ]
                if question_texts and image is not None:
                    pred = vlm.predict(image, question_texts[0])
                    predictions.append(pred)

        example[prediction_column] = predictions
        return example

    return dataset.map(_infer, batched=False)


def run_inference_on_benchmark(model_name, benchmark_items, max_new_tokens=256,
                               device=None):
    """
    Run VLM inference on benchmark items.

    Args:
        model_name: HuggingFace model slug.
        benchmark_items: List of normalized benchmark items with "question",
                        "options", and "images" keys.
        max_new_tokens: Max generation length.
        device: torch device.

    Returns dict mapping item ID -> prediction string.
    """
    vlm = VLMInference(model_name, device=device, max_new_tokens=max_new_tokens)
    predictions = {}

    for item in benchmark_items:
        question = item["question"]
        options = item.get("options", [])
        images = item.get("images", [])

        # Format multi-choice options into the question
        if options:
            option_text = "\n".join(
                f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)
            )
            question = f"{question}\n{option_text}\nAnswer with the letter only."

        image = None
        if images:
            img = images[0]
            if isinstance(img, Image.Image):
                image = img
            elif isinstance(img, str):
                try:
                    image = Image.open(img).convert("RGB")
                except Exception:
                    image = None

        if image is not None:
            pred = vlm.predict(image, question)
        else:
            pred = ""

        predictions[item["id"]] = pred

    return predictions
