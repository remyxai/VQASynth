import random
import re
import numpy as np
import torch
import spacy
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig
)

from transformers.utils.quantization_config import BitsAndBytesConfig
from sam2.sam2_image_predictor import SAM2ImagePredictor

########################################
# Helper for Molmo: parse text -> points
########################################
def extract_points_and_descriptions(molmo_output, image_w, image_h):
    """
    Extract points and their corresponding descriptions from the Molmo output.
    Convert normalized coordinates (0-100) into pixel coordinates.
    """
    pattern = re.compile(
        r'<point\s+x="\s*([0-9]+(?:\.[0-9]+)?)"\s+y="\s*([0-9]+(?:\.[0-9]+)?)"\s+alt="([^"]+)">'
    )
    results = []
    for match in pattern.finditer(molmo_output):
        try:
            x_norm = float(match.group(1))
            y_norm = float(match.group(2))
            description = match.group(3)
        except ValueError:
            continue
        if max(x_norm, y_norm) > 100:
            continue
        x_pixel = (x_norm / 100.0) * image_w
        y_pixel = (y_norm / 100.0) * image_h
        results.append({"points": [x_pixel, y_pixel], "caption": description})
    return results

def extract_captions(raw_text):
    """
    Extracts a list of captions from Molmo's generated text.
    It expects output with tags: <point ... alt="Caption text">.
    Returns a flat list of caption strings.
    """
    pattern = r'<point\s+[^>]*alt="([^"]+)"'
    captions = re.findall(pattern, raw_text)
    return [cap.strip().lower() for cap in captions]

########################################
# 1) BaseCaptionLocalizer
########################################
class BaseCaptionLocalizer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load("en_core_web_sm")
        self.torch_dtype = torch.float16
        self.model = None
        self.processor = None
        self.post_init()

    def post_init(self):
        pass

    def generate_ids(self, inputs):
        raise NotImplementedError

    def run(self, image):
        raise NotImplementedError

########################################
# 2) FlorenceCaptionLocalizer
########################################
class FlorenceCaptionLocalizer(BaseCaptionLocalizer):
    """
    **Key change**: Return a single dictionary with
       { "points": [...], "caption": [...] }
    so it's consistent with MolmoCaptionLocalizer.
    """
    def __init__(self, model_name="microsoft/Florence-2-base", device=None):
        self.model_name = model_name
        super().__init__(device=device)

    def post_init(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).to(self.device)

    def generate_ids(self, inputs):
        expected_dtype = next(self.model.parameters()).dtype
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(expected_dtype)
        return self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            max_length=1024,
            num_beams=1,
            do_sample=False,
        )

    def run(self, image):
        """
        Return:
          {
            "points": a list of bounding boxes (each is [x1, y1, x2, y2]),
            "caption": a list of strings
          }
        """
        # Step 1: Get a more detailed caption.
        prompt_task = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(
            text=prompt_task, images=image, return_tensors="pt"
        ).to(self.device)
        generated_ids = self.generate_ids(inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt_task, image_size=(image.width, image.height)
        )

        # Step 2: For each chunk of the main caption, ground it to bounding boxes.
        main_captions = [c.strip() for c in parsed_answer.get(prompt_task, "").split(".") if c.strip()]
        bboxes_list = []
        captions_list = []

        for chunk in main_captions:
            grounding_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {chunk}"
            grounding_in = self.processor(
                text=grounding_prompt, images=image, return_tensors="pt"
            ).to(self.device)
            grounding_ids = self.generate_ids(grounding_in)
            grounding_text = self.processor.batch_decode(grounding_ids, skip_special_tokens=False)[0]
            parse2 = self.processor.post_process_generation(
                grounding_text,
                task="<CAPTION_TO_PHRASE_GROUNDING>",
                image_size=(image.width, image.height),
            )
            phrase_data = parse2.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
            if "bboxes" not in phrase_data or "labels" not in phrase_data:
                continue

            # The result can be multiple bounding boxes or one. We flatten them.
            # Usually bboxes is a list of floats [x1,y1,x2,y2], or a list of lists.
            bbs = phrase_data["bboxes"]
            if isinstance(bbs, list) and len(bbs) == 4 and isinstance(bbs[0], (int, float)):
                # Single bounding box
                bboxes_list.append([float(x) for x in bbs])
                captions_list.append(chunk)
            elif isinstance(bbs, list) and len(bbs) > 0:
                # Possibly multiple bounding boxes
                for box in bbs:
                    if isinstance(box, list) and len(box) == 4:
                        bboxes_list.append([float(x) for x in box])
                        captions_list.append(chunk)

        return {
            "points": bboxes_list,  # unify name with "points" for consistency
            "caption": captions_list
        }

########################################
# 3) MolmoCaptionLocalizer
########################################
class MolmoCaptionLocalizer(BaseCaptionLocalizer):
    def __init__(self, model_name="cyan2k/molmo-7B-O-bnb-4bit", device=None):
        self.model_name = model_name
        super().__init__(device=device)

    def post_init(self):
        if BitsAndBytesConfig is None:
            raise ImportError("BitsAndBytesConfig not found for 4-bit usage.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        for module in self.model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.float()

    def generate_ids(self, inputs):
        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=300, stop_strings=["<|endoftext|>"]),
                tokenizer=self.processor.tokenizer
            )
            generated_tokens = output[0, inputs["input_ids"].size(1):]
            return generated_tokens.unsqueeze(0)

    def run(self, image):
        """
        Return:
          {
            "points": [ [x1, y1], [x2, y2], ... ],
            "caption": [caption1, caption2, ...]
          }
        """
        prompt = (
            "You are an AI assistant that localizes objects in an image. "
            "Identify each distinct object and output a list of <point> elements in this format: "
            '<point x="X" y="Y" alt="Object description"/>. '
            "Use normalized coordinates from 0 to 100. "
            "Only provide valid points in the specified format."
        )
        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        generated_ids = self.generate_ids(inputs)
        generated_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        image_w, image_h = image.size

        extracted_data = extract_points_and_descriptions(generated_text, image_w, image_h)
        cleaned_captions = extract_captions(generated_text)
        if extracted_data:
            return {
                "points": [entry["points"] for entry in extracted_data],
                "caption": cleaned_captions
            }
        else:
            return {
                "points": [],
                "caption": cleaned_captions
            }

########################################
# 4) LocationRefiner (SAM2)
########################################
class LocationRefiner:
    """
    SAM2-based segmenter that can handle bounding boxes or points.
    """
    def __init__(self, model_name="facebook/sam2-hiera-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_model = SAM2ImagePredictor.from_pretrained(
            model_name, trust_remote_code=True, device=self.device
        )

    def run(self, image, prompts, use_points=False):
        """
        If use_points=False, prompts is a list of bounding boxes [x1, y1, x2, y2].
        If use_points=True, prompts is a list of 2D points [x, y].
        Returns:
          masks: shape (N, H, W)
          scores: shape (N,)
        """
        self.sam2_model.set_image(image)
        if use_points:
            mask_list = []
            score_list = []
            for point in prompts:
                pts = np.array(point, dtype=float).reshape(1, 2)
                labels = np.ones(1, dtype=int)
                masks, scores, _ = self.sam2_model.predict(
                    point_coords=pts,
                    point_labels=labels,
                    box=None,
                    multimask_output=False
                )
                mask_list.append(masks[0])
                score_list.append(scores[0])
            return np.array(mask_list).astype(bool), np.array(score_list)
        else:
            if len(prompts) == 0:
                return np.zeros((0, 1, 1), dtype=bool), np.array([])
            bboxes = np.array(prompts, dtype=float).reshape(-1, 4)
            masks, scores, _ = self.sam2_model.predict(
                point_coords=None,
                box=bboxes,
                multimask_output=False
            )
            return masks.astype(bool), scores

########################################
# 5) Localizer: unify the pipeline
########################################
class Localizer:
    """
    If captioner_type="florence", produce bounding boxes => use_points=False.
    If captioner_type="molmo", produce points => use_points=True.

    Then refine with SAM2 => one mask per bounding box or point.

    The final output from run(image) is always:
      (mask_uint8_list, prompts, captions)

    Where:
      mask_uint8_list: list of shape N, each item is a 2D uint8 mask (0 or 255).
      prompts: list of bounding boxes or points
      captions: list of strings
    """
    def __init__(self, captioner_type="florence", segmenter_model="facebook/sam2-hiera-small", device=None):
        if captioner_type == "florence":
            self.caption_localizer = FlorenceCaptionLocalizer(device=device)
            self.use_points = False
        elif captioner_type == "molmo":
            self.caption_localizer = MolmoCaptionLocalizer(device=device)
            self.use_points = True
        else:
            raise ValueError(f"Unknown captioner_type={captioner_type}")

        self.location_refiner = LocationRefiner(model_name=segmenter_model, device=device)

    def run(self, image):
        """
        1) Use the chosen caption localizer to get:
             { "points": [...], "caption": [...] }

           If Florence => "points" are bounding boxes.
           If Molmo => "points" are coordinate pairs.

        2) Pass those "points" to SAM2 to produce one mask per item.

        3) Convert each mask from bool to uint8 [0..255].

        4) Return a triple of (masks, prompts, captions).
        """
        preds = self.caption_localizer.run(image)
        prompts = preds.get("points", [])
        captions = preds.get("caption", [])

        masks, scores = self.location_refiner.run(image, prompts, use_points=self.use_points)
        if masks is not None and len(masks) > 0:
            mask_uint8_list = [m.astype(np.uint8) * 255 for m in masks]
        else:
            mask_uint8_list = []

        if not (len(mask_uint8_list) == len(prompts) == len(captions)):
            raise ValueError(
                f"Mismatch in lengths. Got {len(mask_uint8_list)} masks, "
                f"{len(prompts)} prompts, {len(captions)} captions."
            )

        return mask_uint8_list, prompts, captions

    def apply_transform(self, example, images):
        """
        For each example in the dataset, produce:
          example["masks"] => a list of mask arrays
          example["bboxes_or_points"] => a list of bounding boxes or points
          example["captions"] => a list of strings
        """
        is_batched = isinstance(example[images], list) and isinstance(example[images][0], (list, Image.Image))

        if is_batched:
            all_masks_list, all_prompts_list, all_captions_list = [], [], []
            for img_list in example[images]:
                # If img_list is [ PIL.Image, ... ], pick the first
                image = img_list[0] if isinstance(img_list, list) else img_list
                if not isinstance(image, Image.Image):
                    raise ValueError("Expected a PIL Image.")
                if image.mode != "RGB":
                    image = image.convert("RGB")

                masks, prompts, captions = self.run(image)
                all_masks_list.append(masks)
                all_prompts_list.append(prompts)
                all_captions_list.append(captions)

            example["masks"] = all_masks_list
            example["bboxes_or_points"] = all_prompts_list
            example["captions"] = all_captions_list

        else:
            # Single example
            image = example[images][0] if isinstance(example[images], list) else example[images]
            if not isinstance(image, Image.Image):
                raise ValueError("Expected a PIL Image.")
            if image.mode != "RGB":
                image = image.convert("RGB")

            masks, prompts, captions = self.run(image)
            example["masks"] = masks
            example["bboxes_or_points"] = prompts
            example["captions"] = captions

        return example

