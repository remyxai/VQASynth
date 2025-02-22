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
    results = []
    pattern = re.compile(
        r'<point\s+x="\s*([0-9]+(?:\.[0-9]+)?)"\s+y="\s*([0-9]+(?:\.[0-9]+)?)"\s+alt="([^"]+)">'
    )

    for match in pattern.finditer(molmo_output):
        try:
            x_norm = float(match.group(1))
            y_norm = float(match.group(2))
            description = match.group(3)
        except ValueError:
            continue

        # Ensure values are within expected range
        if max(x_norm, y_norm) > 100:
            continue

        # Convert from normalized [0,100] to pixel coords
        x_pixel = (x_norm / 100.0) * image_w
        y_pixel = (y_norm / 100.0) * image_h

        results.append({"points": (x_pixel, y_pixel), "caption": description})

    return results

########################################
# 1) BaseCaptionLocalizer
########################################
class BaseCaptionLocalizer:
    """
    Common base class. 
    Subclasses must define:
      - post_init() to load model+processor
      - generate_ids() for custom generation
      - run() to produce location prompts (bboxes or points)
    """
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load("en_core_web_sm")
        self.torch_dtype = torch.float16
        self.model = None
        self.processor = None
        self.post_init()

    def post_init(self):
        """Subclasses override to load self.model, self.processor, etc."""
        pass

    def generate_ids(self, inputs):
        """Subclasses override with model-specific calls (generate or generate_from_batch)."""
        raise NotImplementedError

    def run(self, image):
        """
        Subclasses override. 
        Return something like: [ {"bboxes": [...], "caption": ...} ] 
        or [ {"points": [...], "caption": ...} ] 
        """
        raise NotImplementedError

########################################
# 2) FlorenceCaptionLocalizer
########################################
class FlorenceCaptionLocalizer(BaseCaptionLocalizer):
    """
    Florence has built-in bounding-box extraction with .post_process_generation().
    """
    def __init__(self, model_name="microsoft/Florence-2-base", device=None):
        self.model_name = model_name
        super().__init__(device=device)

    def post_init(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).to(self.device)

    def generate_ids(self, inputs):
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],  # Florence expects pixel_values
            max_length=1024,
            num_beams=1,
            do_sample=False,
        )
        return generated_ids

    def run(self, image):
        """
        1) Generate a “detailed caption” <MORE_DETAILED_CAPTION>
        2) Use post_process_generation(...) => bounding boxes
        3) Possibly do a <CAPTION_TO_PHRASE_GROUNDING> prompt for each chunk
        4) Return: [{"bboxes": [[x1,y1,x2,y2], ...], "caption": text_chunk}, ...]
        """

        # 1) Prepare Florence inputs
        prompt_task = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(
            text=prompt_task,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        # 2) Generate
        generated_ids = self.generate_ids(inputs)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # 3) post_process_generation => parse bounding boxes
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt_task, image_size=(image.width, image.height)
        )
        # The text is split by ".", each chunk might yield bounding boxes
        main_captions = parsed_answer.get(prompt_task, "").split(".")

        results = []
        for chunk in main_captions:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Next prompt for phrase grounding
            grounding_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {chunk}"
            grounding_in = self.processor(
                text=grounding_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            grounding_ids = self.generate_ids(grounding_in)
            grounding_text = self.processor.batch_decode(
                grounding_ids, skip_special_tokens=False
            )[0]
            parse2 = self.processor.post_process_generation(
                grounding_text, 
                task="<CAPTION_TO_PHRASE_GROUNDING>",
                image_size=(image.width, image.height)
            )
            phrase_data = parse2.get("<CAPTION_TO_PHRASE_GROUNDING>", {})

            if "bboxes" not in phrase_data or "labels" not in phrase_data:
                continue

            results.append({
                "bboxes": phrase_data["bboxes"],
                "caption": chunk
            })

        return results

########################################
# 3) MolmoCaptionLocalizer
########################################

class MolmoCaptionLocalizer(BaseCaptionLocalizer):
    """
    Molmo doesn't output bounding boxes directly, so we parse the text for "points".
    The Localizer will then call SAM2 with `use_points=True`.
    """

    def __init__(self, model_name="cyan2k/molmo-7B-O-bnb-4bit", device=None):
        self.model_name = model_name
        super().__init__(device=device)

    def post_init(self):
        """Initializes the model and processor with 4-bit quantization and correct dtype handling."""
        if BitsAndBytesConfig is None:
            raise ImportError("BitsAndBytesConfig not found for 4-bit usage.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16  # Ensure consistent dtype for compute
        )

        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16  # Load processor in FP16
        )

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16  # Load model in FP16
        )

        # Convert all LayerNorm modules to FP32 to avoid type mismatches during layer_norm
        for module in self.model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.float()

    def generate_ids(self, inputs):
        """
        Generates output text tokens using Molmo's `generate_from_batch(...)` method.
        """
        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=300, stop_strings=["<|endoftext|>"]),
                tokenizer=self.processor.tokenizer
            )
            # Extract only the newly generated tokens
            generated_tokens = output[0, inputs["input_ids"].size(1):]
            return generated_tokens.unsqueeze(0)

    def run(self, image):
        """
        Runs inference on the input image, extracting points and object captions.
        Returns a structured output like:
          [{
              "points": [(x1, y1), (x2, y2), ...],
              "captions": ["Object description 1", "Object description 2", ...]
          }]
        """
        prompt = (
            "You are an AI assistant that localizes objects in an image. "
            "Identify each distinct object and output a list of <point> elements in this format: "
            '<point x="X" y="Y" alt="Object description"/>. '
            "Use normalized coordinates from 0 to 100. "
            "Only provide valid points in the specified format."
        )

        # Mimic the test script: process the image without specifying return_tensors="pt"
        inputs = self.processor.process(images=[image], text=prompt)
        # Add one batch dimension to each tensor
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # Generate text output using Molmo
        generated_ids = self.generate_ids(inputs)
        generated_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract points and corresponding descriptions
        image_w, image_h = image.size
        extracted_data = extract_points_and_descriptions(generated_text, image_w, image_h)

        # Structure output as a list of dictionaries (if any points were extracted)
        if extracted_data:
            structured_output = [{
                "points": [entry["points"] for entry in extracted_data],
                "caption": generated_text
            }]
        else:
            structured_output = [{
                "points": [],
                "caption": generated_text
            }]

        return structured_output



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
        self.sam2_model.set_image(image)
        if use_points:
            # interpret prompts as points
            pts = np.array(prompts, dtype=int).reshape(-1, 2)
            labels = np.ones(len(pts), dtype=int)
            masks, scores, _ = self.sam2_model.predict(
                point_coords=pts,
                point_labels=labels,
                box=None,
                multimask_output=False
            )
        else:
            # interpret prompts as bounding boxes
            bboxes = np.array(prompts, dtype=int).reshape(-1, 4)
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
    If captioner_type="florence", produce bounding boxes (use_points=False).
    If captioner_type="molmo", produce points (use_points=True).
    Then refine with SAM2.
    """
    def __init__(
        self,
        captioner_type="florence",
        segmenter_model="facebook/sam2-hiera-small",
        device=None
    ):
        if captioner_type == "florence":
            self.caption_localizer = FlorenceCaptionLocalizer(device=device)
            self.use_points = False
        elif captioner_type == "molmo":
            self.caption_localizer = MolmoCaptionLocalizer(device=device)
            self.use_points = True
        else:
            raise ValueError(f"Unknown captioner_type={captioner_type}")

        self.location_refiner = LocationRefiner(
            model_name=segmenter_model,
            device=device
        )

    def run(self, image):
        preds = self.caption_localizer.run(image)
        all_masks = []
        all_prompts = []
        all_captions = []

        for pred in preds:
            # If Molmo => pred["points"], else Florence => pred["bboxes"]
            if self.use_points:
                prompts = pred.get("points", [])
            else:
                bboxes = pred.get("bboxes", [])
                if len(bboxes) > 0 and isinstance(bboxes[0], (int,float)):
                    # single box
                    prompts = [bboxes]
                else:
                    prompts = bboxes
            caption = pred["caption"]

            # refine with SAM2
            masks, scores = self.location_refiner.run(
                image, prompts, use_points=self.use_points
            )
            mask_uint8 = (masks[0].astype(np.uint8)*255) if len(masks)>0 else None

            all_masks.append(mask_uint8)
            all_prompts.append(prompts)
            all_captions.append(caption)

        return all_masks, all_prompts, all_captions

    def apply_transform(self, example, images):
        """
        For applying to a huggingface dataset row or batch.
        """
        is_batched = isinstance(example[images], list) and \
                     isinstance(example[images][0], (list, Image.Image))

        if is_batched:
            all_masks_list, all_prompts_list, all_captions_list = [], [], []
            for img_list in example[images]:
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

