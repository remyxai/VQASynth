import re
import torch
import numpy as np
import random
import spacy
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
)


class CaptionLocalizer:
    def __init__(self, model_name="microsoft/Florence-2-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)
        self.nlp = spacy.load("en_core_web_sm")

    def find_subject(self, doc):
        for token in doc:
            if "subj" in token.dep_:
                return token.text, token.head
        return None, None

    def extract_descriptions(self, doc, head):
        descriptions = []
        for chunk in doc.noun_chunks:
            if chunk.root.head == head or chunk.root.dep_ == "attr":
                descriptions.append(chunk.text.lower())
        return descriptions

    def caption_refiner(self, caption):
        doc = self.nlp(caption)
        subject, action_verb = self.find_subject(doc)
        if action_verb:
            descriptions = self.extract_descriptions(doc, action_verb)
            return ", ".join(descriptions)
        else:
            return caption

    def compute_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)

        inter_area = inter_width * inter_height

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area != 0 else 0

        return iou

    def bbox_dedupe(self, data, iou_threshold=0.5):
        filtered_bboxes = []
        filtered_labels = []

        for i in range(len(data["bboxes"])):
            current_box = data["bboxes"][i]
            current_label = data["labels"][i]
            is_duplicate = False

            for j in range(len(filtered_bboxes)):
                if current_label == filtered_labels[j]:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_bboxes.append(current_box)
                filtered_labels.append(current_label)

        return {
            "bboxes": filtered_bboxes,
            "labels": filtered_labels,
            "caption": data["caption"],
        }

    def check_resize_image(self, image, max_dimension=512):
        """
        Resizes the image to ensure the largest dimension is at most max_dimension
        while maintaining aspect ratio.

        Parameters:
        - image (Image.Image): The PIL image to resize.
        - max_dimension (int): The maximum allowed size for the largest dimension. Default is 512.

        Returns:
        - Image.Image: The resized image, if resizing was needed; otherwise, the original image.
        """
        if max(image.size) <= max_dimension:
            return image

        scale_factor = max_dimension / max(image.size)
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        return image.resize(new_size, Image.LANCZOS)

    def run(self, image):
        """
        Extract captioned bounding boxes of objects found in the scene.

        Args:
            image: A PIL Image

        Returns:
            list: A list of dictionaries containing bounding boxes and captions for objects found.
        """
        captioned_bboxes = []
        task = "<MORE_DETAILED_CAPTION>"
        prompt = f"{task}"
        image = self.check_resize_image(image)

        # Function to handle generation with CPU fallback
        def safe_generate(inputs):
            try:
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_length=1024,
                    num_beams=1,
                    do_sample=False,
                )
                return generated_ids
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    print("CUDA error encountered, switching to CPU for this sample.")
                    torch.cuda.empty_cache()
                    self.model = self.model.to("cpu")
                    inputs = inputs.to("cpu")
                    return self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_length=1024,
                        num_beams=1,
                        do_sample=False,
                    )
                else:
                    raise e

        try:
            # Initial prompt and inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
                self.device, self.torch_dtype
            )

            # Generate detailed captions
            generated_ids = safe_generate(inputs)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            # Process the generated text
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=prompt, image_size=(image.width, image.height)
            )
            captions = parsed_answer.get(task, "").split(".")

            for caption in captions:
                if caption:
                    # For each caption, ground it to a phrase
                    task = "<CAPTION_TO_PHRASE_GROUNDING>"
                    prompt = f"{task} {caption}"

                    inputs = self.processor(
                        text=prompt, images=image, return_tensors="pt"
                    ).to(self.device, self.torch_dtype)
                    generated_ids = safe_generate(inputs)

                    generated_text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=False
                    )[0]
                    parsed_answer = self.processor.post_process_generation(
                        generated_text,
                        task=task,
                        image_size=(image.width, image.height),
                    )
                    caption_bbox = parsed_answer.get(task, {})

                    if "bboxes" not in caption_bbox or "labels" not in caption_bbox:
                        print("Skipping entry due to missing bounding boxes or labels.")
                        continue

                    # Normalize and deduplicate
                    caption_bbox["caption"] = self.normalize_caption(caption)
                    caption_bbox = self.bbox_dedupe(caption_bbox)

                    # Handle multiple bounding boxes
                    if len(caption_bbox["bboxes"]) > 1:
                        flip = random.choice(["heads", "tails"])
                        idx = (
                            random.randint(1, len(caption_bbox["bboxes"]) - 1)
                            if flip == "heads"
                            else 0
                        )
                        if idx > 0:
                            caption_bbox[
                                "caption"
                            ] = f"{caption_bbox['labels'][idx].lower()} with {caption_bbox['labels'][0].lower()}"
                        caption_bbox["bboxes"] = [caption_bbox["bboxes"][idx]]
                        caption_bbox["labels"] = [caption_bbox["labels"][idx]]

                    captioned_bboxes.append(caption_bbox)

            # Filter out the largest bounding box if it encompasses all others
            captioned_bboxes = self.filter_large_bbox(captioned_bboxes)

            # Calculate distances between bounding boxes and select the top 3 distinct ones
            if captioned_bboxes:
                bboxes = [item["bboxes"][0] for item in captioned_bboxes]
                n = len(bboxes)
                distance_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            distance_matrix[i][j] = 1 - self.compute_iou(
                                bboxes[i], bboxes[j]
                            )

                scores = np.sum(distance_matrix, axis=1)
                selected_indices = np.argsort(scores)[-3:]
                captioned_bboxes = [
                    {
                        "bboxes": captioned_bboxes[i]["bboxes"][0],
                        "caption": captioned_bboxes[i]["caption"],
                    }
                    for i in selected_indices
                ]

        except Exception as e:
            print(f"Error during localization: {str(e)}")
            return []

        return captioned_bboxes

    def filter_large_bbox(self, captioned_bboxes):
        """
        Filters out the largest bounding box if it encapsulates all others, assuming it's the entire image.
        """
        if len(captioned_bboxes) > 1:
            # Ensure we have valid bounding boxes
            valid_bboxes = [
                item for item in captioned_bboxes if "bboxes" in item and item["bboxes"]
            ]
            if len(valid_bboxes) > 1:
                largest_idx = self.get_largest_bbox_idx(
                    [item["bboxes"][0] for item in valid_bboxes]
                )
                largest_bbox = valid_bboxes[largest_idx]["bboxes"][0]

                # Check if all other bounding boxes are inside the largest one
                if all(
                    self.is_bbox_inside(bbox["bboxes"][0], largest_bbox)
                    for i, bbox in enumerate(valid_bboxes)
                    if i != largest_idx
                ):
                    valid_bboxes.pop(largest_idx)

            return valid_bboxes

        return captioned_bboxes

    def normalize_caption(self, caption):
        """
        Normalize the caption using Spacy to remove duplicates like "apple" and "apples".
        """
        doc = self.nlp(caption)
        normalized_nouns = set()
        final_caption = []

        for token in doc:
            if token.pos_ in ["NOUN", "ADJ"]:
                lemma = token.lemma_.lower()
                if lemma not in normalized_nouns:
                    normalized_nouns.add(lemma)
                    final_caption.append(lemma)

        return " ".join(final_caption)

    def is_bbox_inside(self, inner_bbox, outer_bbox):
        """
        Check if one bounding box is inside another.
        """
        return (
            inner_bbox[0] >= outer_bbox[0]
            and inner_bbox[1] >= outer_bbox[1]
            and inner_bbox[2] <= outer_bbox[2]
            and inner_bbox[3] <= outer_bbox[3]
        )

    def get_largest_bbox_idx(self, bboxes):
        """
        Returns the index of the largest bounding box by area.
        """
        return max(
            range(len(bboxes)),
            key=lambda i: (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1]),
        )


class LocationRefiner:
    def __init__(self, model_name="facebook/sam2-hiera-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_model = SAM2ImagePredictor.from_pretrained(
            model_name, trust_remote_code=True, device=self.device
        )

    def run(self, image, input_box):
        """
        Extract and refine object segmentation masks.

        Args:
            image: A PIL Image

        Returns:
            numpy.ndarray: A boolean NumPy array representing produced segmentation masks for objects found.

        """
        input_box = np.array(list(map(int, input_box)))
        self.sam2_model.set_image(image)
        masks, scores, _ = self.sam2_model.predict(
            point_coords=None, box=input_box[None, :], multimask_output=False
        )
        return masks.astype(bool)


class Localizer:
    def __init__(self):
        self.caption_localizer = CaptionLocalizer()
        self.location_refiner = LocationRefiner()

    def run(self, image):
        """
        Produce object location masks, bboxes, and captions

        Args:
            image: A PIL Image

        Returns:
            tuple: list of  boolean NumPy arrays representing produced segmentation masks for objects found, a list of bounding boxes, a list of captions

        """

        try:
            preds = self.caption_localizer.run(image)
            sam_masks = []
            original_size = image.size
            all_bboxes = []
            all_captions = []

            object_counter = 1
            for pred in preds:
                bbox = pred.get("bboxes", [])
                caption = pred.get("caption", [])

                mask_tensor = self.location_refiner.run(image, bbox)
                mask = mask_tensor[0]
                mask_uint8 = (mask.astype(np.uint8)) * 255
                sam_masks.append(mask_uint8)

                all_bboxes.append(bbox)
                all_captions.append(caption)

            return sam_masks, all_bboxes, all_captions
        except Exception as e:
            print(f"Error during localization: {str(e)}")
            return [], [], []

    def apply_transform(self, example, images):
        """
        Process one or more rows in the dataset, adding masks, bboxes, and captions.

        Args:
            example: A single example or a batch of examples from the dataset.
            images: The column in the dataset containing the images.

        Returns:
            Updated example(s) with masks, bboxes, and captions.
        """
        is_batched = isinstance(example[images], list) and isinstance(
            example[images][0], (list, Image.Image)
        )

        try:
            if is_batched:
                all_masks = []
                all_bboxes = []
                all_captions = []

                for img_list in example[images]:
                    image = img_list[0] if isinstance(img_list, list) else img_list

                    if not isinstance(image, Image.Image):
                        raise ValueError(f"Expected a PIL image but got {type(image)}")

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    masks, bboxes, captions = self.run(image)
                    all_masks.append(masks)
                    all_bboxes.append(bboxes)
                    all_captions.append(captions)

                example["masks"] = all_masks
                example["bboxes"] = all_bboxes
                example["captions"] = all_captions

            else:
                image = (
                    example[images][0]
                    if isinstance(example[images], list)
                    else example[images]
                )

                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected a PIL image but got {type(image)}")

                if image.mode != "RGB":
                    image = image.convert("RGB")

                masks, bboxes, captions = self.run(image)
                example["masks"] = masks
                example["bboxes"] = bboxes
                example["captions"] = captions

        except Exception as e:
            print(f"Error processing image, skipping: {e}")
            if is_batched:
                example["masks"] = [None] * len(example[images])
                example["bboxes"] = [None] * len(example[images])
                example["captions"] = [None] * len(example[images])
            else:
                example["masks"] = None
                example["bboxes"] = None
                example["captions"] = None

        return example


class Captioner:
    def generate_caption(self, image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")


class MolmoCaptioner(Captioner):
    def __init__(self, model_name="allenai/Molmo-7B-O-0924"):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def generate_caption(self, image, prompt):
        inputs = self.processor.process(images=[image], text=prompt)
        inputs["images"] = inputs["images"].to(torch.bfloat16)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
                tokenizer=self.processor.tokenizer,
            )
            generated_tokens = output[0, inputs["input_ids"].size(1) :]
            return self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )


class FlorenceCaptioner(Captioner):
    def __init__(self, model_name="microsoft/Florence-2-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to(self.device)

    def generate_caption(self, image, prompt):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_length=1024,
                num_beams=1,
                do_sample=False,
            )
            return self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]


class SAM2Segmenter:
    def __init__(
        self,
        model_name="sam2_hiera_l.yaml",
        checkpoint="../checkpoints/sam2_hiera_large.pt",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SAM2ImagePredictor.from_pretrained(
            model_name, trust_remote_code=True, device=self.device
        )

    def segment_with_points(self, image, points):
        self.predictor.set_image(image)
        input_labels = np.ones(len(points))
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=input_labels,
            box=None,
            multimask_output=False,
        )
        return masks, scores

    def segment_with_bboxes(self, image, bboxes):
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=None, box=np.array(bboxes), multimask_output=False
        )
        return masks, scores


def extract_points_from_caption(caption, image_w, image_h):
    """Parse points from caption."""
    points = []
    for match in re.finditer(
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', caption
    ):
        try:
            point = [float(match.group(1)), float(match.group(2))]
            if max(point) > 100:  # Invalid if values exceed 100
                continue
            point = np.array(point) / 100.0  # Normalize to [0, 1]
            points.append(point * np.array([image_w, image_h]))
        except ValueError:
            continue
    return points


class Localizer:
    def __init__(
        self,
        captioner_type="molmo",
        sam2_cfg="sam2_hiera_l.yaml",
        sam2_checkpoint="../checkpoints/sam2_hiera_large.pt",
    ):
        if captioner_type == "molmo":
            self.captioner = MolmoCaptioner()
        elif captioner_type == "florence":
            self.captioner = FlorenceCaptioner()
        else:
            raise ValueError("Unsupported captioner type.")
        self.segmenter = SAM2Segmenter(sam2_cfg, sam2_checkpoint)

    def run(
        self,
        image,
        prompt="Point to the object of interest in the scene.",
        use_points=True,
    ):
        """Run the localization pipeline."""
        # Generate caption and points
        caption = self.captioner.generate_caption(image, prompt)
        w, h = image.size
        points = extract_points_from_caption(caption, w, h) if use_points else None

        if use_points and points:
            masks, scores = self.segmenter.segment_with_points(image, points)
        elif not use_points:
            # Use bounding boxes (if bbox extraction logic is added in caption parsing)
            bboxes = []  # Replace with bbox parsing logic
            masks, scores = self.segmenter.segment_with_bboxes(image, bboxes)
        else:
            masks, scores = [], []

        return {"masks": masks, "scores": scores, "points": points, "caption": caption}
