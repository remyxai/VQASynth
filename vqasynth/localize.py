import torch
import numpy as np
import random

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForCausalLM, AutoProcessor


class CaptionLocalizer:
    def __init__(self, model_name="microsoft/Florence-2-large"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)

    def run(self, image):
        """
        Extract captioned bounding boxes of objects found in the scene.

        Args:
            image: A PIL Image

        Returns:
            list: A list of dicitonaries containing bounding boxes and captions for objects found

        """
        captioned_bboxes = []
        task = "<MORE_DETAILED_CAPTION>"
        prompt = f"{task}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        captions = parsed_answer[task].split(".")

        for caption in captions:
            if caption:
                task = "<CAPTION_TO_PHRASE_GROUNDING>"
                prompt = f"{task} {caption}"
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, task=task, image_size=(image.width, image.height)
                )
                captioned_bboxes.append(parsed_answer[task])

        return captioned_bboxes


class LocationRefiner:
    def __init__(self, model_name="facebook/sam2-hiera-large"):
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
            final_bboxes = []
            final_captions = []

            original_size = image.size

            object_counter = 1
            for pred in preds:
                bboxes = pred.get("bboxes", [])
                captions = pred.get("labels", [])

                if bboxes and captions and len(bboxes) == len(captions):
                    random_index = random.randint(0, len(bboxes) - 1)
                    selected_bbox = bboxes[random_index]
                    selected_caption = captions[random_index]

                    mask_tensor = self.location_refiner.run(image, selected_bbox)
                    mask = mask_tensor[0]
                    mask_uint8 = (mask.astype(np.uint8)) * 255
                    sam_masks.append(mask_uint8)

                    final_bboxes.append(selected_bbox)
                    final_captions.append(selected_caption)

            return sam_masks, final_bboxes, final_captions
        except Exception as e:
            print(f"Error during localization: {str(e)}")
            return [], [], []

    def apply_transform(self, example, images):
        """
        Process a single row in the dataset, adding masks, bboxes, and captions.

        Args:
            example: A single example from the dataset.
            images: The column in the dataset containing the images.

        Returns:
            Updated example with masks, bboxes, and captions.
        """
        depth_map, focallength = self.run(example[images])
        example['depth_map'] = depth_map
        example['focallength'] = focallength
        return example
