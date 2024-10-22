import torch
import numpy as np
import random
import spacy
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForCausalLM, AutoProcessor


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
            if chunk.root.head == head or chunk.root.dep_ == 'attr':
                descriptions.append(chunk.text.lower())
        return descriptions

    def caption_refiner(self, caption):
        doc = self.nlp(caption)
        subject, action_verb = self.find_subject(doc)
        if action_verb:
            descriptions = self.extract_descriptions(doc, action_verb)
            return ', '.join(descriptions)
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

        for i in range(len(data['bboxes'])):
            current_box = data['bboxes'][i]
            current_label = data['labels'][i]
            is_duplicate = False

            for j in range(len(filtered_bboxes)):
                if current_label == filtered_labels[j]: 
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_bboxes.append(current_box)
                filtered_labels.append(current_label)

        return {'bboxes': filtered_bboxes, 'labels': filtered_labels, 'caption': data['caption']}

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
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        captions = parsed_answer[task].split(".")

        for caption in captions:
            if caption:
                task = "<CAPTION_TO_PHRASE_GROUNDING>"
                prompt = f"{task} {caption}"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, task=task, image_size=(image.width, image.height)
                )
                caption_bbox = parsed_answer[task]

                # Normalize the caption using Spacy (to remove duplicates like "apple" and "apples")
                caption_bbox["caption"] = self.normalize_caption(caption)
                caption_bbox = self.bbox_dedupe(caption_bbox)

                if len(caption_bbox['bboxes']) > 1:
                    flip = random.choice(['heads', 'tails'])
                    if flip == 'heads':
                        idx = random.randint(1, len(caption_bbox['bboxes']) - 1)
                    else:
                        idx = 0
                    if idx > 0:
                        caption_bbox['caption'] = caption_bbox['labels'][idx].lower() + ' with ' + caption_bbox['labels'][0].lower()
                    caption_bbox['bboxes'] = [caption_bbox['bboxes'][idx]]
                    caption_bbox['labels'] = [caption_bbox['labels'][idx]]

                captioned_bboxes.append(caption_bbox)

        # Filter out the largest bounding box if it encompasses all others
        captioned_bboxes = self.filter_large_bbox(captioned_bboxes)

        # Calculate distances between bounding boxes and select the top 3 distinct ones
        bboxes = [item['bboxes'][0] for item in captioned_bboxes]
        n = len(bboxes)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = 1 - self.compute_iou(bboxes[i], bboxes[j])

        scores = np.sum(distance_matrix, axis=1)
        selected_indices = np.argsort(scores)[-3:]
        captioned_bboxes = [{"bboxes": captioned_bboxes[i]['bboxes'][0], "caption": captioned_bboxes[i]['caption']} for i in selected_indices]

        return captioned_bboxes

    def filter_large_bbox(self, captioned_bboxes):
        """
        Filters out the largest bounding box if it encapsulates all others, assuming it's the entire image.
        """
        if len(captioned_bboxes) > 1:
            # Ensure we have valid bounding boxes
            valid_bboxes = [item for item in captioned_bboxes if 'bboxes' in item and item['bboxes']]
            if len(valid_bboxes) > 1:
                largest_idx = self.get_largest_bbox_idx([item['bboxes'][0] for item in valid_bboxes])
                largest_bbox = valid_bboxes[largest_idx]['bboxes'][0]

                # Check if all other bounding boxes are inside the largest one
                if all(self.is_bbox_inside(bbox['bboxes'][0], largest_bbox)
                       for i, bbox in enumerate(valid_bboxes) if i != largest_idx):
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
            inner_bbox[0] >= outer_bbox[0] and
            inner_bbox[1] >= outer_bbox[1] and
            inner_bbox[2] <= outer_bbox[2] and
            inner_bbox[3] <= outer_bbox[3]
        )


    def get_largest_bbox_idx(self, bboxes):
        """
        Returns the index of the largest bounding box by area.
        """
        return max(range(len(bboxes)), key=lambda i: (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1]))




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
        is_batched = isinstance(example[images], list) and isinstance(example[images][0], (list, Image.Image))

        try:
            if is_batched:
                all_masks = []
                all_bboxes = []
                all_captions = []

                for img_list in example[images]:
                    image = img_list[0] if isinstance(img_list, list) else img_list

                    if not isinstance(image, Image.Image):
                        raise ValueError(f"Expected a PIL image but got {type(image)}")

                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    masks, bboxes, captions = self.run(image)
                    all_masks.append(masks)
                    all_bboxes.append(bboxes)
                    all_captions.append(captions)

                example['masks'] = all_masks
                example['bboxes'] = all_bboxes
                example['captions'] = all_captions

            else:
                image = example[images][0] if isinstance(example[images], list) else example[images]

                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected a PIL image but got {type(image)}")

                if image.mode != 'RGB':
                    image = image.convert('RGB')

                masks, bboxes, captions = self.run(image)
                example['masks'] = masks
                example['bboxes'] = bboxes
                example['captions'] = captions

        except Exception as e:
            print(f"Error processing image, skipping: {e}")
            if is_batched:
                example['masks'] = [None] * len(example[images])
                example['bboxes'] = [None] * len(example[images])
                example['captions'] = [None] * len(example[images])
            else:
                example['masks'] = None
                example['bboxes'] = None
                example['captions'] = None

        return example
