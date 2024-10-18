import torch
import numpy as np
import random
import spacy

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForCausalLM, AutoProcessor


nlp = spacy.load("en_core_web_sm")


class CaptionLocalizer:
    def __init__(self, model_name="microsoft/Florence-2-large", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)

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
        doc = nlp(caption)
        subject, action_verb = self.find_subject(doc)
        if action_verb:
            descriptions = self.extract_descriptions(doc, action_verb)
            return ', '.join(descriptions)
        else:
            return caption

    def compute_iou(self, box1, box2):
        # Extract the coordinates
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Compute the intersection rectangle
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        # Intersection width and height
        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)

        # Intersection area
        inter_area = inter_width * inter_height

        # Boxes areas
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Union area
        union_area = box1_area + box2_area - inter_area

        # Intersection over Union
        iou = inter_area / union_area if union_area != 0 else 0

        return iou

    def filter_objects(self, data, iou_threshold=0.5):
        filtered_bboxes = []
        filtered_labels = []

        for i in range(len(data['bboxes'])):
            current_box = data['bboxes'][i]
            current_label = data['labels'][i]
            is_duplicate = False

            for j in range(len(filtered_bboxes)):
                if current_label == filtered_labels[j] and self.compute_iou(current_box, filtered_bboxes[j]) > iou_threshold:
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
                caption_bbox = parsed_answer[task]
                caption_bbox["caption"] = self.caption_refiner(caption)
                caption_bbox = self.filter_objects(caption_bbox)

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

        # Final filtering based on bbox distances
        bboxes = [item['bboxes'][0] for item in captioned_bboxes]
        n = len(bboxes)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = 1 - self.compute_iou(bboxes[i], bboxes[j])

        scores = np.sum(distance_matrix, axis=1)
        selected_indices = np.argsort(scores)[-3:]
        captioned_bboxes = [{"bboxes" : captioned_bboxes[i]['bboxes'][0], "caption": captioned_bboxes[i]['caption']} for i in selected_indices]

        return captioned_bboxes


class LocationRefiner:
    def __init__(self, model_name="facebook/sam2-hiera-large", device=None):
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

    def calculate_iou(self, bbox1, bbox2):
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = bbox1_area + bbox2_area - inter_area

        # Avoid division by zero
        if union_area == 0:
            return 0.0

        return inter_area / union_area


    def filter_most_separated_objects(self, masks, bboxes, captions, top_k=3):
        num_objects = len(bboxes)

        # Return all objects if there are less than or equal to top_k objects
        if num_objects <= top_k:
            return masks, bboxes, captions

        # Calculate pairwise IoU for all bounding boxes
        iou_matrix = np.zeros((num_objects, num_objects))
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                iou_matrix[i, j] = self.calculate_iou(bboxes[i], bboxes[j])
                iou_matrix[j, i] = iou_matrix[i, j]  # IoU is symmetric

        # Get sum of IoU for each object (lower sum indicates more separated)
        iou_sums = iou_matrix.sum(axis=1)

        # Select top_k objects with the lowest IoU sums (most separated objects)
        selected_indices = np.argsort(iou_sums)[:top_k]

        # Filter masks, bboxes, and captions based on selected indices
        filtered_masks = [masks[i] for i in selected_indices]
        filtered_bboxes = [bboxes[i] for i in selected_indices]
        filtered_captions = [captions[i] for i in selected_indices]

        return filtered_masks, filtered_bboxes, filtered_captions

    def run(self, image):
        """
        Produce object location masks, bboxes, and captions

        Args:
            image: A PIL Image

        Returns:
            tuple: list of  boolean NumPy arrays representing produced segmentation masks for objects found, a list of bounding boxes, a list of captions

        """

        #try:
        preds = self.caption_localizer.run(image)
        sam_masks = []
        original_size = image.size
        all_bboxes = []
        all_captions = []


        object_counter = 1
        for pred in preds:
            bboxes = pred.get("bboxes", [])
            captions = pred.get("caption", [])

            if bboxes and captions and len(bboxes) == len(captions):
                for bbox in bboxes:
                    mask_tensor = self.location_refiner.run(image, bbox)
                    mask = mask_tensor[0]
                    mask_uint8 = (mask.astype(np.uint8)) * 255
                    sam_masks.append(mask_uint8)

                all_bboxes.extend(bboxes)
                all_captions.extend(captions)

        return sam_masks, all_bboxes, all_captions
        #except Exception as e:
        #    print(f"Error during localization: {str(e)}")
        #    return [], [], []

    def apply_transform(self, example, images):
        """
        Process a single row in the dataset, adding masks, bboxes, and captions.

        Args:
            example: A single example from the dataset.
            images: The column in the dataset containing the images.

        Returns:
            Updated example with masks, bboxes, and captions.
        """
        #try:
        if isinstance(example[images], list):
            image = example[images][0]
        else:
            image = example[images]
        masks, bboxes, captions = self.run(image)
        example['masks'] = masks
        example['bboxes'] = bboxes
        example['captions'] = captions
        #except Exception as e:
        #    print(f"Error processing image, skipping: {e}")
        #    example['masks'] = None
        #    example['bboxes'] = None
        #    example['captions'] = None
        return example
