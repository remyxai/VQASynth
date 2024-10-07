import torch
import numpy as np

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForCausalLM, AutoProcessor

class FlorenceSeg:
    def __init__(self, model_name="microsoft/Florence-2-large", device="cuda"):
        self.device = device
        self.torch_dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)

    def run_inference(self, image):
        final_answers = []
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
        captions = parsed_answer[task].split('.')
        
        for caption in captions:
            if caption:
                task = "<CAPTION_TO_PHRASE_GROUNDING>"
                prompt = f"{task} {caption}"


                url = "https://remyx.ai/assets/spatialvlm/warehouse_rgb.jpg?download=true"
                image = Image.open("/content/pliers.png").convert("RGB") #requests.get(url, stream=True).raw)

                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

                parsed_answer = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
                final_answers.append(parsed_answer[task])

        return final_answers

class SAM2:
    def __init__(self, model_name="facebook/sam2-hiera-large", device="cuda"):
        self.device = device
        self.sam2_model = SAM2ImagePredictor.from_pretrained(model_name, device=self.device)

    def run_inference_from_points(self, image, input_box):
        input_box = np.array(list(map(int, input_box['bboxes'][0])))
        self.sam2_model.set_image(image)
        masks, scores, _ = self.sam2_model.predict(
            point_coords=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks.astype(bool)

def find_medoid_and_closest_points(points, num_closest=5):
    """
    Find the medoid from a collection of points and the closest points to the medoid.

    Parameters:
    points (np.array): A numpy array of shape (N, D) where N is the number of points and D is the dimensionality.
    num_closest (int): Number of closest points to return.

    Returns:
    np.array: The medoid point.
    np.array: The closest points to the medoid.
    """
    distances = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis=-1))
    distance_sums = distances.sum(axis=1)
    medoid_idx = np.argmin(distance_sums)
    medoid = points[medoid_idx]
    sorted_indices = np.argsort(distances[medoid_idx])
    closest_indices = sorted_indices[1:num_closest + 1]
    return medoid, points[closest_indices]

def sample_points_from_heatmap(heatmap, original_size, num_points=5, percentile=0.95):
    """
    Sample points from the given heatmap, focusing on areas with higher values.
    """
    width, height = original_size
    threshold = np.percentile(heatmap.numpy(), percentile)
    masked_heatmap = torch.where(heatmap > threshold, heatmap, torch.tensor(0.0))
    probabilities = torch.softmax(masked_heatmap.flatten(), dim=0)

    attn = torch.sigmoid(heatmap)
    w = attn.shape[0]
    sampled_indices = torch.multinomial(torch.tensor(probabilities.ravel()), num_points, replacement=True)

    sampled_coords = np.array(np.unravel_index(sampled_indices, attn.shape)).T
    medoid, sampled_coords = find_medoid_and_closest_points(sampled_coords)
    pts = []
    for pt in sampled_coords.tolist():
        x, y = pt
        x = height * x / w
        y = width * y / w
        pts.append([y, x])
    return pts

def sample_points_from_bbox(bbox, num_points=5):
    """
    Sample points from the given bounding box and find the medoid and closest points.

    Parameters:
    bbox (list or tuple): Bounding box coordinates [xmin, ymin, xmax, ymax].
    num_points (int): Number of points to sample.

    Returns:
    list: A list of sampled points with coordinates [x, y].
    """
    xmin, ymin, xmax, ymax = bbox

    # Sample random points within the bounding box
    sampled_x = np.random.uniform(xmin, xmax, num_points)
    sampled_y = np.random.uniform(ymin, ymax, num_points)
    sampled_points = np.vstack((sampled_x, sampled_y)).T  # Shape (num_points, 2)

    # Find the medoid and closest points
    medoid, closest_points = find_medoid_and_closest_points(sampled_points)

    # Return as a list of [x, y] coordinates
    return [list(pt) for pt in closest_points]


def apply_mask_to_image(image, mask):
    """
    Apply a binary mask to an image. The mask should be a binary array where the regions to keep are True.
    """
    masked_image = image.copy()
    for c in range(masked_image.shape[2]):
        masked_image[:, :, c] = masked_image[:, :, c] * mask
    return masked_image
