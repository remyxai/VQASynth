import torch
import numpy as np

from transformers import SamModel, SamProcessor
from transformers import AutoProcessor, CLIPSegForImageSegmentation


class CLIPSeg:
    def __init__(self, model_name="CIDAS/clipseg-rd64-refined"):
        self.clipseg_processor = AutoProcessor.from_pretrained(model_name)
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(model_name)

    def run_inference(self, image, text_descriptions):
        inputs = self.clipseg_processor(text=text_descriptions, images=[image] * len(text_descriptions), padding=True, return_tensors="pt")
        outputs = self.clipseg_model(**inputs)
        logits = outputs.logits
        return logits.detach().unsqueeze(1)

class SAM:
    def __init__(self, model_name="facebook/sam-vit-huge", device="cuda"):
        self.device = device
        self.sam_model = SamModel.from_pretrained(model_name).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(model_name)

    def run_inference_from_points(self, image, points):
        sam_inputs = self.sam_processor(image, input_points=points, return_tensors="pt").to(self.device)
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)
        return self.sam_processor.image_processor.post_process_masks(sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu())

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


def apply_mask_to_image(image, mask):
    """
    Apply a binary mask to an image. The mask should be a binary array where the regions to keep are True.
    """
    masked_image = image.copy()
    for c in range(masked_image.shape[2]):
        masked_image[:, :, c] = masked_image[:, :, c] * mask
    return masked_image
