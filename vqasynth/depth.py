import os
import cv2
import torch
import tempfile
import subprocess
import numpy as np
from PIL import Image

import depth_pro

def create_temp_image(pillow_image):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        pillow_image.save(temp_file, format='PNG')
        temp_file_name = temp_file.name
    return temp_file_name

def ensure_weights_exist(cache_location, checkpoint_url):
    """Ensure the model weights exist in a persistent cache location."""
    if not os.path.exists(cache_location):
        os.makedirs(cache_location, exist_ok=True)

    checkpoint_path = os.path.join(cache_location, "depth_pro.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Downloading weights...")
        subprocess.run(["wget", "-q", "-O", checkpoint_path, checkpoint_url], check=True)

    return checkpoint_path

class DepthEstimator:
    def __init__(self):
        """Initialize the model and ensure weights are in a consistent cache location."""
        # Define consistent cache location and checkpoint URL
        consistent_cache_location = os.path.join(os.path.expanduser("~"), ".depth_pro_cache", "checkpoints")
        checkpoint_url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"

        # Ensure weights exist in the cache location (persistent directory)
        checkpoint_path = ensure_weights_exist(consistent_cache_location, checkpoint_url)

        # Prepare the configuration for the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DEFAULT_MONODEPTH_CONFIG_DICT = depth_pro.depth_pro.DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=checkpoint_path,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )

        # Create the model and transformations
        self.model, self.transform = depth_pro.create_model_and_transforms(config=DEFAULT_MONODEPTH_CONFIG_DICT, device=device)
        self.model.eval()

    def run(self, image):
        """
        Returns a depth map and formatted focal length from an image

        Args:
            image: A PIL.Image image

        Returns:
            tuple: depth in meters, focal length in pixels
        """

        try:
            image_path = create_temp_image(image)
            image, _, f_px = depth_pro.load_rgb(image_path)
            image_tensor = self.transform(image)
            prediction = self.model.infer(image_tensor, f_px=f_px)
            depth = prediction["depth"].detach().cpu().numpy()
            depth_map = cv2.normalize(np.array(depth), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            focallength_px = prediction["focallength_px"] 
            focallength_px = focallength_px.item()

            if depth_map.dtype != np.uint16:
                depth_map = (depth_map / np.max(depth_map) * 65535).astype(np.uint16)
                depth_image = Image.fromarray(depth_map, mode='I;16')

            return depth_image, focallength_px
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            return [], 0

    def apply_transform(self, example, images):
        """
        Process a single row in the dataset, adding depth map and focal length.

        Args:
            example: A single example from the dataset.
            images: The column in the dataset containing the images.

        Returns:
            Updated example with depth map and focal length.
        """
        depth_map, focallength = self.run(example[images])
        example['depth_map'] = depth_map
        example['focallength'] = focallength
        return example
