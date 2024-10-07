import cv2
import numpy as np
from PIL import Image
import depth_pro
from vqasynth.datasets.utils import colorize

class DepthEstimator:
    def __init__(self):
        """Initialize the model and transforms."""
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()

    def run_inference(self, image: Image.Image):
        """
        Takes a Pillow RGB image, preprocesses it, and returns the depth and focal length in pixels.

        Args:
            image (Image.Image): Pillow RGB image

        Returns:
            tuple: depth in meters, focal length in pixels
        """
        image_tensor = self.transform(image)
        _, _, f_px = depth_pro.load_rgb(image)
        prediction = self.model.infer(image_tensor, f_px=f_px)
        depth = prediction["depth"]
        depth_normalized = cv2.normalize(np.array(depth), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        focallength_px = prediction["focallength_px"] 

        return depth_normalized, focallength_px.item()
