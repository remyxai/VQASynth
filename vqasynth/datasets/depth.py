import cv2
import numpy as np
from PIL import Image

import depth_pro

class DepthEstimator:
    def __init__(self):
        """Initialize the model and transforms."""
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()

    def run(self, image_path):
        """
        Returns a depth map and formatted focal length from an image

        Args:
            image_path: path to image

        Returns:
            tuple: depth in meters, focal length in pixels
        """

        try:
            image, _, f_px = depth_pro.load_rgb(image_path)
            image_tensor = self.transform(image)
            prediction = self.model.infer(image_tensor, f_px=f_px)
            depth = prediction["depth"]
            depth_map = cv2.normalize(np.array(depth), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            focallength_px = prediction["focallength_px"] 

            if depth_map.dtype != np.uint16:
                depth_map = (depth_map / np.max(depth_map) * 65535).astype(np.uint16)
                depth_image = Image.fromarray(depth_map, mode='I;16')

            return depth_image, focallength_px
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            return [], 0
