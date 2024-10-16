import os
import cv2
import torch
import tempfile
import subprocess
import numpy as np
from PIL import Image
import onnxruntime as ort
import depth_pro

def create_temp_image(pillow_image):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        pillow_image.save(temp_file, format='PNG')
        temp_file_name = temp_file.name
    return temp_file_name

def ensure_weights_exist(cache_location, checkpoint_url, model_name="depth_pro.pt"):
    """Ensure the model weights exist in a persistent cache location."""
    if not os.path.exists(cache_location):
        os.makedirs(cache_location, exist_ok=True)

    checkpoint_path = os.path.join(cache_location, model_name)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Downloading weights...")
        subprocess.run(["wget", "-q", "-O", checkpoint_path, checkpoint_url], check=True)

    return checkpoint_path

class DepthEstimator:
    def __init__(self, from_onnx=True):
        """Initialize the model and ensure weights are in a consistent cache location."""
        self.from_onnx = from_onnx
        
        consistent_cache_location = os.path.join(os.path.expanduser("~"), ".depth_pro_cache", "checkpoints")

        if self.from_onnx:
            onnx_url = "https://huggingface.co/onnx-community/DepthPro-ONNX/resolve/main/onnx/model_q4.onnx"
            onnx_model_name = "model_q4.onnx"
            
            onnx_model_path = ensure_weights_exist(consistent_cache_location, onnx_url, model_name=onnx_model_name)
            
            available_providers = ort.get_available_providers()
            providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(onnx_model_path, providers=providers)
            except:
                # Hard fallback to cpu
                self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

            self.transform = self._get_onnx_transform()
        else:
            checkpoint_url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
            checkpoint_path = ensure_weights_exist(consistent_cache_location, checkpoint_url)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            DEFAULT_MONODEPTH_CONFIG_DICT = depth_pro.depth_pro.DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                checkpoint_uri=checkpoint_path,
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            )

            self.model, self.transform = depth_pro.create_model_and_transforms(config=DEFAULT_MONODEPTH_CONFIG_DICT, device=device)
            self.model.eval()

    def _get_onnx_transform(self):
        """Get transformation for ONNX model."""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run(self, image):
        """
        Returns a depth map and formatted focal length from an image

        Args:
            image: A PIL.Image image

        Returns:
            tuple: depth in meters, focal length in pixels
        """
        try:
            if self.from_onnx:
                return self._run_onnx(image)
            else:
                return self._run_pytorch(image)
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            return [], 0

    def _run_onnx(self, image):
        """
        Run inference using ONNX model and return depth map and focal length.
        """
        original_size = image.size

        image_data = self.transform(image).unsqueeze(0).numpy()
        onnx_inputs = {self.session.get_inputs()[0].name: image_data}
        onnx_outputs = self.session.run(None, onnx_inputs)

        predicted_depth = onnx_outputs[0].squeeze()
        focallength_px = onnx_outputs[1].item()

        min_depth = predicted_depth.min()
        max_depth = predicted_depth.max()
        normalized_depth = (predicted_depth - min_depth) / (max_depth - min_depth)

        # Invert the normalized depth to correct the light/dark inversion
        inverted_depth = 1 - normalized_depth

        depth_map_resized = cv2.resize(inverted_depth, original_size, interpolation=cv2.INTER_LINEAR)

        depth_map_uint16 = (depth_map_resized * 65535).astype(np.uint16)
        depth_image_pil = Image.fromarray(depth_map_uint16, mode='I;16')

        return depth_image_pil, focallength_px

    def _run_pytorch(self, image):
        """
        Run inference using the PyTorch model and return depth map and focal length.
        """
        image_path = create_temp_image(image)
        image, _, f_px = depth_pro.load_rgb(image_path)
        image_tensor = self.transform(image)
        prediction = self.model.infer(image_tensor, f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy()
        depth_map = cv2.normalize(np.array(depth), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        focallength_px = prediction["focallength_px"].item()

        if depth_map.dtype != np.uint16:
            depth_map = (depth_map / np.max(depth_map) * 65535).astype(np.uint16)
            depth_image = Image.fromarray(depth_map, mode='I;16')

        return depth_image, focallength_px

    def apply_transform(self, example, images):
        """
        Process a single row in the dataset, adding depth map and focal length.

        Args:
            example: A single example from the dataset.
            images: The column in the dataset containing the images.

        Returns:
            Updated example with depth map and focal length, or empty values on failure.
        """
        try:
            if isinstance(example[images], list):
                image = example[images][0]
            else:
                image = example[images]

            if image.mode != "RGB":
                image = image.convert("RGB")

            depth_map, focallength = self.run(image)

            example['depth_map'] = depth_map
            example['focallength'] = focallength
        except Exception as e:
            print(f"Error processing image, skipping: {e}")
            example['depth_map'] = None
            example['focallength'] = None

        return example

