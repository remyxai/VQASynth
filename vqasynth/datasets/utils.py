import io
import os
import clip
import base64
import numpy as np
import torch
from PIL import Image

import matplotlib
import matplotlib.cm
from torchvision.datasets import CIFAR100

class ImageTagger:
    def __init__(self, model_name='ViT-B/32', device=None):
        """Initialize the CLIP model and load CIFAR100 dataset."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, self.device)
        self.cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.cifar100.classes]).to(self.device)

    def get_top_tags(self, image, top_k=5):
        """
        Run inference on an image and return the top K predicted tags as strings.

        Args:
            image (PIL.Image.Image): Input image for inference.
            top_k (int): Number of top predictions to return (default: 5).

        Returns:
            list: Top K predicted tags as strings.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(self.text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity[0].topk(top_k)

        return [self.cifar100.classes[index] for index in indices]

def image_to_base64_data_uri(image_input):
    # Check if the input is a file path (string)
    if isinstance(image_input, str):
        with open(image_input, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Check if the input is a PIL Image
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")  # You can change the format if needed
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    else:
        raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")
    
    return f"data:image/png;base64,{base64_data}"

def colorize(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.
    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.
    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img
