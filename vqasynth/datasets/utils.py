import io
import os
import clip
import torch
import base64
import numpy as np
import matplotlib.cm
from PIL import Image

class EmbeddingFilter:
    def __init__(self, model_name='ViT-B/32', device=None):
        """Initialize the CLIP model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, self.device)

    def generate_image_embeddings(self, image: Image.Image):
        """
        Generate CLIP embeddings for an image.

        Args:
            image (PIL.Image.Image): The input image for which embeddings are generated.

        Returns:
            torch.Tensor: Normalized CLIP embeddings for the image.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    def get_best_matching_tag(self, image_embeddings: np.ndarray, tags: list):
        """
        Get the tag with the highest confidence match for the given image embeddings.

        Args:
            image_embeddings (np.ndarray): Precomputed embeddings for the image as a NumPy array.
            tags (list): List of tags to compare with the image embeddings.

        Returns:
            str: The tag with the highest confidence score.
        """
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {tag}") for tag in tags]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_embeddings_tensor = torch.from_numpy(image_embeddings).to(self.device)
        similarity = (100.0 * image_embeddings_tensor @ text_features.T).softmax(dim=-1)

        best_index = similarity.argmax().item()
        best_tag = tags[best_index]

        return best_tag

    def filter_by_tag(self, best_tag: str, include_tags: list, exclude_tags: list):
        """
        Filter the image based on the best-matching tag by comparing against the include/exclude lists.

        Args:
            best_tag (str): The tag with the highest confidence match.
            include_tags (list): Tags to include if present (optional).
            exclude_tags (list): Tags to exclude if present (optional).

        Returns:
            bool: True if the image passes filtering, False otherwise.
        """
        if exclude_tags and best_tag in exclude_tags:
            return False

        if include_tags and best_tag not in include_tags:
            return False

        return True


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
