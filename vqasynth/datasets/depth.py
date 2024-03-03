import sys
sys.path.append("./ZoeDepth")

import numpy as np
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from vqasynth.datasets.utils import colorize

# ZoeD_N
def load_zoe_depth():
    conf = get_config("zoedepth", "infer")
    depth_model = build_model(conf)
    return depth_model

def depth(img, depth_model):
    depth = depth_model.infer_pil(img)
    colored_depth = colorize(depth, cmap='gray_r')
    output_depth = Image.fromarray(colored_depth).convert('L')
    return output_depth
