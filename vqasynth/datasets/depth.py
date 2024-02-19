import sys
sys.path.append("./ZoeDepth")

import numpy as np
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# ZoeD_N
def load_zoe_depth():
    conf = get_config("zoedepth", "infer")
    depth_model = build_model(conf)
    return depth_model

def depth(img, depth_model):
    depth = depth_model.infer_pil(img)
    raw_depth = Image.fromarray((depth*256).astype('uint16'))
    return raw_depth
