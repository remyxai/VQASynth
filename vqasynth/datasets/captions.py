import os
import cv2
import gc
import sys
import math
import torch
import json
import base64
import argparse
import numpy as np
import open3d as o3d

import matplotlib
import matplotlib.cm
from matplotlib import pyplot as plt
from PIL import Image

# llava v1.6
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_llava():
    chat_handler = Llava15ChatHandler(clip_model_path="mmproj-model-f16.gguf", verbose=True)
    llm = Llama(model_path="llava-v1.6-34b.Q4_K_M.gguf",chat_handler=chat_handler,n_ctx=2048,logits_all=True, n_gpu_layers=-1)
    return llm

def extract_descriptions_from_incomplete_json(json_like_str):
    last_object_idx = json_like_str.rfind(',"object')

    if last_object_idx != -1:
        json_str = json_like_str[:last_object_idx] + '}'
    else:
        json_str = json_like_str.strip()
        if not json_str.endswith('}'):
            json_str += '}'

    try:
        json_obj = json.loads(json_str)
        descriptions = [details['description'].replace(".","") for key, details in json_obj.items() if 'description' in details]

        return descriptions
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
