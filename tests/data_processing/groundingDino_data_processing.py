import gc
import sys
sys.path.append("./efficientvit")
sys.path.append("./segment_anything")
import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image
import litellm

# llava v1.6
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything.utils.transforms import ResizeLongestSide
# New efficientsam
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as TS

import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

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

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'tags': tags,
        'mask':[{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=False, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=False, default="groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    image_path = args.input_image
    split = args.split
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device

    
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize Recognize Anything Model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])
    
    # load model
    data_uri = image_to_base64_data_uri(image_path)

    chat_handler = Llava15ChatHandler(clip_model_path="mmproj-model-f16.gguf", verbose=True)
    llm = Llama(model_path="llava-v1.6-34b.Q4_K_M.gguf",chat_handler=chat_handler,n_ctx=2048,logits_all=True, n_gpu_layers=-1)
    res = llm.create_chat_completion(
         messages = [
             {"role": "system", "content": "You are an assistant who perfectly describes images."},
             {
                 "role": "user",
                 "content": [
                     {"type": "image_url", "image_url": {"url": data_uri}},
                     {"type" : "text", "text": 'Create a JSON representation where each entry consists of a key "object" with a numerical suffix starting from 1, and a corresponding "description" key with a value that is a concise, up to six-word sentence describing each main, distinct object or person in the image. Each pair should uniquely describe one element without repeating keys. An example: {"object1": { "description": "Man in red hat walking." },"object2": { "description": "Wooden pallet with boxes." },"object3": { "description": "Cardboard boxes stacked." },"object4": { "description": "Man in green vest standing." }}'}
                ]
             }
         ]
    )

    res_cleaned = list(set(extract_descriptions_from_incomplete_json(res["choices"][0]["message"]["content"])))
    tags = ",".join(res_cleaned)


    print("Image Tags: ", tags)
    del chat_handler
    del llm
    gc.collect()
    torch.cuda.empty_cache()


    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, tags, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    efficientvit_sam = create_sam_model(
      name="l0", weight_url="l0.pt",
    )
    efficientvit_sam = efficientvit_sam.cuda().eval()
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    efficientvit_sam_predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")

    target_length = 1024
    resizer = ResizeLongestSide(target_length)
    transformed_boxes = resizer.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = efficientvit_sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "automatic_label_output.jpg"), 
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, tags, masks, boxes_filt, pred_phrases)
