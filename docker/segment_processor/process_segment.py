import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
from vqasynth.datasets.segment import CLIPSeg, SAM, sample_points_from_heatmap
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

device = "cuda" 
assert torch.cuda.is_available()
torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
sam = SAM(model_name="facebook/sam-vit-huge", device="cuda")

def florence_caption(image):
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

    prompt = parsed_answer[prompt]

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(image.width, image.height))
    
    parsed_answer = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']
    bboxes = parsed_answer['bboxes']
    labels = parsed_answer['labels']
    return list(zip(bboxes, labels))


def segment_image_data(row):
    try:

        caption_points = florence_caption(row["image"])
        sam_masks = []


        original_size = row["image"].size

        for idx in range(len(caption_points)):
            mask_tensor = sam.run_inference_from_points(row["image"], [caption_points[idx]])
            mask = cv2.cvtColor(255 * mask_tensor[0].numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            sam_masks.append(mask)

        return sam_masks
    except:
        return []

def main(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(output_dir, filename)
            # Load the DataFrame from the .pkl file
            df = pd.read_pickle(pkl_path)

            # Process each image and add the results to a new column
            df['masks'] = df.apply(segment_image_data, axis=1)

            # Save the updated DataFrame back to the .pkl file
            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to directory containing .pkl files")
    args = parser.parse_args()

    main(args.output_dir)
