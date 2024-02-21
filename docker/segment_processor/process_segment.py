import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
from vqasynth.datasets.segment import CLIPSeg, SAM, sample_points_from_heatmap


clipseg = CLIPSeg(model_name="CIDAS/clipseg-rd64-refined")
sam = SAM(model_name="facebook/sam-vit-huge", device="cuda")

def segment_image_data(row):
    preds = clipseg.run_inference(row["image"], row["captions"])

    sampled_points = []
    sam_masks = []

    original_size = row["image"].size

    for idx in range(preds.shape[0]):
        sampled_points.append(sample_points_from_heatmap(preds[idx][0], original_size, num_points=10))

    for idx in range(preds.shape[0]):
        mask_tensor = sam.run_inference_from_points(row["image"], [sampled_points[idx]])
        mask = cv2.cvtColor(255 * mask_tensor[0].numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        sam_masks.append(mask)

    return sam_masks

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
