import os
import pickle
import argparse
import numpy as np
import pandas as pd
from vqasynth.datasets.segment import Florence2, SAM2

florence2 = Florence2()
sam2 = SAM2()

def segment_image_data(row):
    try:
        preds = florence2.run_inference(row["image"])
        sam_masks = []

        original_size = row["image"].size

        for pred in preds:
            bboxes = pred['bboxes']

            for bbox in bboxes:
                mask_tensor = sam2.run_inference(row["image"], bbox)
                mask = mask_tensor[0]
                mask_uint8 = (mask.astype(np.uint8)) * 255
                sam_masks.append(mask_uint8)

        return sam_masks, preds
    except:
        return [], []

def main(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(output_dir, filename)
            df = pd.read_pickle(pkl_path)
            df[['masks', 'bboxes']] = df.apply(lambda row: pd.Series(segment_image_data(row)), axis=1)
            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to directory containing .pkl files")
    args = parser.parse_args()

    main(args.output_dir)
