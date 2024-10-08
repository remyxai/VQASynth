import os
import pickle
import random
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
        final_bboxes = []
        final_captions = []

        original_size = row["image"].size

        object_counter = 1
        for pred in preds:
            bboxes = pred.get("bboxes", [])
            captions = pred.get("labels", [])

            if bboxes and captions and len(bboxes) == len(captions):
                random_index = random.randint(0, len(bboxes) - 1)
                selected_bbox = bboxes[random_index]
                selected_caption = captions[random_index]

                mask_tensor = sam2.run_inference(row["image"], selected_bbox)
                mask = mask_tensor[0]
                mask_uint8 = (mask.astype(np.uint8)) * 255
                sam_masks.append(mask_uint8)

                final_bboxes.append(selected_bbox)
                final_captions.append(selected_caption)

        return sam_masks, final_bboxes, final_captions
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return [], [], []


def main(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith(".pkl"):
            pkl_path = os.path.join(output_dir, filename)
            df = pd.read_pickle(pkl_path)
            df[["masks", "bboxes", "captions"]] = pd.DataFrame(
                df.apply(lambda row: segment_image_data(row), axis=1).tolist(),
                index=df.index,
            )
            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images from .pkl files", add_help=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to directory containing .pkl files",
    )
    args = parser.parse_args()

    main(args.output_dir)
