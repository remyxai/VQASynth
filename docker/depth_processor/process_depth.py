import os
import pickle
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from vqasynth.datasets.depth import DepthEstimator

depth = DepthEstimator()

def depth_data(row):
    try:
        image_path = row["full_path"]
        depth_map, focallength_px = depth.run_inference(image_path)
        if depth_map.dtype != np.uint16:
            depth_map = (depth_map / np.max(depth_map) * 65535).astype(np.uint16)
            depth_image = Image.fromarray(depth_map, mode='I;16')

        return depth_image, focallength_px
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return [], 0


def main(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith(".pkl"):
            pkl_path = os.path.join(output_dir, filename)
            df = pd.read_pickle(pkl_path)
            df[["depth_map", "focallength"]] = pd.DataFrame(
                df.apply(lambda row: depth_data(row), axis=1).tolist(),
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
