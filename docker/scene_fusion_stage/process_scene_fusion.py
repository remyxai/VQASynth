import os
import pickle
import argparse
import pandas as pd
from vqasynth.datasets.scene_fusion import SpatialSceneConstructor


def main(output_dir):
    spatial_scene_constructor = SpatialSceneConstructor()

    point_cloud_dir = os.path.join(output_dir, "pointclouds")
    if not os.path.exists(point_cloud_dir):
        os.makedirs(point_cloud_dir)

    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(output_dir, filename)
            df = pd.read_pickle(pkl_path)

            # Initialize empty lists to hold the pointclouds and canonicalization flags
            pointclouds = []
            is_canonicalized = []

            # Update to process each row and append results to lists
            for index, row in df.iterrows():
                pcd_data, canonicalized = spatial_scene_constructor.run(row["image_filename"], row["image"], row["depth_map"], row["focallength"], row["masks"], output_dir)
                pointclouds.append(pcd_data)
                is_canonicalized.append(canonicalized)

            # Assign lists to new DataFrame columns
            df['pointclouds'] = pointclouds
            df['is_canonicalized'] = is_canonicalized

            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to directory containing .pkl files")
    args = parser.parse_args()

    main(args.output_dir)
