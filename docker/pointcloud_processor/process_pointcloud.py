import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
from vqasynth.datasets.segment import apply_mask_to_image
from vqasynth.datasets.pointcloud import create_point_cloud_from_rgbd, serialize_pointclouds


def pointcloud_image_data(row):
    original_image_cv = cv2.cvtColor(np.array(row["image"].convert('RGB')), cv2.COLOR_RGB2BGR)
    depth_image_cv = cv2.cvtColor(np.array(row["depth_map"].convert('RGB')), cv2.COLOR_RGB2BGR)

    width, height = row["image"].size
    intrinsic_parameters = {
        'width': width,
        'height': height,
        'fx': 1.5 * width,
        'fy': 1.5 * width,
        'cx': width / 2,
        'cy': height / 2,
    }

    point_clouds = []
    for i, mask in enumerate(row["masks"]):
        mask_binary = mask > 0

        masked_rgb = apply_mask_to_image(original_image_cv, mask_binary)
        masked_depth = apply_mask_to_image(depth_image_cv, mask_binary)

        masked_rgb_path = f'temp_masked_rgb_{i}.png'
        masked_depth_path = f'temp_masked_depth_{i}.png'
        cv2.imwrite(masked_rgb_path, masked_rgb)
        cv2.imwrite(masked_depth_path, masked_depth)

        pcd = create_point_cloud_from_rgbd(masked_rgb_path, masked_depth_path, intrinsic_parameters)
        point_clouds.append(pcd)

        os.remove(masked_rgb_path)
        os.remove(masked_depth_path)

    for idx, pcd in enumerate(point_clouds):
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        inlier_cloud = pcd.select_by_index(ind)
        point_clouds[idx] = inlier_cloud

    serialized_data = serialize_pointclouds(point_clouds)

    return serialized_data

def main(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(output_dir, filename)
            # Load the DataFrame from the .pkl file
            df = pd.read_pickle(pkl_path)

            # Process each image and add the results to a new column
            df['pointclouds'] = df.apply(pointcloud_image_data, axis=1)

            # Save the updated DataFrame back to the .pkl file
            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to directory containing .pkl files")
    args = parser.parse_args()

    main(args.output_dir)
