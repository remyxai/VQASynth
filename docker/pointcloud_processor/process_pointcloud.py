import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from vqasynth.datasets.segment import apply_mask_to_image
from vqasynth.datasets.pointcloud import create_point_cloud_from_rgbd, save_pointcloud, canonicalize_point_cloud

def pointcloud_image_data(row, output_dir):
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
    point_cloud_data = []

    original_pcd = create_point_cloud_from_rgbd(original_image_cv, depth_image_cv, intrinsic_parameters)
    pcd, canonicalized, transformation = canonicalize_point_cloud(original_pcd, canonicalize_threshold=0.3)

    for i, mask in enumerate(row["masks"]):
        mask_binary = mask > 0

        masked_rgb = apply_mask_to_image(original_image_cv, mask_binary)
        masked_depth = apply_mask_to_image(depth_image_cv, mask_binary)

        pcd = create_point_cloud_from_rgbd(masked_rgb, masked_depth, intrinsic_parameters)
        point_clouds.append(pcd)

    for idx, pcd in enumerate(point_clouds):
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        inlier_cloud = pcd.select_by_index(ind)
        if canonicalized:
            pcd.transform(transformation)
        pointcloud_filepath = os.path.join(output_dir, "pointclouds", f"pointcloud_{Path(row['image_filename']).stem}_{idx}.pcd")
        save_pointcloud(inlier_cloud, pointcloud_filepath)
        point_cloud_data.append(pointcloud_filepath)

    # Now, return both point_cloud_data and the canonicalized flag
    return point_cloud_data, canonicalized

def main(output_dir):
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
                pcd_data, canonicalized = pointcloud_image_data(row, output_dir)
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
