import os
import cv2
import pickle
import random
import numpy as np
import open3d as o3d
from pathlib import Path
import PIL
from PIL import Image


class SpatialSceneConstructor:
    def __init__(self):
        pass

    def save_pointcloud(self, pcd, file_path):
        """
        Save a point cloud to a file using Open3D.
        """
        o3d.io.write_point_cloud(file_path, pcd)

    def restore_pointclouds(self, pointcloud_paths):
        if len(pointcloud_paths) == 1 and isinstance(pointcloud_paths[0], list):
            pointcloud_paths = pointcloud_paths[0]

        restored_pointclouds = []
        for path in pointcloud_paths:
            restored_pointclouds.append(o3d.io.read_point_cloud(path))

        return restored_pointclouds

    def apply_mask_to_image(self, image, mask):
        """
        Apply a binary mask to an image. The mask should be a binary array where the regions to keep are True.
        """
        masked_image = image.copy()
        for c in range(masked_image.shape[2]):
            masked_image[:, :, c] = masked_image[:, :, c] * mask
        return masked_image

    def create_point_cloud_from_rgbd(
        self, rgb_image, depth_image, intrinsic_parameters
    ):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image),
            o3d.geometry.Image(depth_image),
            depth_scale=10.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            intrinsic_parameters["width"],
            intrinsic_parameters["height"],
            intrinsic_parameters["fx"],
            intrinsic_parameters["fy"],
            intrinsic_parameters["cx"],
            intrinsic_parameters["cy"],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        return pcd

    def canonicalize_point_cloud(self, pcd, canonicalize_threshold=0.3):
        # Segment the largest plane, assumed to be the floor
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )

        canonicalized = False
        if len(inliers) / len(pcd.points) > canonicalize_threshold:
            canonicalized = True

            if np.dot(plane_model[:3], [0, 1, 0]) < 0:
                plane_model = -plane_model

            normal = plane_model[:3] / np.linalg.norm(plane_model[:3])

            new_y = normal
            new_x = np.cross(new_y, [0, 0, -1])
            new_x /= np.linalg.norm(new_x)
            new_z = np.cross(new_x, new_y)

            transformation = np.identity(4)
            transformation[:3, :3] = np.vstack((new_x, new_y, new_z)).T
            transformation[:3, 3] = -np.dot(
                transformation[:3, :3], pcd.points[inliers[0]]
            )

            pcd.transform(transformation)

            rotation_z_180 = np.array(
                [
                    [np.cos(np.pi), -np.sin(np.pi), 0],
                    [np.sin(np.pi), np.cos(np.pi), 0],
                    [0, 0, 1],
                ]
            )
            pcd.rotate(rotation_z_180, center=(0, 0, 0))

            return pcd, canonicalized, transformation
        else:
            return pcd, canonicalized, None

    def calculate_centroid(self, pcd):
        """Calculate the centroid of a point cloud."""
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        return centroid

    def calculate_relative_positions(self, centroids):
        """Calculate the relative positions between centroids of point clouds."""
        num_centroids = len(centroids)
        relative_positions_info = []

        for i in range(num_centroids):
            for j in range(i + 1, num_centroids):
                relative_vector = centroids[j] - centroids[i]

                distance = np.linalg.norm(relative_vector)
                relative_positions_info.append(
                    {
                        "pcd_pair": (i, j),
                        "relative_vector": relative_vector,
                        "distance": distance,
                    }
                )

        return relative_positions_info

    def get_bounding_box_height(self, pcd):
        """
        Compute the height of the bounding box for a given point cloud.

        Parameters:
        pcd (open3d.geometry.PointCloud): The input point cloud.

        Returns:
        float: The height of the bounding box.
        """
        aabb = pcd.get_axis_aligned_bounding_box()
        return aabb.get_extent()[1]

    def compare_bounding_box_height(self, pcd_i, pcd_j):
        """
        Compare the bounding box heights of two point clouds.

        Parameters:
        pcd_i (open3d.geometry.PointCloud): The first point cloud.
        pcd_j (open3d.geometry.PointCloud): The second point cloud.

        Returns:
        bool: True if the bounding box of pcd_i is taller than that of pcd_j, False otherwise.
        """
        height_i = self.get_bounding_box_height(pcd_i)
        height_j = self.get_bounding_box_height(pcd_j)

        return height_i > height_j

    def run(
        self,
        image_filename: str,
        image: PIL.Image,
        depth_map: PIL.Image,
        focal_length: float,
        masks: list,
        output_dir: str,
    ):
        """
        Process an image and depth map to create and save segmented point clouds for each mask.

        Args:
            image_filename (str): Name of the original image file.
            image (PIL.Image): Original RGB image.
            depth_map (PIL.Image): Depth map corresponding to the image.
            focal_length (float): Focal length of the camera.
            masks (list): List of masks, one for each segmented region.
            output_dir (str): Directory to save the point cloud files.

        Returns:
            tuple: (List of file paths to saved point clouds, boolean indicating if canonicalization was applied).
        """
        image = image.convert("RGB")
        width, height = image.size
        depth_array = (
            np.array(depth_map, dtype=np.uint16).astype(np.float32) / 1000.0
        )  # Convert to meters

        intrinsic_parameters = {
            "width": width,
            "height": height,
            "fx": focal_length,
            "fy": focal_length * height / width,
            "cx": width / 2,
            "cy": height / 2,
        }

        original_pcd = self.create_point_cloud_from_rgbd(
            np.array(image), depth_array, intrinsic_parameters
        )
        normed_pcd, canonicalized, transformation = self.canonicalize_point_cloud(
            original_pcd, canonicalize_threshold=0.3
        )

        output_pointcloud_dir = os.path.join(output_dir, "pointclouds")
        Path(output_pointcloud_dir).mkdir(parents=True, exist_ok=True)

        point_cloud_filepaths = []

        points = np.asarray(normed_pcd.points)
        colors = np.asarray(normed_pcd.colors)
        for idx, mask in enumerate(masks):
            print(f"[INFO] Processing mask {idx + 1}/{len(masks)}")

            mask = np.array(mask, dtype=bool).squeeze()
            assert mask.shape == (
                height,
                width,
            ), f"Mask shape {mask.shape} does not match image dimensions {(height, width)}"

            depth_flat = depth_array.ravel()
            mask_flat = mask.ravel()

            valid_depth_indices = np.where(depth_flat > 0)[0]
            valid_mask_indices = valid_depth_indices[
                np.where(mask_flat[valid_depth_indices])[0]
            ]

            if len(valid_mask_indices) == 0:
                print(f"[WARNING] Mask {idx + 1} produced no valid points, skipping.")
                continue

            masked_points = points[valid_mask_indices]
            masked_colors = colors[valid_mask_indices]

            masked_pcd = o3d.geometry.PointCloud()
            masked_pcd.points = o3d.utility.Vector3dVector(masked_points)
            masked_pcd.colors = o3d.utility.Vector3dVector(masked_colors)

            if masked_pcd.is_empty():
                print(f"[WARNING] Masked point cloud is empty for mask {idx + 1}.")
                continue

            if canonicalized:
                masked_pcd.transform(transformation)

            if masked_pcd.is_empty():
                print(f"[WARNING] Transformed point cloud is empty for mask {idx + 1}.")
                continue

            pointcloud_filepath = os.path.join(
                output_pointcloud_dir,
                f"pointcloud_{Path(image_filename).stem}_{idx}.pcd",
            )
            o3d.io.write_point_cloud(pointcloud_filepath, masked_pcd)
            point_cloud_filepaths.append(pointcloud_filepath)

        return point_cloud_filepaths, canonicalized

    def apply_transform(self, example, idx, output_dir, images):
        """
        Process one or more rows of the dataset to generate point clouds and canonicalization status.

        Args:
            example (dict): A single example or a batch of examples from the dataset.
            idx (int or list): The index or indices of the current example(s).
            output_dir (str): The directory where the output point clouds will be saved.
            images (str): The column containing image data.

        Returns:
            dict: Updated example(s) with point clouds and canonicalization status.
        """
        is_batched = isinstance(example[images], list) and isinstance(
            example[images][0], (list, Image.Image)
        )

        try:
            if is_batched:
                pointclouds = []
                canonicalization_status = []

                for i, img_list in enumerate(example[images]):
                    if isinstance(img_list, list):
                        image = (
                            img_list[0]
                            if isinstance(img_list[0], Image.Image)
                            else img_list
                        )
                    else:
                        image = img_list

                    if not isinstance(image, Image.Image):
                        raise ValueError(
                            f"Expected a PIL image but got {type(image)} at index {i}"
                        )
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    pcd_data, canonicalized = self.run(
                        str(idx[i]),
                        image,
                        example["depth_map"][i],
                        example["focallength"][i],
                        example["masks"][i],
                        output_dir,
                    )
                    pointclouds.append(pcd_data)
                    canonicalization_status.append(canonicalized)

                example["pointclouds"] = pointclouds
                example["is_canonicalized"] = canonicalization_status

            else:
                image = (
                    example[images][0]
                    if isinstance(example[images], list)
                    else example[images]
                )

                if not isinstance(image, Image.Image):
                    raise ValueError("The image is not a valid PIL image.")
                if image.mode != "RGB":
                    image = image.convert("RGB")

                pcd_data, canonicalized = self.run(
                    str(idx),
                    image,
                    example["depth_map"],
                    example["focallength"],
                    example["masks"],
                    output_dir,
                )

                example["pointclouds"] = [pcd_data]
                example["is_canonicalized"] = [canonicalized]

        except Exception as e:
            print(f"Error processing image, skipping: {e}")
            if is_batched:
                example["pointclouds"] = [None] * len(example[images])
                example["is_canonicalized"] = [None] * len(example[images])
            else:
                example["pointclouds"] = [None]
                example["is_canonicalized"] = [None]

        return example
