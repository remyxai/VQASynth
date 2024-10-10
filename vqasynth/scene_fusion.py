import os
import cv2
import pickle
import random
import numpy as np
import open3d as o3d
from pathlib import Path


class SpatialSceneConstructor:
    def __init__(self):
        pass

    def save_pointcloud(self, pcd, file_path):
        """
        Save a point cloud to a file using Open3D.
        """
        o3d.io.write_point_cloud(file_path, pcd)

    def restore_pointclouds(self, pointcloud_paths):
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
            depth_scale=1.0,
            depth_trunc=10.0,
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

            # Ensure the plane normal points upwards
            if np.dot(plane_model[:3], [0, 1, 0]) < 0:
                plane_model = -plane_model

            # Normalize the plane normal vector
            normal = plane_model[:3] / np.linalg.norm(plane_model[:3])

            # Compute the new basis vectors
            new_y = normal
            new_x = np.cross(new_y, [0, 0, -1])
            new_x /= np.linalg.norm(new_x)
            new_z = np.cross(new_x, new_y)

            # Create the transformation matrix
            transformation = np.identity(4)
            transformation[:3, :3] = np.vstack((new_x, new_y, new_z)).T
            transformation[:3, 3] = -np.dot(
                transformation[:3, :3], pcd.points[inliers[0]]
            )

            # Apply the transformation
            pcd.transform(transformation)

            # Additional 180-degree rotation around the Z-axis
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

    def calculate_distances_between_point_clouds(self, A, B):
        dist_pcd1_to_pcd2 = np.asarray(A.compute_point_cloud_distance(B))
        dist_pcd2_to_pcd1 = np.asarray(B.compute_point_cloud_distance(A))
        combined_distances = np.concatenate((dist_pcd1_to_pcd2, dist_pcd2_to_pcd1))
        avg_dist = np.mean(combined_distances)
        return avg_dist

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
        return aabb.get_extent()[1]  # Assuming the Y-axis is the up-direction

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

    def run(self, image_filename, image, depth_map, focallength, masks, output_dir):
        original_image_cv = cv2.cvtColor(
            np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR
        )
        depth_image_cv = cv2.cvtColor(
            np.array(depth_map.convert("RGB")), cv2.COLOR_RGB2BGR
        )

        width, height = image.size
        intrinsic_parameters = {
            "width": width,
            "height": height,
            "fx": focallength,
            "fy": focallength,
            "cx": width / 2,
            "cy": height / 2,
        }

        point_clouds = []
        point_cloud_data = []

        original_pcd = self.create_point_cloud_from_rgbd(
            original_image_cv, depth_image_cv, intrinsic_parameters
        )
        pcd, canonicalized, transformation = self.canonicalize_point_cloud(
            original_pcd, canonicalize_threshold=0.3
        )

        for i, mask in enumerate(masks):
            mask_binary = mask > 0

            masked_rgb = self.apply_mask_to_image(original_image_cv, mask_binary)
            masked_depth = self.apply_mask_to_image(depth_image_cv, mask_binary)

            pcd = self.create_point_cloud_from_rgbd(
                masked_rgb, masked_depth, intrinsic_parameters
            )
            point_clouds.append(pcd)

        for idx, pcd in enumerate(point_clouds):
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            inlier_cloud = pcd.select_by_index(ind)
            if canonicalized:
                pcd.transform(transformation)
            pointcloud_filepath = os.path.join(
                output_dir,
                "pointclouds",
                f"pointcloud_{Path(image_filename).stem}_{idx}.pcd",
            )
            self.save_pointcloud(inlier_cloud, pointcloud_filepath)
            point_cloud_data.append(pointcloud_filepath)

        # Now, return both point_cloud_data and the canonicalized flag
        return point_cloud_data, canonicalized
