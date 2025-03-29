import os
import numpy as np
import torch
import open3d as o3d
import cv2
from pathlib import Path
import tempfile
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    major_cap = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if major_cap >= 8 else torch.float16
else:
    dtype = torch.float32


def restore_pointclouds(pointcloud_paths):
    if len(pointcloud_paths) == 1 and isinstance(pointcloud_paths[0], list):
        pointcloud_paths = pointcloud_paths[0]

    restored_pointclouds = []
    for path in pointcloud_paths:
        restored_pointclouds.append(o3d.io.read_point_cloud(path))

    return restored_pointclouds

class SpatialSceneConstructor:
    def __init__(self):
        """
        Initialize the constructor and load the VGGT model.
        """
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self.model.eval()

    def save_pointcloud(self, pcd, file_path):
        o3d.io.write_point_cloud(file_path, pcd)

    def canonicalize_point_cloud(self, pcd, canonicalize_threshold=0.3):
        """
        Segment a 'floor' plane, if large enough, orient the cloud so that plane is XZ.
        """
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        canonicalized = False
        if len(inliers) / len(pcd.points) > canonicalize_threshold:
            canonicalized = True

            # Ensure floor normal is 'up'
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

            # optional rotate 180 deg around Z
            rotation_z_180 = np.array(
                [
                    [np.cos(np.pi), -np.sin(np.pi), 0],
                    [np.sin(np.pi),  np.cos(np.pi),  0],
                    [0,              0,             1],
                ]
            )
            pcd.rotate(rotation_z_180, center=(0, 0, 0))

            return pcd, canonicalized, transformation

        return pcd, canonicalized, None

    def extract_focal_from_intrinsic(self, intrinsic_1):
        """
        Read a single 'focal length' from the given intrinsic matrix,
        """

        if isinstance(intrinsic_1, torch.Tensor):
            intrinsic_1 = intrinsic_1.cpu()

        shape = intrinsic_1.shape

        if shape == (1, 3, 3):
            intrinsic_1 = intrinsic_1.squeeze(0)
            shape = intrinsic_1.shape

        if shape == (3, 3):
            # standard pinhole => top-left is fx
            if torch.is_tensor(intrinsic_1):
                focal = intrinsic_1[0, 0].item()
            else:
                focal = float(intrinsic_1[0, 0])
            return focal

        elif shape == (3,):
            val = intrinsic_1[0].item() if torch.is_tensor(intrinsic_1) else float(intrinsic_1[0])
            return val

        elif len(shape) == 2 and shape[0] == 1 and shape[1] == 3:
            val = (intrinsic_1[0, 0].item()
                   if torch.is_tensor(intrinsic_1) else float(intrinsic_1[0, 0]))
            return val

        else:
            # fallback for unexpected shape
            raise ValueError(f"[ERROR] Cannot interpret intrinsic of shape {shape} to extract focal length!")


    def create_point_cloud_from_model(self, pil_image):
        """
        1) Resize & normalize PIL image for VGGT,
        2) aggregator + camera + depth heads,
        3) Unproject depth => 3D point cloud,
        4) Return (pcd, depth_map_np, focal_val).
        """

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            pil_image.save(tmpfile.name)
            temp_path = tmpfile.name
        pil_image.save(temp_path)

        images = load_and_preprocess_images([temp_path])  # (B=1,C=3,H,W)
        images = images.to(device, dtype=dtype)
        images = images.unsqueeze(1)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)

        depth_map_2d = depth_map.squeeze(0)
        extrinsic_1 = extrinsic.squeeze(0)
        intrinsic_1 = intrinsic.squeeze(0)

        # Convert depth map to NumPy
        if isinstance(depth_map_2d, torch.Tensor):
            depth_map_np = depth_map_2d.cpu().numpy()
        else:
            depth_map_np = depth_map_2d
        depth_map_np = np.squeeze(depth_map_np)

        # Safely extract focal
        focal_val = self.extract_focal_from_intrinsic(intrinsic_1)

        point_map = unproject_depth_map_to_point_map(depth_map_2d, extrinsic_1, intrinsic_1)
        if isinstance(point_map, torch.Tensor):
            point_map = point_map.cpu().numpy()

        if point_map.ndim == 4 and point_map.shape[0] == 1:
            point_map = point_map[0]

        if point_map.shape[-1] != 3:
            raise ValueError(f"[ERROR] Unexpected shape for unprojected points: {point_map.shape}")

        H, W, _ = point_map.shape

        pil_image_resized = pil_image.resize((W, H), Image.BILINEAR)
        np_image = np.array(pil_image_resized, dtype=np.uint8)

        coords_3d = point_map.reshape(-1, 3)
        color_np = np_image.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_3d)
        pcd.colors = o3d.utility.Vector3dVector(color_np)
        os.remove(temp_path)

        return pcd, depth_map_np, focal_val

    def run(self, image_filename, image, masks, output_dir):
        """
        Create full scene from model, canonicalize, segment with masks,
        return pcd filepaths, canonicalized, plus depth/focal.
        """
        import math
        import numpy as np
        import os
        import cv2
        from pathlib import Path
        import open3d as o3d

        scene_pcd, depth_map_np, focal_val = self.create_point_cloud_from_model(image)

        normed_pcd, canonicalized, transformation = self.canonicalize_point_cloud(
            scene_pcd, canonicalize_threshold=0.3
        )

        points = np.asarray(normed_pcd.points)   # shape (N, 3)
        colors = np.asarray(normed_pcd.colors)   # shape (N, 3)
        total_points = points.shape[0]

        def factor_hw(n):
            root = int(math.isqrt(n))
            for i in range(root, 0, -1):
                if n % i == 0:
                    return i, n // i
            return 1, n

        H, W = factor_hw(total_points)

        output_pointcloud_dir = os.path.join(output_dir, "pointclouds")
        Path(output_pointcloud_dir).mkdir(parents=True, exist_ok=True)

        all_indices = np.arange(total_points)
        point_cloud_filepaths = []

        for idx, mask_img in enumerate(masks):
            mask_array = np.array(mask_img, dtype=np.uint8)

            if (mask_array.ndim == 2) and ((mask_array.shape[0] != H) or (mask_array.shape[1] != W)):
                mask_array = cv2.resize(mask_array, (W, H), interpolation=cv2.INTER_NEAREST)

            elif mask_array.ndim != 2:
                continue

            mask_bool = mask_array.astype(bool)
            mask_flat = mask_bool.ravel()

            valid_mask_indices = all_indices[mask_flat]
            if len(valid_mask_indices) == 0:
                print(f"[WARNING] Mask {idx+1} produced no valid points, skipping.")
                continue

            masked_points = points[valid_mask_indices]
            masked_colors = colors[valid_mask_indices]
            if masked_points.size == 0:
                print(f"[WARNING] No points left after indexing for mask {idx+1}, skipping.")
                continue

            masked_pcd = o3d.geometry.PointCloud()
            masked_pcd.points = o3d.utility.Vector3dVector(masked_points)
            masked_pcd.colors = o3d.utility.Vector3dVector(masked_colors)
            if masked_pcd.is_empty():
                print(f"[WARNING] Empty PCD for mask {idx+1}, skipping.")
                continue

            pointcloud_filepath = os.path.join(
                output_pointcloud_dir,
                f"pointcloud_{Path(image_filename).stem}_{idx}.pcd"
            )
            self.save_pointcloud(masked_pcd, pointcloud_filepath)
            point_cloud_filepaths.append(pointcloud_filepath)
        return point_cloud_filepaths, canonicalized, depth_map_np, focal_val


    def apply_transform(self, example, idx, output_dir, images):
        """
        Called by dataset.map(...) => produce pcd + store depth_map/focallength.
        """
        is_batched = (
            isinstance(example[images], list)
            and isinstance(example[images][0], (list, Image.Image))
        )

        try:
            if is_batched:
                pointclouds_all = []
                canonicalizations_all = []
                depthmaps_all = []
                focals_all = []

                for i, img_list in enumerate(example[images]):
                    if isinstance(img_list, list):
                        image = img_list[0] if isinstance(img_list[0], Image.Image) else img_list
                    else:
                        image = img_list

                    if not isinstance(image, Image.Image):
                        raise ValueError(f"[ERROR] Expected a PIL image but got {type(image)} at index {i}")
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    pcd_files, is_canonical, depth_map_np, focal_val = self.run(
                        str(idx[i]),
                        image,
                        example["masks"][i],
                        output_dir,
                    )
                    pointclouds_all.append(pcd_files)
                    canonicalizations_all.append(is_canonical)
                    depthmaps_all.append(depth_map_np)
                    focals_all.append(focal_val)

                example["pointclouds"] = pointclouds_all
                example["is_canonicalized"] = canonicalizations_all
                example["depth_map"] = depthmaps_all
                example["focallength"] = focals_all

            else:
                image = example[images]
                if isinstance(image, list) and len(image) > 0:
                    image = image[0]
                if not isinstance(image, Image.Image):
                    raise ValueError("[ERROR] The image is not a valid PIL image.")
                if image.mode != "RGB":
                    image = image.convert("RGB")

                pcd_files, is_canonical, depth_map_np, focal_val = self.run(
                    str(idx),
                    image,
                    example["masks"],
                    output_dir,
                )
                example["pointclouds"] = [pcd_files]
                example["is_canonicalized"] = [is_canonical]
                example["depth_map"] = [depth_map_np]
                example["focallength"] = [focal_val]

        except Exception as e:
            if is_batched:
                length = len(example[images])
                example["pointclouds"] = [None] * length
                example["is_canonicalized"] = [None] * length
                example["depth_map"] = [None] * length
                example["focallength"] = [None] * length
            else:
                example["pointclouds"] = [None]
                example["is_canonicalized"] = [None]
                example["depth_map"] = [None]
                example["focallength"] = [None]

        return example

