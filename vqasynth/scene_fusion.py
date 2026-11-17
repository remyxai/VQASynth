"""3D scene construction for VQASynth: images (+masks) -> registered point clouds.

Multi-view fusion (VQASynth++ Phase 2a): ``SpatialSceneConstructor`` accepts
S frames per scene instead of one. VGGT's aggregator/camera_head/depth_head
run over all S frames in a single pass and predict per-frame extrinsics,
intrinsics, depth and depth confidence; each frame is unprojected with its
own pose, which registers it into VGGT's common world frame. Fusing the
scene is then concatenation in frame order, and an object segmented in
several frames is merged by unioning its per-frame pixel sets. The
single-image path is preserved byte-for-byte (a lone PIL image is a 1-frame
scene), as is the historical return arity of every public method — extended
outputs are opt-in via keyword flags.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from PIL import Image
from torchvision import transforms as TF

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    major_cap = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if major_cap >= 8 else torch.float16
else:
    dtype = torch.float32

def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def _coerce_frame_list(pil_images):
    """
    Normalize scene input into a flat list of frames.

    Mirrors the frame-list convention of ``vqasynth.correspondence``: a lone
    PIL image is a single-frame scene (S=1); a sequence is a multi-view clip
    whose order defines the frame index used for per-frame masks, per-frame
    extrinsics and frame-major point ordering.
    """
    if isinstance(pil_images, Image.Image):
        return [pil_images]
    frames = list(pil_images)
    if len(frames) == 0:
        raise ValueError("At least one frame is required")
    return frames

def calibrate_metric_scale(depth_values):
    """
    Robust per-object metric scale anchor: the median depth over the object's
    pixels, aggregated across every frame the object appears in.

    VGGT depth lives in the scene's predicted scale, so an object's median
    camera distance is the distance anchor carried alongside its fused cloud.
    This is the minimal Phase-2a stand-in for the planned Phase-1a global
    metric calibration (known camera height / baseline): swap the body here
    when that lands — callers keep the same contract.
    """
    return float(np.median(np.asarray(depth_values, dtype=np.float64)))

def compute_object_depth_confidence(depth_conf_values):
    """
    Aggregate VGGT's per-pixel depth confidence over an object's pixels
    across all frames it appears in (mean — a relative per-object quality
    signal). Minimal Phase-2a stand-in for the Phase-1b helper of the same
    name; swap the body here when that lands.
    """
    values = np.asarray(depth_conf_values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(values.mean())

@dataclass
class SceneFusionMetadata:
    """
    Per-frame outputs of one (possibly multi-view) VGGT pass. Every array is
    frame-major: index s along the leading axis is frame s of the input list.
    """
    extrinsics: np.ndarray    # (S, 3, 4) camera-from-world, OpenCV convention
    intrinsics: np.ndarray    # (S, 3, 3) pixel intrinsics
    depth: np.ndarray         # (S, H, W) per-frame depth
    depth_conf: np.ndarray    # (S, H, W) per-frame depth confidence
    focals: list = field(default_factory=list)  # per-frame focal length (pixels)

    @property
    def num_frames(self):
        return int(self.depth.shape[0])

    @property
    def frame_shape(self):
        return tuple(int(d) for d in self.depth.shape[1:3])

def _coerce_frame_masks(masks, num_frames, mode="crop"):
    """
    Normalize mask input into per-frame boolean mask stacks aligned with the
    preprocessed frames.

    Accepts either:
      - per-frame masks: a sequence whose first entry is itself a sequence of
        (H, W) masks — entry s holds object masks for frame s. Objects absent
        from a frame are padded with empty masks.
      - single-frame masks: a flat sequence of (H, W) masks, applied to the
        primary (first) frame only (the historical contract).

    Returns (frame_masks, num_objects) where frame_masks[s][idx] is a boolean
    (H, W) array over frame s's preprocessed pixels for object idx.

    Masks are resampled with NEAREST interpolation (binary values must stay
    binary) in one batched call. Masks are zero-padded up-front (centered,
    matching ``preprocess_images``'s positions) to the common extents so the
    image path's cross-shape white-padding never applies to a mask — a
    padded mask region must read False, not the 1.0 images pad with. (The
    pad-mode square-to-518 padding keeps its historical image-path value,
    as in the previous single-frame implementation.)
    """
    def _is_mask_stack(entry):
        if isinstance(entry, (list, tuple)):
            return len(entry) > 0 and np.asarray(entry[0]).ndim == 2
        if isinstance(entry, np.ndarray):
            return entry.ndim == 3
        return False

    looks_per_frame = (
        isinstance(masks, (list, tuple))
        and len(masks) > 0
        and _is_mask_stack(masks[0])
    )
    if looks_per_frame:
        per_frame_raw = list(masks)[:num_frames]
        while len(per_frame_raw) < num_frames:
            per_frame_raw.append([])
    else:
        per_frame_raw = [masks] + [[]] * (num_frames - 1)

    num_objects = max((len(m) for m in per_frame_raw), default=0)
    if num_objects == 0:
        return [], 0

    # Anchor the empty-mask shape on the first present mask so absent
    # objects/frames pad correctly even when leading frames are empty.
    anchor_shape = None
    for raw in per_frame_raw:
        for m in raw:
            if m is not None:
                anchor_shape = np.asarray(m).shape
                break
        if anchor_shape is not None:
            break

    items = []
    for raw in per_frame_raw:
        for idx in range(num_objects):
            if idx < len(raw) and raw[idx] is not None:
                arr = np.asarray(raw[idx])
            else:
                arr = np.zeros(anchor_shape)  # object absent from this frame
            items.append(arr)

    # Zero-pad every mask to the common extents BEFORE preprocessing so the
    # image path's centered white-padding never touches a mask value.
    max_h = max(arr.shape[0] for arr in items)
    max_w = max(arr.shape[1] for arr in items)
    for i, arr in enumerate(items):
        pad_top = (max_h - arr.shape[0]) // 2
        pad_bottom = max_h - arr.shape[0] - pad_top
        pad_left = (max_w - arr.shape[1]) // 2
        pad_right = max_w - arr.shape[1] - pad_left
        items[i] = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)))

    # Single-channel "images": one channel dim keeps Resize/crop slicing on
    # the right axes inside preprocess_images.
    stacked = torch.tensor(np.stack(items)).unsqueeze(1) / 255.0  # (S*N,1,H,W)
    processed = preprocess_images(stacked, mode=mode, interpolation=Image.Resampling.NEAREST)
    processed = processed.reshape(num_frames, num_objects, *processed.shape[-2:])
    frame_masks = [frame >= 0.5 for frame in processed]
    return frame_masks, num_objects

def preprocess_images(image_list, mode="crop", interpolation=Image.Resampling.BICUBIC):
    """
    Modified from https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/vggt/utils/load_fn.py#L97-L230

    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_list (list): List of images (PIL.Image or torch.Tensor)
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for img in image_list:
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            # If there's an alpha channel, blend onto white background:
            if img.mode == "RGBA":
                # Create white background
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                # Alpha composite onto the white background
                img = Image.alpha_composite(background, img)

            # Now convert to "RGB" (this step assigns white for transparent areas)
            img = img.convert("RGB")

            width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (new_width / width) / 14) * 14
                )  # Make divisible by 14
            else:
                new_height = target_size
                new_width = (
                    round(width * (new_height / height) / 14) * 14
                )  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = TF.Resize((new_height, new_width), interpolation=interpolation)(img)
        if not isinstance(img, torch.Tensor):
            img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def restore_pointclouds(pointcloud_paths):
    if len(pointcloud_paths) == 1 and isinstance(pointcloud_paths[0], list):
        pointcloud_paths = pointcloud_paths[0]

    restored_pointclouds = []
    for path in pointcloud_paths:
        restored_pointclouds.append(o3d.io.read_point_cloud(path))

    return restored_pointclouds

class SpatialSceneConstructor:
    def __init__(self, compile_aggregator: bool = False, avggt_step1: bool = False):
        """
        Initialize the constructor and load the VGGT model.

        Args:
            compile_aggregator: if True, wrap ``model.aggregator`` with
                ``torch.compile(mode="reduce-overhead")``. Off by default —
                measured wins on T4 are modest (~1.09× at S=1) and VGGT's
                rope layer triggers a dynamo graph break on the
                ``int(positions.max())`` scalar-tensor conversion, producing
                warmup cost + warning spam. Ampere+ users on longer-running
                workloads may want to opt in; consider also setting
                ``torch._dynamo.config.capture_scalar_outputs = True`` to
                trace through the rope graph break.
            avggt_step1: if True (or env var ``VQASYNTH_AVGGT_STEP1=1``),
                apply AVGGT Step 1 (early-global-as-frame attention). See
                :mod:`vqasynth.vggt_speedups` for details. Off by default;
                measured neutral at S≤4 (VQASynth's current call pattern);
                begins paying off at S≥15+.
        """
        # Cast weights to the target dtype instead of loading fp32 and relying
        # on autocast alone — cuts weight memory in half and lets the kernel
        # scheduler pick fp16/bf16 paths from the first call. VGGT uses
        # PyTorchModelHubMixin which doesn't intercept torch_dtype the way
        # transformers.PretrainedModel does, so cast happens in the .to() call.
        # Integer buffers are preserved (nn.Module.to only casts floating tensors).
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(device=device, dtype=dtype)
        self.model.eval()

        if avggt_step1 or os.environ.get("VQASYNTH_AVGGT_STEP1") == "1":
            from vqasynth.vggt_speedups import apply_avggt_step1
            apply_avggt_step1(self.model, early_g2f=9)

        if compile_aggregator and os.environ.get("VQASYNTH_DISABLE_COMPILE") != "1":
            try:
                self.model.aggregator = torch.compile(
                    self.model.aggregator, mode="reduce-overhead"
                )
            except Exception as e:  # older torch, unsupported backend, etc.
                print(f"[SpatialSceneConstructor] torch.compile disabled ({type(e).__name__}: {e})")

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


    def create_point_cloud_from_model(self, pil_images, mode="crop", return_metadata=False):
        """
        1) Resize & normalize PIL frame(s) for VGGT,
        2) aggregator + camera + depth heads over all S frames,
        3) Unproject each frame's depth with its own predicted extrinsic /
           intrinsic => per-frame points in VGGT's common world frame, fused
           (concatenated in frame order) into ONE registered scene cloud,
        4) Return (pcd, depth_map_np, focal_val) — plus a SceneFusionMetadata
           with the per-frame extrinsics/intrinsics/depth_conf when
           ``return_metadata=True``.

        ``pil_images`` may be a lone PIL image (S=1 — the historical
        single-view path: same point cloud, same (H, W) depth map, same
        scalar focal) or a list of S frames. With S>1 and ``return_metadata``
        unset, ``depth_map_np`` is the stacked (S, H, W) per-frame depth and
        ``focal_val`` the primary frame's focal.
        """
        frames = _coerce_frame_list(pil_images)

        preprocessed_images = preprocess_images(frames, mode=mode)  # (S,C,H,W)
        images = preprocessed_images.to(device, dtype=dtype)
        images = images.unsqueeze(0)  # (B=1,S,C,H,W)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)

        depth_frames = _to_numpy(depth_map.squeeze(0))        # (S,H,W)
        extrinsic_frames = _to_numpy(extrinsic.squeeze(0))    # (S,3,4)
        intrinsic_frames = _to_numpy(intrinsic.squeeze(0))    # (S,3,3)
        depth_conf_frames = _to_numpy(depth_conf.squeeze(0))  # (S,H,W)

        # Safely extract focal (the primary frame keeps the historical
        # single-view return; per-frame focals ride on the metadata).
        focal_val = self.extract_focal_from_intrinsic(intrinsic_frames[:1])

        # Per-frame unprojection with each frame's own predicted pose
        # registers every frame into VGGT's common world frame — fusion is
        # concatenation in frame order.
        point_map = unproject_depth_map_to_point_map(
            depth_frames, extrinsic_frames, intrinsic_frames
        )
        if isinstance(point_map, torch.Tensor):
            point_map = point_map.cpu().numpy()
        point_map = np.asarray(point_map)

        if point_map.ndim != 4 or point_map.shape[-1] != 3:
            raise ValueError(f"[ERROR] Unexpected shape for unprojected points: {point_map.shape}")

        coords_3d = point_map.reshape(-1, 3)
        color_np = preprocessed_images.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_3d)
        pcd.colors = o3d.utility.Vector3dVector(color_np)

        metadata = SceneFusionMetadata(
            extrinsics=extrinsic_frames,
            intrinsics=intrinsic_frames,
            depth=depth_frames,
            depth_conf=depth_conf_frames,
            focals=[
                self.extract_focal_from_intrinsic(intrinsic_frames[s : s + 1])
                for s in range(len(frames))
            ],
        )

        if return_metadata:
            return pcd, depth_frames, focal_val, metadata
        return pcd, np.squeeze(depth_frames), focal_val

    def run(self, image_filename, image, masks, output_dir, mode="crop", return_scene_stats=False):
        """
        Create full scene from model, canonicalize, segment with masks,
        return pcd filepaths, canonicalized, plus depth/focal.

        ``image`` may be a lone PIL image or a list of S frames (fused into
        one registered cloud). ``masks`` may hold per-frame masks (a sequence
        of S per-frame mask sequences — an object appearing in several frames
        is merged into ONE cloud spanning them) or the historical flat
        single-frame mask list, applied to the primary (first) frame. Every
        saved object cloud carries a metric scale
        (``calibrate_metric_scale``) and an aggregated depth confidence
        (``compute_object_depth_confidence``) over its pixels in all frames
        it appears in.

        With ``return_scene_stats=True`` a fifth element is returned — a dict
        with ``num_frames``, per-frame ``extrinsics``/``intrinsics``/
        ``focals`` and per-``objects`` stats aligned with the saved
        filepaths — so dataset transforms can surface camera poses and
        object stats as columns.
        """
        frames = _coerce_frame_list(image)

        scene_pcd, depth_map_np, focal_val, metadata = self.create_point_cloud_from_model(
            frames, mode=mode, return_metadata=True
        )

        num_frames = metadata.num_frames
        frame_h, frame_w = metadata.frame_shape
        points_per_frame = frame_h * frame_w

        normed_pcd, canonicalized, transformation = self.canonicalize_point_cloud(
            scene_pcd, canonicalize_threshold=0.3
        )

        points = np.asarray(normed_pcd.points)   # shape (S*H*W, 3), frame-major
        colors = np.asarray(normed_pcd.colors)   # shape (S*H*W, 3)

        output_pointcloud_dir = os.path.join(output_dir, "pointclouds")
        Path(output_pointcloud_dir).mkdir(parents=True, exist_ok=True)

        point_cloud_filepaths = []

        frame_masks, num_objects = _coerce_frame_masks(masks, num_frames, mode=mode)

        depth_flat = metadata.depth.reshape(-1)
        conf_flat = metadata.depth_conf.reshape(-1)

        object_stats = []
        for idx in range(num_objects):
            # Union this object's pixel sets across every frame it appears
            # in — the cross-frame merge. All frames already share VGGT's
            # world frame, so the union is a single registered cloud.
            per_frame_indices = []
            frames_seen = []
            for s, frame_mask in enumerate(frame_masks):
                mask_flat = frame_mask[idx].ravel()
                valid_mask_indices = np.flatnonzero(mask_flat) + s * points_per_frame
                if len(valid_mask_indices) == 0:
                    continue
                per_frame_indices.append(valid_mask_indices)
                frames_seen.append(s)

            if not per_frame_indices:
                print(f"[WARNING] Mask {idx+1} produced no valid points, skipping.")
                continue

            valid_mask_indices = np.concatenate(per_frame_indices)
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
            object_stats.append(
                {
                    "metric_scale": calibrate_metric_scale(depth_flat[valid_mask_indices]),
                    "depth_confidence": compute_object_depth_confidence(
                        conf_flat[valid_mask_indices]
                    ),
                    "frames_seen": frames_seen,
                    "point_count": int(valid_mask_indices.size),
                }
            )

        depth_out = np.squeeze(depth_map_np) if num_frames == 1 else depth_map_np

        scene_stats = {
            "num_frames": num_frames,
            "extrinsics": metadata.extrinsics,
            "intrinsics": metadata.intrinsics,
            "focals": metadata.focals,
            "objects": object_stats,
        }

        result = (point_cloud_filepaths, canonicalized, depth_out, focal_val)
        return result + (scene_stats,) if return_scene_stats else result


    def apply_transform(self, example, idx, output_dir, images):
        """
        Called by dataset.map(...) => produce pcd + store depth_map/focallength.

        An image entry may be a single PIL image or a list of PIL frames —
        one (possibly multi-view) scene per entry, fused into one registered
        cloud. Per-frame camera poses are surfaced on the
        ``camera_extrinsics`` / ``camera_intrinsics`` columns (frame-major
        lists, OpenCV camera-from-world convention) so downstream
        perspective-taking QA can reason about frame of reference; per-object
        metric scale + aggregated depth confidence ride on
        ``object_metric_scale`` / ``object_depth_confidence`` aligned with
        the saved point clouds.

        Batched vs single is decided by ``idx`` (``datasets.map`` passes a
        sequence of indices when ``with_indices=True, batched=True``) plus a
        list-of-lists image column; a scalar ``idx`` with a flat list of PIL
        frames is one unbatched multi-view scene.
        """
        is_batched = isinstance(idx, (list, tuple)) or (
            isinstance(example[images], list)
            and len(example[images]) > 0
            and isinstance(example[images][0], list)
        )

        try:
            if is_batched:
                pointclouds_all = []
                canonicalizations_all = []
                depthmaps_all = []
                focals_all = []
                extrinsics_all = []
                intrinsics_all = []
                object_scales_all = []
                object_confs_all = []

                for i, row in enumerate(example[images]):
                    rgb_frames = self._coerce_rgb_frames(row, i)

                    (pcd_files, is_canonical, depth_map_np, focal_val, scene_stats) = self.run(
                        str(idx[i]),
                        rgb_frames,
                        example["masks"][i],
                        output_dir,
                        return_scene_stats=True,
                    )
                    num_frames = scene_stats["num_frames"]
                    pointclouds_all.append(pcd_files)
                    canonicalizations_all.append(is_canonical)
                    depthmaps_all.append(depth_map_np if num_frames == 1 else list(depth_map_np))
                    focals_all.append(focal_val if num_frames == 1 else scene_stats["focals"])
                    extrinsics_all.append(scene_stats["extrinsics"].tolist())
                    intrinsics_all.append(scene_stats["intrinsics"].tolist())
                    object_scales_all.append([o["metric_scale"] for o in scene_stats["objects"]])
                    object_confs_all.append([o["depth_confidence"] for o in scene_stats["objects"]])

                example["pointclouds"] = pointclouds_all
                example["is_canonicalized"] = canonicalizations_all
                example["depth_map"] = depthmaps_all
                example["focallength"] = focals_all
                example["camera_extrinsics"] = extrinsics_all
                example["camera_intrinsics"] = intrinsics_all
                example["object_metric_scale"] = object_scales_all
                example["object_depth_confidence"] = object_confs_all

            else:
                rgb_frames = self._coerce_rgb_frames(example[images])

                (pcd_files, is_canonical, depth_map_np, focal_val, scene_stats) = self.run(
                    str(idx),
                    rgb_frames,
                    example["masks"],
                    output_dir,
                    return_scene_stats=True,
                )
                num_frames = scene_stats["num_frames"]
                example["pointclouds"] = [pcd_files]
                example["is_canonicalized"] = [is_canonical]
                example["depth_map"] = [depth_map_np if num_frames == 1 else list(depth_map_np)]
                example["focallength"] = [focal_val if num_frames == 1 else scene_stats["focals"]]
                example["camera_extrinsics"] = [scene_stats["extrinsics"].tolist()]
                example["camera_intrinsics"] = [scene_stats["intrinsics"].tolist()]
                example["object_metric_scale"] = [
                    [o["metric_scale"] for o in scene_stats["objects"]]
                ]
                example["object_depth_confidence"] = [
                    [o["depth_confidence"] for o in scene_stats["objects"]]
                ]

        except Exception as e:
            if is_batched:
                length = len(example[images])
                example["pointclouds"] = [None] * length
                example["is_canonicalized"] = [None] * length
                example["depth_map"] = [None] * length
                example["focallength"] = [None] * length
                example["camera_extrinsics"] = [None] * length
                example["camera_intrinsics"] = [None] * length
                example["object_metric_scale"] = [None] * length
                example["object_depth_confidence"] = [None] * length
            else:
                example["pointclouds"] = [None]
                example["is_canonicalized"] = [None]
                example["depth_map"] = [None]
                example["focallength"] = [None]
                example["camera_extrinsics"] = [None]
                example["camera_intrinsics"] = [None]
                example["object_metric_scale"] = [None]
                example["object_depth_confidence"] = [None]

        return example

    @staticmethod
    def _coerce_rgb_frames(row, i=None):
        """
        Normalize one image entry (a PIL image or a list of PIL frames) into
        a validated list of RGB PIL frames.
        """
        where = f" at index {i}" if i is not None else ""
        frame_list = list(row) if isinstance(row, (list, tuple)) else [row]
        if len(frame_list) == 0:
            raise ValueError(f"[ERROR] Expected at least one PIL image{where}")
        rgb_frames = []
        for j, image in enumerate(frame_list):
            if not isinstance(image, Image.Image):
                raise ValueError(
                    f"[ERROR] Expected a PIL image but got {type(image)}{where}[{j}]"
                )
            if image.mode != "RGB":
                image = image.convert("RGB")
            rgb_frames.append(image)
        return rgb_frames

