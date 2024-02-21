import os
import cv2
import gc
import sys
import math
import torch
import json
import base64
import argparse
import numpy as np
import open3d as o3d

import matplotlib
import matplotlib.cm
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("./ZoeDepth")
sys.path.append("./efficientvit")

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from transformers import SamModel, SamProcessor

# llava v1.6
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

from transformers import AutoProcessor, CLIPSegForImageSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

# ZoeD_N
conf = get_config("zoedepth", "infer")
depth_model = build_model(conf)


def depth(img):
    depth = depth_model.infer_pil(img)
    raw_depth = Image.fromarray((depth*256).astype('uint16'))
    return raw_depth

def create_point_cloud_from_rgbd(rgb_image, depth_image, intrinsic_parameters):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.io.read_image(rgb_image),
        o3d.io.read_image(depth_image),
        depth_scale=1000.0,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(intrinsic_parameters['width'], intrinsic_parameters['height'],
                             intrinsic_parameters['fx'], intrinsic_parameters['fy'],
                             intrinsic_parameters['cx'], intrinsic_parameters['cy'])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd

def canonicalize_point_cloud(pcd, canonicalize_threshold=0.3):
    # Segment the largest plane, assumed to be the floor
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

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
        transformation[:3, 3] = -np.dot(transformation[:3, :3], pcd.points[inliers[0]])

        # Apply the transformation
        pcd.transform(transformation)

        # Additional 180-degree rotation around the Z-axis
        rotation_z_180 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                                   [np.sin(np.pi), np.cos(np.pi), 0],
                                   [0, 0, 1]])
        pcd.rotate(rotation_z_180, center=(0, 0, 0))

        return pcd, canonicalized, transformation
    else:
        return pcd, canonicalized, None

def extract_descriptions_from_incomplete_json(json_like_str):
    last_object_idx = json_like_str.rfind(',"object')

    if last_object_idx != -1:
        json_str = json_like_str[:last_object_idx] + '}'
    else:
        json_str = json_like_str.strip()
        if not json_str.endswith('}'):
            json_str += '}'

    try:
        json_obj = json.loads(json_str)
        descriptions = [details['description'].replace(".","") for key, details in json_obj.items() if 'description' in details]

        return descriptions
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

def find_medoid_and_closest_points(points, num_closest=5):
    """
    Find the medoid from a collection of points and the closest points to the medoid.

    Parameters:
    points (np.array): A numpy array of shape (N, D) where N is the number of points and D is the dimensionality.
    num_closest (int): Number of closest points to return.

    Returns:
    np.array: The medoid point.
    np.array: The closest points to the medoid.
    """
    distances = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis=-1))
    distance_sums = distances.sum(axis=1)
    medoid_idx = np.argmin(distance_sums)
    medoid = points[medoid_idx]
    sorted_indices = np.argsort(distances[medoid_idx])
    closest_indices = sorted_indices[1:num_closest + 1]
    return medoid, points[closest_indices]

def sample_points_from_heatmap(heatmap, original_size, num_points=5, percentile=0.95):
    """
    Sample points from the given heatmap, focusing on areas with higher values.
    """
    threshold = np.percentile(heatmap.numpy(), percentile)
    masked_heatmap = torch.where(heatmap > threshold, heatmap, torch.tensor(0.0))
    probabilities = torch.softmax(masked_heatmap.flatten(), dim=0)

    attn = torch.sigmoid(heatmap)
    w = attn.shape[0]
    sampled_indices = torch.multinomial(torch.tensor(probabilities.ravel()), num_points, replacement=True)

    sampled_coords = np.array(np.unravel_index(sampled_indices, attn.shape)).T
    medoid, sampled_coords = find_medoid_and_closest_points(sampled_coords)
    pts = []
    for pt in sampled_coords.tolist():
        x, y = pt
        x = height * x / w
        y = width * y / w
        pts.append([y, x])
    return pts

def apply_mask_to_image(image, mask):
    """
    Apply a binary mask to an image. The mask should be a binary array where the regions to keep are True.
    """
    masked_image = image.copy()
    for c in range(masked_image.shape[2]):
        masked_image[:, :, c] = masked_image[:, :, c] * mask
    return masked_image

def calculate_distances_between_point_clouds(point_clouds):
    distances_info = []

    for i, pcd1 in enumerate(point_clouds):
        for j in range(i + 1, len(point_clouds)):
            pcd2 = point_clouds[j]
            dist_pcd1_to_pcd2 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
            dist_pcd2_to_pcd1 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
            combined_distances = np.concatenate((dist_pcd1_to_pcd2, dist_pcd2_to_pcd1))
            min_dist = np.min(combined_distances)
            avg_dist = np.mean(combined_distances)
            max_dist = np.max(combined_distances)

            distances_info.append({
                'pcd_pair': (i, j),
                'min_distance': min_dist,
                'average_distance': avg_dist,
                'max_distance': max_dist
            })

    return distances_info

def calculate_centroid(pcd):
    """Calculate the centroid of a point cloud."""
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    return centroid

def calculate_relative_positions(centroids):
    """Calculate the relative positions between centroids of point clouds."""
    num_centroids = len(centroids)
    relative_positions_info = []

    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            relative_vector = centroids[j] - centroids[i]

            distance = np.linalg.norm(relative_vector)
            relative_positions_info.append({
                'pcd_pair': (i, j),
                'relative_vector': relative_vector,
                'distance': distance
            })

    return relative_positions_info

def get_bounding_box_height(pcd):
    """
    Compute the height of the bounding box for a given point cloud.

    Parameters:
    pcd (open3d.geometry.PointCloud): The input point cloud.

    Returns:
    float: The height of the bounding box.
    """
    aabb = pcd.get_axis_aligned_bounding_box()
    return aabb.get_extent()[1]  # Assuming the Y-axis is the up-direction

def compare_bounding_box_height(pcd_i, pcd_j):
    """
    Compare the bounding box heights of two point clouds.

    Parameters:
    pcd_i (open3d.geometry.PointCloud): The first point cloud.
    pcd_j (open3d.geometry.PointCloud): The second point cloud.

    Returns:
    bool: True if the bounding box of pcd_i is taller than that of pcd_j, False otherwise.
    """
    height_i = get_bounding_box_height(pcd_i)
    height_j = get_bounding_box_height(pcd_j)

    return height_i > height_j

if __name__ == "__main__":

    parser = argparse.ArgumentParser("ClipSeg-based data processing", add_help=True)
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    args = parser.parse_args()

    image_path = args.input_image

    # Step 1: Depth estimation
    print("Depth estimation")
    original_image = Image.open(image_path)
    depth_image = depth(original_image)
    width, height = original_image.size
    intrinsic_parameters = {
        'width': width,
        'height': height,
        'fx': 1.5 * width,
        'fy': 1.5 * width,
        'cx': width / 2,
        'cy': height / 2,
    }
    depth_path = "tmp_depth_out.png"
    depth_image.save(depth_path)

    pcd = create_point_cloud_from_rgbd(image_path, depth_path, intrinsic_parameters)
    pcd_canonicalized, canonicalized, transformation = canonicalize_point_cloud(pcd)
    cloud = np.asarray(pcd_canonicalized.points)

    # Step 2: Llava Captions
    print("llava captions")
    data_uri = image_to_base64_data_uri(image_path)

    chat_handler = Llava15ChatHandler(clip_model_path="mmproj-model-f16.gguf", verbose=True)
    llm = Llama(model_path="llava-v1.6-34b.Q4_K_M.gguf",chat_handler=chat_handler,n_ctx=2048,logits_all=True, n_gpu_layers=-1)
    res = llm.create_chat_completion(
         messages = [
             {"role": "system", "content": "You are an assistant who perfectly describes images."},
             {
                 "role": "user",
                 "content": [
                     {"type": "image_url", "image_url": {"url": data_uri}},
                     {"type" : "text", "text": 'Create a JSON representation where each entry consists of a key "object" with a numerical suffix starting from 1, and a corresponding "description" key with a value that is a concise, up to six-word sentence describing each main, distinct object or person in the image. Each pair should uniquely describe one element without repeating keys. An example: {"object1": { "description": "Man in red hat walking." },"object2": { "description": "Wooden pallet with boxes." },"object3": { "description": "Cardboard boxes stacked." },"object4": { "description": "Man in green vest standing." }}'}
                ]
             }
         ]
    )

    text_descriptions = list(set(extract_descriptions_from_incomplete_json(res["choices"][0]["message"]["content"])))

    del chat_handler
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3 : Clipseg
    print("clipseg")
    clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = clipseg_processor(text=text_descriptions, images=[original_image] * len(text_descriptions), padding=True, return_tensors="pt")
    outputs = clipseg_model(**inputs)
    logits = outputs.logits
    preds = logits.detach().unsqueeze(1)

    sampled_points = []
    original_image_cv = cv2.imread(image_path)
    original_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
    original_size = original_image_cv.shape[:2][::-1]
    for idx in range(preds.shape[0]):
        sampled_points.append(sample_points_from_heatmap(preds[idx][0], original_size, num_points=10))


    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to("cuda" if torch.cuda.is_available() else "cpu")
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    sam_masks = []
    for idx in range(preds.shape[0]):
        sam_inputs = sam_processor(original_image, input_points=[sampled_points[idx]], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)

        sam_masks.append(sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
            ))

    # Step 4: Get segmented pointcloud
    print("pointcloud seg")
    depth_image_cv = cv2.imread(depth_path)
    point_clouds = []
    for i, mask_tensor in enumerate(sam_masks):
        mask = cv2.cvtColor(255 * mask_tensor[0].numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
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

    # Step 5: Calculate cloud distances
    print("Distance calcs")
    distance_info = calculate_distances_between_point_clouds(point_clouds)

    # Now 'distance_info' contains the distance info for each pair of point clouds
    for info in distance_info:
        print(f"Between '{text_descriptions[info['pcd_pair'][0]]}' and '{text_descriptions[info['pcd_pair'][1]]}':")
        print(f"  Min distance: {info['min_distance']}")
        print(f"  Average distance: {info['average_distance']}")
        print(f"  Max distance: {info['max_distance']}")

    centroids = [calculate_centroid(pcd) for pcd in point_clouds]
    relative_positions_info = calculate_relative_positions(centroids)

    # Now 'relative_positions_info' contains the relative position vectors and distances
    for info in relative_positions_info:
        print(f"Between '{text_descriptions[info['pcd_pair'][0]]}' and '{text_descriptions[info['pcd_pair'][1]]}':")
        print(f"  Relative vector: {info['relative_vector']}")
        print(f"  Distance: {info['distance']}")

    # Example usage:
    # Assuming 'point_clouds' is your list of Open3D point cloud objects
    # And you want to compare the box of point cloud at index i with the box of point cloud at index j
    i = 0
    j = 2
    is_taller = compare_bounding_box_height(point_clouds[i], point_clouds[j])

    print(f"Bounding box of '{text_descriptions[i]}' is {'taller' if is_taller else 'not taller'} than bounding box of '{text_descriptions[j]}'.")
