import pickle
import numpy as np
import open3d as o3d

def serialize_pointclouds(pointclouds):
    serialized_data = []
    for pcd in pointclouds:
        # Extract relevant data from each PointCloud
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None

        # Serialize each component to bytes
        pcd_data = {
            'points': pickle.dumps(points),
            'colors': pickle.dumps(colors) if colors is not None else None,
            'normals': pickle.dumps(normals) if normals is not None else None,
        }

        # Add the serialized data for the current PointCloud to the list
        serialized_data.append(pcd_data)
    return serialized_data

def restore_pointclouds(serialized_data):
    restored_pointclouds = []
    for data in serialized_data:
        # Deserialize each component
        points = pickle.loads(data['points'])
        colors = pickle.loads(data['colors']) if data['colors'] is not None else None
        normals = pickle.loads(data['normals']) if data['normals'] is not None else None

        # Create a new PointCloud object and assign the deserialized data
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        # Add the restored PointCloud to the list
        restored_pointclouds.append(pcd)
    return restored_pointclouds


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
