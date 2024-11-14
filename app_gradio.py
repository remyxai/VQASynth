import PIL.Image
from vqasynth.datasets import Dataloader
from vqasynth.embeddings import EmbeddingGenerator
from PIL import Image
import os
from vqasynth.embeddings import TagFilter
from vqasynth.depth import DepthEstimator
from vqasynth.localize import Localizer
from vqasynth.scene_fusion import SpatialSceneConstructor
from vqasynth.prompts import PromptGenerator
from datasets import DatasetDict, Dataset
import tempfile

import open3d as o3d
import os
import gradio as gr
import PIL
import numpy as np



def resize_image_keep_aspect_ratio(example, images_column, max_width=500):
        # Get the original image
        image = example[images_column]

        # Get the original size
        width, height = image.size

        # Calculate the new height to maintain aspect ratio
        if width > max_width:
            new_width = max_width
            new_height = int((new_width / width) * height)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)  # Replaced ANTIALIAS with LANCZOS
        else:
            # If width is already less than or equal to max_width, don't resize
            resized_image = image

        # Replace the image in the example
        example[images_column] = resized_image
        return example

        

def run_vqasynth_pipeline(dataset: DatasetDict, cache_dir: str):
    image_column = "images"
    # dataset = dataset.map(lambda example: resize_image_keep_aspect_ratio(example, images_column=image_column))
    dataset = dataset.map(lambda example: depth.apply_transform(example, image_column))
    dataset = dataset.map(lambda example: localizer.apply_transform(example, image_column))
    point_cloud_dir = os.path.join(cache_dir, "pointclouds")
    if not os.path.exists(point_cloud_dir):
        os.makedirs(point_cloud_dir)
    dataset = dataset.map(lambda example, idx: spatial_scene_constructor.apply_transform(example, idx, cache_dir, image_column), with_indices=True)
    print("applying prompt")
    dataset = dataset.map(lambda example: prompt_generator.apply_transform(example))
    pcds = [file for sublist in dataset['train']['pointclouds'][0] for file in sublist]
    print(pcds)
    print(dataset)
    caption = dataset['train']['prompts'][0][0]
    depth_map = dataset['train']['depth_map'][0]
    focal_length = dataset['train']['focallength'][0]
    return pcds, caption, depth_map, focal_length



def merge_masked_pcds(pcd_files, output_dir):
    # Initialize an empty point cloud to combine all points
    combined_pcd = o3d.geometry.PointCloud()

    # Load each point cloud and combine them
    for ply_file in pcd_files:
        pcd = o3d.io.read_point_cloud(ply_file)
        combined_pcd += pcd 
    
    ply_output_path = os.path.join(output_dir, 'pointclouds', 'masked_combined_pcd.ply')
    o3d.io.write_point_cloud(ply_output_path, combined_pcd)

    
    mesh = o3d.io.read_triangle_mesh(ply_output_path)
    obj_output_path = os.path.join(output_dir, 'pointclouds', 'masked_combined_mesh.obj')
    o3d.io.write_triangle_mesh(obj_output_path, mesh)

    return obj_output_path



def run(image: PIL.Image):
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_dict({"images": [image]})
    # Use a persistent temporary directory to keep the .obj file accessible by Gradio
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory created at {temp_dir}")
    # Run pipeline to generate point cloud paths and caption
    pcd_files, caption, depth_map, focal_length = run_vqasynth_pipeline(dataset, temp_dir)
    # Path for the output .obj file
    obj_file = merge_masked_pcds(pcd_files[:2], temp_dir)
    return obj_file, caption





iface = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[gr.Model3D(label="Segmented Point Cloud"), gr.Text(label="Caption")],
    title="Synthesizing SpatialVQA Samples with VQASynth",
    description="Upload an image to run the VQASynth which will constuct a 3d representation of the image and generates the caption with spatial information"
)

if __name__ == "__main__":
    depth = DepthEstimator(from_onnx=False)
    localizer = Localizer()
    prompt_generator = PromptGenerator()
    spatial_scene_constructor = SpatialSceneConstructor()

    iface.launch(debug=True)







