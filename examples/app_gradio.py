import os
import uuid
import tempfile

import cv2
import open3d as o3d
import PIL
from PIL import Image

from vqasynth.depth import DepthEstimator
from vqasynth.localize import Localizer
from vqasynth.scene_fusion import SpatialSceneConstructor
from vqasynth.prompts import PromptGenerator

import numpy as np
import gradio as gr

depth = DepthEstimator(from_onnx=False)
localizer = Localizer()
spatial_scene_constructor = SpatialSceneConstructor()
prompt_generator = PromptGenerator()


def combine_segmented_pointclouds(
    pointcloud_ply_files: list, captions: list, prompts: list, cache_dir: str
):
    """
    Process a list of segmented point clouds to combine two based on captions and return the resulting 3D point cloud and the identified prompt.

    Args:
        pointcloud_ply_files (list): List of file paths to `.pcd` files representing segmented point clouds.
        captions (list): List of captions corresponding to the segmented point clouds.
        prompts (list): List of prompts containing questions and answers about the captions.
        cache_dir (str): Directory to save the final `.ply` and `.obj` files.

    Returns:
        tuple: The path to the generated `.obj` file and the identified prompt text.
    """
    selected_prompt = None
    selected_indices = None
    for i, caption1 in enumerate(captions):
        for j, caption2 in enumerate(captions):
            if i != j:
                for prompt in prompts:
                    if caption1 in prompt and caption2 in prompt:
                        selected_prompt = prompt
                        selected_indices = (i, j)
                        break
                if selected_prompt:
                    break
        if selected_prompt:
            break

    if not selected_prompt or not selected_indices:
        raise ValueError("No prompt found containing two captions.")

    idx1, idx2 = selected_indices
    pointcloud_files = [pointcloud_ply_files[idx1], pointcloud_ply_files[idx2]]
    captions = [captions[idx1], captions[idx2]]

    combined_point_cloud = o3d.geometry.PointCloud()
    for idx, pointcloud_file in enumerate(pointcloud_files):
        pcd = o3d.io.read_point_cloud(pointcloud_file)
        if pcd.is_empty():
            continue

        combined_point_cloud += pcd

    if combined_point_cloud.is_empty():
        raise ValueError(
            "Combined point cloud is empty after loading the selected segments."
        )

    uuid_out = str(uuid.uuid4())
    ply_file = os.path.join(cache_dir, f"combined_output_{uuid_out}.ply")
    obj_file = os.path.join(cache_dir, f"combined_output_{uuid_out}.obj")

    o3d.io.write_point_cloud(ply_file, combined_point_cloud)

    mesh = o3d.io.read_triangle_mesh(ply_file)
    o3d.io.write_triangle_mesh(obj_file, mesh)

    return obj_file, selected_prompt


def run_vqasynth_pipeline(image: PIL.Image, cache_dir: str):
    depth_map, focal_length = depth.run(image)
    masks, bounding_boxes, captions = localizer.run(image)
    pointcloud_data, cannonicalized = spatial_scene_constructor.run(
        str(0), image, depth_map, focal_length, masks, cache_dir
    )
    prompts = prompt_generator.run(captions, pointcloud_data, cannonicalized)
    obj_file, selected_prompt = combine_segmented_pointclouds(
        pointcloud_data, captions, prompts, cache_dir
    )
    return obj_file, selected_prompt


def process_image(image: str):
    # Use a persistent temporary directory to keep the .obj file accessible by Gradio
    temp_dir = tempfile.mkdtemp()
    image = Image.open(image).convert("RGB")
    obj_file, prompt = run_vqasynth_pipeline(image, temp_dir)
    return obj_file, prompt


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
        # Synthesizing SpatialVQA Samples with VQASynth
        This space helps test the full [VQASynth](https://github.com/remyxai/VQASynth) scene reconstruction pipeline on a single image with visualizations. 
        ### [Github](https://github.com/remyxai/VQASynth) | [Collection](https://huggingface.co/collections/remyxai/spacevlms-66a3dbb924756d98e7aec678) 
        """
        )

        gr.Markdown(
            """
        ## Instructions
        Upload an image, and the tool will generate a corresponding 3D point cloud visualization of the objects found and an example prompt and response describing a spatial relationship between the objects.
        """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Upload an Image")
                generate_button = gr.Button("Generate")

            with gr.Column():
                model_output = gr.Model3D(label="3D Point Cloud")  # Only used as output
                caption_output = gr.Text(label="Caption")

        generate_button.click(
            process_image, inputs=image_input, outputs=[model_output, caption_output]
        )

        gr.Examples(
            examples=[["./assets/warehouse_rgb.jpg"], ["./assets/spooky_doggy.png"]],
            inputs=image_input,
            label="Example Images",
            examples_per_page=5,
        )

        gr.Markdown(
            """
        ## Citation
        ```
        @article{chen2024spatialvlm,
          title = {SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities},
          author = {Chen, Boyuan and Xu, Zhuo and Kirmani, Sean and Ichter, Brian and Driess, Danny and Florence, Pete and Sadigh, Dorsa and Guibas, Leonidas and Xia, Fei},
          journal = {arXiv preprint arXiv:2401.12168},
          year = {2024},
          url = {https://arxiv.org/abs/2401.12168},
        }
        ```
        """
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)
