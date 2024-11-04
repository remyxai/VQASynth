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

import open3d as o3d
import os
import gradio as gr
import PIL

cache_dir = './vqasynth_output'
images = 'images'


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

def run_vqasynth_pipeline(dataset: DatasetDict):

    # Apply the resizing to your dataset
    dataset = dataset.map(lambda example: resize_image_keep_aspect_ratio(example, images_column=images))


    dataset = dataset.map(lambda example: depth.apply_transform(example, images))

    del embedding_generator
    del tag_filter
    del depth

    
    dataset = dataset.map(lambda example: localizer.apply_transform(example, images))

    

    # Storing pointclouds for viewing
    point_cloud_dir = os.path.join(cache_dir, "pointclouds")
    if not os.path.exists(point_cloud_dir):
        os.makedirs(point_cloud_dir)

    dataset = dataset.map(lambda example, idx: spatial_scene_constructor.apply_transform(example, idx, cache_dir, images), with_indices=True)
    
    pcds = dataset['train']['pointclouds'][0][0]

    return pcds[1], 'ok'
    


def run(image: PIL.Image):
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_dict({"images": [image]})

    pcd_file, final_dataset = run_vqasynth_pipeline(dataset)
    output_dir = "./vqasynth_output/gradio"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ply_file = f"{output_dir}/output.ply"
    obj_file = f"{output_dir}/output.obj"
    pcd = o3d.io.read_point_cloud('output.ply')
    o3d.io.write_point_cloud(ply_file, pcd)

    mesh = o3d.io.read_triangle_mesh(ply_file)

    o3d.io.write_triangle_mesh(obj_file, mesh)
    
    return obj_file, "OK"


iface = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[gr.Model3D(label="3D Point Cloud"), gr.Text(label="Caption")],
    title="3D Point Cloud Generator without Bounding Boxes",
    description="Upload an image to generate and visualize a 3D point cloud of segmented regions without bounding boxes."
)

if __name__ == "__main__":
    depth = DepthEstimator()
    localizer = Localizer()
    spatial_scene_constructor = SpatialSceneConstructor()
    
    iface.launch(debug=True)







