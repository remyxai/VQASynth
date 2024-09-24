import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.wrappers.zoedepth import ZoeDepth

from datasets import load_dataset
import os
import io
from dotenv import load_dotenv



def load_from_hf(image_dir: str, hf_dataset: str, hf_token: str):

    try:
        dataset = load_dataset(hf_dataset, use_auth_token=hf_token)
        os.makedirs(image_dir, exist_ok=True)

        for i, example in enumerate(dataset['train']):
            image = example['image']
            if isinstance(image, Image.Image):
                image.save(f'{image_dir}/image_{i}.png')
            else:
                Image.open(image).save(f'{image_dir}/image_{i}.png')
        print(f"Successfully loaded {len(dataset['train'])} images from '{hf_dataset}' to '{image_dir}'")
        return dataset
    
    except Exception as e:
        if 'Authentication' in str(e) and not hf_token:
            print("Error: Authentication required. Please set the HF_TOKEN environment variable.")
        else:
            print(f"Something went wrong to load dataset from HuggingFace!")
        return None


def process_images_in_chunks(image_dir, chunk_size=100):
    """Generator function to yield chunks of images from the directory."""
    chunk = []
    for image_filename in os.listdir(image_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            chunk.append(image_filename)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:  # yield the last chunk if it's not empty
        yield chunk

def main(image_dir, hf_dataset, output_dir, hf_token):

    if hf_dataset:
        load_from_hf(image_dir, hf_dataset, hf_token)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    zoe_depth = ZoeDepth()
    chunk_index = 0

    for chunk in process_images_in_chunks(image_dir):
        print("Processing chunk ", chunk_index)
        records = []

        for image_filename in chunk:
            image_path = os.path.join(image_dir, image_filename)

            img = Image.open(image_path).convert('RGB')
            depth_map = zoe_depth.infer_depth(img)

            records.append({
                "image_filename": image_filename,
                "image": img,
                "depth_map": depth_map
            })

        # Convert records to a pandas DataFrame
        df = pd.DataFrame(records)

        # Save the DataFrame to a .pkl file
        output_filepath = os.path.join(output_dir, f"chunk_{chunk_index}.pkl")
        df.to_pickle(output_filepath)

        print(f"Processed chunk {chunk_index} with {len(chunk)} images.")
        chunk_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Depth extraction", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to output dataset directory")
    parser.add_argument("--image_dir", type=str, required=False, default=None, help="path to image directory")
    parser.add_argument("--hf_dataset", type=str, required=False, default=None, help="repo id of huggingface dataset")
    parser.add_argument("--hf_token", type=str, required=False, default=None, help="token for huggingface")
    args = parser.parse_args()

    assert args.image_dir or args.hf_dataset, "Either --image_dir or --hf_dataset must be provided"

    main(args.image_dir, args.hf_dataset, args.output_dir, args.hf_token)

