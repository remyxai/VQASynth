import os
import io
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.wrappers.zoedepth import ZoeDepth

def process_images_in_chunks(image_dir, chunk_size=500):
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

def main(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    zoe_depth = ZoeDepth()
    chunk_index = 0

    for chunk in process_images_in_chunks(image_dir):
        records = []

        for image_filename in chunk:
            image_path = os.path.join(image_dir, image_filename)
            # Load the image with Pillow
            with Image.open(image_path) as img:
                # Use a BytesIO object to capture the image's binary data
                img_binary_io = io.BytesIO()
                img.save(img_binary_io, format=img.format)
                image_binary = img_binary_io.getvalue()

                # Assuming ZoeDepthWrapper can take a Pillow image
                depth_map = zoe_depth.infer_depth(img)

            records.append({
                "image_filename": image_filename,
                "image_binary": image_binary,
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
    parser.add_argument("--image_dir", type=str, required=True, help="path to image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="path to output dataset directory")
    args = parser.parse_args()
    main(args.image_dir, args.output_dir)

