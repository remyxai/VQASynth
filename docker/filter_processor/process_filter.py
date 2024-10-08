import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.datasets.utils import ImageTagger

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

def filter_by_tags(tags, include_tags, exclude_tags):
    """
    Determines if an image should be included based on the tags, include_tags, and exclude_tags.

    Args:
        tags (list): List of tags for the image.
        include_tags (list): Tags to include (if present in image, include it).
        exclude_tags (list): Tags to exclude (if present in image, discard it).

    Returns:
        bool: True if image should be included, False otherwise.
    """
    if include_tags and not any(tag in include_tags for tag in tags):
        return False
    if exclude_tags and any(tag in exclude_tags for tag in tags):
        return False
    return True

def main(image_dir, output_dir, include_tags=None, exclude_tags=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert empty or None strings to empty lists
    if isinstance(include_tags, str) and include_tags.strip():
        include_tags = include_tags.split(',')
    else:
        include_tags = []

    if isinstance(exclude_tags, str) and exclude_tags.strip():
        exclude_tags = exclude_tags.split(',')
    else:
        exclude_tags = []

    tagger = ImageTagger()
    chunk_index = 0

    for chunk in process_images_in_chunks(image_dir):
        print("Processing chunk ", chunk_index)
        records = []

        for image_filename in chunk:
            image_path = os.path.join(image_dir, image_filename)

            img = Image.open(image_path).convert('RGB')
            tags = tagger.get_top_tags(img)

            if should_include_image(tags, include_tags, exclude_tags):
                records.append({
                    "full_path": image_path,
                    "image_filename": image_filename,
                    "image": img,
                    "tags": tags
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
    parser.add_argument("--include_tags", type=str, required=False, default=None, help="Comma-separated list of tags to include (optional)")
    parser.add_argument("--exclude_tags", type=str, required=False, default=None, help="Comma-separated list of tags to exclude (optional)")
    args = parser.parse_args()
    main(args.image_dir, args.output_dir, args.include_tags, args.exclude_tags)
