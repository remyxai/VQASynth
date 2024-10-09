import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.embeddings import EmbeddingGenerator
from vqasynth.utils import process_images_in_chunks

def main(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    embedding_generator = EmbeddingGenerator()
    chunk_index = 0

    for chunk in process_images_in_chunks(image_dir):
        print("Processing chunk ", chunk_index)
        records = []

        for image_filename in chunk:
            image_path = os.path.join(image_dir, image_filename)

            img = Image.open(image_path).convert('RGB')
            embedding = embedding_generator.run(img)

            records.append({
                "full_path": image_path,
                "image_filename": image_filename,
                "image": img,
                "embedding": embedding
                })

        # Convert records to a pandas DataFrame
        df = pd.DataFrame(records)

        # Save the DataFrame to a .pkl file
        output_filepath = os.path.join(output_dir, f"chunk_{chunk_index}.pkl")
        df.to_pickle(output_filepath)

        print(f"Processed chunk {chunk_index} with {len(chunk)} images.")
        chunk_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Embedding extraction", add_help=True)
    parser.add_argument("--image_dir", type=str, required=True, help="path to image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="path to output dataset directory")
    args = parser.parse_args()
    main(args.image_dir, args.output_dir)
