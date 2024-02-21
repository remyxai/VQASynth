import os
import pickle
import argparse
import pandas as pd
from PIL import Image
from vqasynth.datasets.captions import Llava


llm_model = Llava(mmproj="/app/models/mmproj-model-f16.gguf", model_path="/app/models/llava-v1.6-34b.Q4_K_M.gguf", gpu=True)
prompt = 'Create a JSON representation where each entry consists of a key "object" with a numerical suffix starting from 1, and a corresponding "description" key with a value that is a concise, up to six-word sentence describing each main, distinct object or person in the image. Each pair should uniquely describe one element without repeating keys. An example: {"object1": { "description": "Man in red hat walking." },"object2": { "description": "Wooden pallet with boxes." },"object3": { "description": "Cardboard boxes stacked." },"object4": { "description": "Man in green vest standing." }}'

def caption_image_data(image):
    try:
        return llm_model.run_inference(image, prompt, return_json=True)
    except:
        return []

def main(output_dir):
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(output_dir, filename)
            # Load the DataFrame from the .pkl file
            df = pd.read_pickle(pkl_path)

            # Process each image and add the results to a new column
            df['captions'] = df['image'].apply(caption_image_data)

            # Save the updated DataFrame back to the .pkl file
            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--output_dir", type=str, required=True, help="path to directory containing .pkl files")
    args = parser.parse_args()

    main(args.output_dir)
