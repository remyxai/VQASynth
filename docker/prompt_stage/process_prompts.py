import os
import cv2
import json
import pickle
import random
import itertools
import argparse
import numpy as np
import pandas as pd
from vqasynth.prompts import PromptGenerator

def main(image_dir, output_dir):
    prompt_generator = PromptGenerator()
    final_samples = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(output_dir, filename)
            df = pd.read_pickle(pkl_path)

            df['prompts'] = df.apply(lambda row: prompt_generator.run(row["image_filename"], row["captions"], row["pointclouds"], row["is_canonicalized"]), axis=1)

            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")

            for index, row in df.iterrows():
                if row['prompts'] and any(row['prompts']):
                    image_filename = row['image_filename']
                    conversations = []
                    first_prompt = True
                    for prompt in row['prompts']:
                        if 'Answer: ' in prompt:
                            question, answer = prompt.split('Answer: ', 1)
                            human_value = f"<image>\n{question}" if first_prompt else question
                            conversations.append({
                                "from": "human",
                                "value": human_value
                            })
                            conversations.append({
                                "from": "gpt",
                                "value": answer
                            })
                            first_prompt = False

                    if conversations:  # Ensure we have valid conversation data
                        sample = {
                            "id": image_filename,
                            "image": os.path.join(image_dir, image_filename),
                            "conversations": conversations
                        }
                        final_samples.append(sample)

    output_json = os.path.join(output_dir, "processed_dataset.json")
    with open(output_json, "w") as json_file:
        json.dump(final_samples, json_file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--image_dir", type=str, required=True, help="path to image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="path to directory containing .pkl files")
    args = parser.parse_args()

    main(args.image_dir, args.output_dir)
