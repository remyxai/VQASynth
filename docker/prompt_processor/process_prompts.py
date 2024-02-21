import os
import cv2
import json
import pickle
import random
import itertools
import argparse
import numpy as np
import pandas as pd
from vqasynth.datasets.pointcloud import restore_pointclouds
from vqasynth.datasets.prompts import evaluate_predicates_on_pairs


def prompt_image_data(row):
    image_file = row["image_filename"]
    captions = row["captions"]
    pointclouds = restore_pointclouds(row["pointclouds"])

    try:
        objects = list(zip(captions, pointclouds))
        all_pairs = [(i, j) for i in range(len(objects)) for j in range(len(objects)) if i != j]
        random.shuffle(all_pairs)
        selected_pairs = all_pairs[:5]
        object_pairs = [(objects[i], objects[j]) for i,j in selected_pairs]
        prompts = evaluate_predicates_on_pairs(object_pairs)
    except:
        prompts = []
    return prompts

def main(image_dir, output_dir):
    final_samples = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(output_dir, filename)
            # Load the DataFrame from the .pkl file
            df = pd.read_pickle(pkl_path)

            # Process each image and add the results to a new column
            df['prompts'] = df.apply(prompt_image_data, axis=1)

            # Save the updated DataFrame back to the .pkl file
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
                            # Ensure the image path format fits your structure
                            "image": os.path.join(image_dir, image_filename),
                            "conversations": conversations
                        }
                        final_samples.append(sample)

    # Save the final samples to the output JSON file
    output_json = os.path.join(output_dir, "processed_dataset.json")
    with open(output_json, "w") as json_file:
        json.dump(final_samples, json_file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--image_dir", type=str, required=True, help="path to image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="path to directory containing .pkl files")
    args = parser.parse_args()

    main(args.image_dir, args.output_dir)
