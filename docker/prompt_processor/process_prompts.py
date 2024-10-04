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
from datasets import Dataset
from huggingface_hub import HfApi

def save_results_to_hf(results, hf_dataset_name, hf_token):
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face API token.")

    try:
        hf_dataset = Dataset.from_list(results)
        hf_dataset.push_to_hub(hf_dataset_name, token=hf_token)
        print(f"Results successfully saved to Hugging Face dataset: {hf_dataset_name}")

        api = HfApi()
        api.update_repo_visibility(hf_dataset_name, private=True, token=hf_token)  # Set to private by default
        api.update_repo_metadata(
            repo_id=hf_dataset_name,
            metadata={
                'description': 'Dataset generated by VQASynth powered by Remyx AI',
                'tags': ['VQASynth', 'spatial-reasoning', 'visual-question-answering']
            },
            token=hf_token
        )

    except Exception as e:
        print(f"An error occurred while saving to Hugging Face: {str(e)}")
        raise

def prompt_image_data(row):
    image_file = row["image_filename"]
    captions = row["captions"]
    pointclouds = restore_pointclouds(row["pointclouds"])
    is_canonicalized = row["is_canonicalized"]

    try:
        objects = list(zip(captions, pointclouds))
        all_pairs = [(i, j) for i in range(len(objects)) for j in range(len(objects)) if i != j]
        random.shuffle(all_pairs)
        selected_pairs = all_pairs[:5]
        object_pairs = [(objects[i], objects[j]) for i,j in selected_pairs]
        prompts = evaluate_predicates_on_pairs(object_pairs, is_canonicalized)
    except:
        prompts = []
    return prompts

def save_as_json_local(output_dir, final_samples):
    output_json = os.path.join(output_dir, "processed_dataset.json")
    with open(output_json, "w") as json_file:
        json.dump(final_samples, json_file, indent=4)

def main(image_dir, output_dir, hf_save_dataset_name, hf_token):
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

    # If hf_dataset_name is provided, push to Hugging Face
    if hf_save_dataset_name:
        try:
            save_results_to_hf(final_samples, hf_save_dataset_name, hf_token)
        except Exception as e:
            print(f"Failed to push to Hugging Face: {str(e)}")
            save_as_json_local(output_dir, final_samples)
            print("The processed dataset is still available locally.")
            
    else:
        save_as_json_local(output_dir, final_samples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--image_dir", type=str, required=True, default=None, help="path to image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="path to directory containing .pkl files")
    parser.add_argument("--hf_save_dataset_name", type=str, required=False, default=None, help="name of the dataset to push to huggingface, \
                            if not provided, it will store as a json object locally")
    parser.add_argument("--hf_token", type=str, required=False, default=None, help="token for huggingface")
    args = parser.parse_args()

    main(args.image_dir, args.output_dir, args.hf_save_dataset_name, args.hf_token)
