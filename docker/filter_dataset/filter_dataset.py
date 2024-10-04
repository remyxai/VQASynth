import os
import csv
from PIL import Image
import torch
import clip
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import argparse
import logging
from datasets import load_dataset

def load_from_hf(image_dir: str, hf_dataset: str, hf_token: str):
    print(image_dir, hf_dataset, hf_token)
    try:
        dataset = load_dataset(hf_dataset)
        logging.info(f"Loaded dataset from HuggingFace: {hf_dataset}")
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

def main(args: argparse.Namespace):
    
    if args.input_dir is None:
        input_dir = args.output_dir + "/images"
        dataset = load_from_hf(input_dir, args.hf_dataset, args.hf_token)
        if dataset is None:
            logging.error("Failed to load dataset from HuggingFace. Exiting.")
            return
    else:
        input_dir = args.input_dir

    if not args.filter_dataset:
        logging.info("Running with no filtering")
        return

    logging.info("Running with filtering")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    classes = args.classes

    text_inputs = clip.tokenize([f'{class_}' for class_ in classes]).to(device)

    @torch.no_grad()
    def encode_text():
        text_features = model.encode_text(text_inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    text_features = encode_text()

    def process_batch(batch: list[Image.Image], preprocess: callable, model: callable):
        inputs = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(inputs)
        return image_features.cpu().numpy()

    def process_image_batch(image_paths: list[str]):
        results = []
        try:
            images = [Image.open(path).convert('RGB') for path in image_paths]
            
            image_features = process_batch(images, preprocess, model)
            image_features = torch.from_numpy(image_features).to(device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            max_confidences, predicted_classes = similarity.max(dim=-1)
            
            for path, confidence, class_idx in zip(image_paths, max_confidences, predicted_classes):
                if confidence > 0.75:
                    logging.info(f"Image {path} classified as {classes[class_idx]} with confidence {confidence.item()}")
                    results.append((path, confidence.item(), classes[class_idx]))
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
        
        return results

    def process_directory(input_dir: str):
        image_paths = []
        for root, _, files in os.walk(input_dir):
            for idx, file in enumerate(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_paths.append(os.path.join(root, file))
        
        results = []
        batch_size = 32
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_batch = {executor.submit(process_image_batch, image_paths[i:i+batch_size]): i 
                               for i in range(0, len(image_paths), batch_size)}
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(future_to_batch), desc="Processing batches"):
                results.extend(future.result())
        
        return results

    

    results = process_directory(input_dir)

    for result in results:
        os.remove(result[0])

    logging.info(f"Filtered dataset in {input_dir}. Removed non-'other' images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter dataset using CLIP model')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--input_dir', type=str, default=None, help='Input directory containing images')
    parser.add_argument('--classes', nargs='+', default=["mobile screenshot", "digital graphic", "outdoor scene", "digital logo", "e-commerce website", "indoor scene"], help='Classes to match')
    parser.add_argument("--hf_dataset", type=str, default=None, help="repo id of huggingface dataset")
    parser.add_argument("--hf_token", type=str, default=None, help="token for huggingface")
    parser.add_argument("--filter_dataset", type=bool, default=False, help="filter dataset")
    args = parser.parse_args()
    
    assert args.input_dir or args.hf_dataset, "Either --input_dir or --hf_dataset must be provided"
    
    main(args)