#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

IMAGE_DIR=$(yq e '.directories.image_dir' $CONFIG_FILE)
OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)

# Export these values as environment variables
export IMAGE_DIR
export OUTPUT_DIR

# Start the filtering process
echo "Starting image embedding extraction process..."
python3 process_embeddings.py \
    --image_dir="${IMAGE_DIR}" \
    --output_dir="${OUTPUT_DIR}"

# Mark filtering as done
touch "${OUTPUT_DIR}/embeddings_done.txt"
