#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

IMAGE_DIR=$(yq e '.directories.image_dir' $CONFIG_FILE)
OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)

# Export these values as environment variables
export IMAGE_DIR
export OUTPUT_DIR

echo "Using output directory: $OUTPUT_DIR"

echo "Waiting for filter processing to complete..."

while [ ! -f "${OUTPUT_DIR}/filter_done.txt" ]; do
  sleep 10
done

echo "Starting depth processing..."
python3 process_depth.py \
    --output_dir="${OUTPUT_DIR}"

rm "${OUTPUT_DIR}/filter_done.txt"
touch "${OUTPUT_DIR}/depth_done.txt"
