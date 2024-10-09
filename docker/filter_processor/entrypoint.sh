#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

IMAGE_DIR=$(yq e '.directories.image_dir' $CONFIG_FILE)
OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
INCLUDE_TAGS=$(yq e '.arguments.include_tags' $CONFIG_FILE)
EXCLUDE_TAGS=$(yq e '.arguments.exclude_tags' $CONFIG_FILE)

# Export these values as environment variables
export IMAGE_DIR
export OUTPUT_DIR

echo "Using output directory: $OUTPUT_DIR"
echo "Waiting for embedding processing to complete..."

while [ ! -f "${OUTPUT_DIR}/embeddings_done.txt" ]; do
  sleep 10
done

echo "Starting filtering processing..."
python3 process_filter.py \
    --output_dir="${OUTPUT_DIR}" \
    --include_tags="${INCLUDE_TAGS}" \
    --exclude_tags="${EXCLUDE_TAGS}"

rm "${OUTPUT_DIR}/embeddings_done.txt"
touch "${OUTPUT_DIR}/filter_done.txt"
