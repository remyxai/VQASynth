#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
SOURCE_REPO_ID=$(yq e '.arguments.source_repo_id' $CONFIG_FILE)
IMAGE_COL=$(yq e '.arguments.image_col' $CONFIG_FILE)

# Export these values as environment variables
export IMAGE_DIR
export OUTPUT_DIR

echo "Using output directory: $OUTPUT_DIR"
echo "Waiting for depth processing to complete..."

while [ ! -f "${OUTPUT_DIR}/depth_done.txt" ]; do
  sleep 10
done

echo "Starting location refinement processing..."
python3 process_location_refinement.py \
    --output_dir="${OUTPUT_DIR}" \
    --source_repo_id="${SOURCE_REPO_ID}" \
    --image_col="${IMAGE_COL}"

rm "${OUTPUT_DIR}/depth_done.txt" 
touch "${OUTPUT_DIR}/location_refinement_done.txt"
