#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
SOURCE_REPO_ID=$(yq e '.arguments.source_repo_id' $CONFIG_FILE)
TARGET_REPO_NAME=$(yq e '.arguments.target_repo_name' $CONFIG_FILE)
IMAGES=$(yq e '.arguments.images' $CONFIG_FILE)

# Export these values as environment variables
export OUTPUT_DIR

echo "Using output directory: ${OUTPUT_DIR}"
echo "Starting correspondence processing..."

# Standalone stage: does not depend on the main spatial-VQA chain's done-files.
python3 process_correspondence.py \
    --output_dir="${OUTPUT_DIR}" \
    --source_repo_id="${SOURCE_REPO_ID}" \
    --target_repo_name="${TARGET_REPO_NAME}" \
    --images="${IMAGES}"

touch "${OUTPUT_DIR}/correspondence_done.txt"
