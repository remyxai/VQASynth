#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
SOURCE_REPO_ID=$(yq e '.arguments.source_repo_id' $CONFIG_FILE)
IMAGES=$(yq e '.arguments.images' $CONFIG_FILE)

# Export these values as environment variables
export OUTPUT_DIR

# Start the filtering process
echo "Starting image embedding extraction process..."
python3 process_embeddings.py \
    --output_dir="${OUTPUT_DIR}" \
    --source_repo_id="${SOURCE_REPO_ID}" \
    --images="${IMAGES}"

# Mark filtering as done
touch "${OUTPUT_DIR}/embeddings_done.txt"
