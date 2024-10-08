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

# Start the filtering process
echo "Starting image filtering process..."
python3 process_filter.py \
    --image_dir="${IMAGE_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --include_tags="${INCLUDE_TAGS}" \
    --exclude_tags="${EXCLUDE_TAGS}"

# Mark filtering as done
touch "${OUTPUT_DIR}/filter_done.txt"
