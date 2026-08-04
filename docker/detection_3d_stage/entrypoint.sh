#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
SOURCE_REPO_ID=$(yq e '.arguments.source_repo_id' $CONFIG_FILE)
IMAGES=$(yq e '.arguments.images' $CONFIG_FILE)

# Export these values as environment variables
export OUTPUT_DIR

echo "Using output directory: $OUTPUT_DIR"

echo "Waiting for scene fusion processing to complete..."

while [ ! -f "${OUTPUT_DIR}/scene_fusion_done.txt" ]; do
  sleep 10
done

echo "Starting 3D object-detection processing..."
python3 process_detection_3d.py \
    --output_dir="${OUTPUT_DIR}" \
    --source_repo_id="${SOURCE_REPO_ID}" \
    --images="${IMAGES}"

# NOTE: scene_fusion_done.txt is intentionally left in place for the prompt
# stage to consume; this stage only signals its own completion.
touch "${OUTPUT_DIR}/detection_3d_done.txt"
