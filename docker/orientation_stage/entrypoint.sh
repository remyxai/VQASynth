#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
SOURCE_REPO_ID=$(yq e '.arguments.source_repo_id' $CONFIG_FILE)
IMAGES=$(yq e '.arguments.images' $CONFIG_FILE)

# Export these values as environment variables
export OUTPUT_DIR

echo "Using output directory: $OUTPUT_DIR"

echo "Waiting for location refinement (object masks) to complete..."

# Read-only wait: do NOT remove location_refinement_done.txt. scene_fusion also
# waits on it and consumes (rm) it. Orientation only needs the per-object masks
# produced by the location_refinement stage, so it runs alongside scene_fusion.
while [ ! -f "${OUTPUT_DIR}/location_refinement_done.txt" ]; do
  sleep 10
done

echo "Starting orientation processing..."
python3 process_orientation.py \
    --output_dir="${OUTPUT_DIR}" \
    --source_repo_id="${SOURCE_REPO_ID}" \
    --images="${IMAGES}" \
    --masks="masks"

touch "${OUTPUT_DIR}/orientation_done.txt"
