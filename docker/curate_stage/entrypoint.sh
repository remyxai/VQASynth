#!/bin/bash

# Parse the config.yaml file using yq. Curation-specific keys default here so a
# stock config.yaml works unchanged; override them under `arguments:` to tune.
CONFIG_FILE=/app/config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
SOURCE_REPO_ID=$(yq e '.arguments.source_repo_id' $CONFIG_FILE)
TARGET_REPO_NAME=$(yq e '.arguments.target_repo_name // "vqasynth_sample_curated"' $CONFIG_FILE)
IMAGES=$(yq e '.arguments.images' $CONFIG_FILE)

CURATE_STRATEGY=$(yq e '.arguments.curate_strategy // "farthest"' $CONFIG_FILE)
CURATE_FRACTION=$(yq e '.arguments.curate_fraction // "0.25"' $CONFIG_FILE)
CURATE_COUNT=$(yq e '.arguments.curate_count // ""' $CONFIG_FILE)
CURATE_SEED=$(yq e '.arguments.curate_seed // "0"' $CONFIG_FILE)
CURATE_METRIC=$(yq e '.arguments.curate_metric // "euclidean"' $CONFIG_FILE)
CURATE_SPLIT=$(yq e '.arguments.curate_split // "train"' $CONFIG_FILE)
CURATE_PUSH=$(yq e '.arguments.curate_push // "false"' $CONFIG_FILE)

# Export these values as environment variables
export OUTPUT_DIR

echo "Using output directory: $OUTPUT_DIR"
echo "Waiting for embedding processing to complete..."

# Curation consumes the `embedding` column written by the embeddings stage, so
# it waits on the same sentinel the filter stage waits on. It does NOT remove
# the sentinel — the filter stage owns that in the main pipeline, and curation
# is an auxiliary consumer that may run alongside it.
while [ ! -f "${OUTPUT_DIR}/embeddings_done.txt" ]; do
  sleep 10
done

echo "Starting curation (strategy=${CURATE_STRATEGY}, fraction=${CURATE_FRACTION}, count=${CURATE_COUNT})..."

ARGS=(
  python3 process_curate.py
  --output_dir="${OUTPUT_DIR}"
  --source_repo_id="${SOURCE_REPO_ID}"
  --target_repo_name="${TARGET_REPO_NAME}"
  --strategy="${CURATE_STRATEGY}"
  --fraction="${CURATE_FRACTION}"
  --count="${CURATE_COUNT}"
  --seed="${CURATE_SEED}"
  --metric="${CURATE_METRIC}"
  --images="${IMAGES}"
  --split="${CURATE_SPLIT}"
)
if [ "${CURATE_PUSH}" = "true" ]; then
  ARGS+=(--push_to_hub)
fi
"${ARGS[@]}"

touch "${OUTPUT_DIR}/curate_done.txt"
