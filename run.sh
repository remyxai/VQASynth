#!/bin/bash

CONFIG_FILE=./config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
HF_TOKEN=$(cat ~/.cache/huggingface/token)

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Local output directory specified in config.yaml does not exist."
    exit 1
fi

export OUTPUT_DIR="$OUTPUT_DIR"
export HF_TOKEN="$HF_TOKEN"

echo "Building base image..."
docker build -f docker/base_image/Dockerfile -t vqasynth:base .

echo "Launching pipeline"
docker compose -f pipelines/spatialvqa.yaml up --build
