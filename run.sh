#!/bin/bash

CONFIG_FILE=./config/config.yaml

IMAGE_DIR=$(yq e '.directories.image_dir' $CONFIG_FILE)
OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)

if [ ! -d "$IMAGE_DIR" ] || [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: One or both directories specified in config.yaml do not exist."
    exit 1
fi

export IMAGE_DIR="$IMAGE_DIR"
export OUTPUT_DIR="$OUTPUT_DIR"

docker compose -f pipelines/spatialvqa.yaml up --build
