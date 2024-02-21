#!/bin/bash

output_dir="/checkpoint"
original_args=("$@")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_dir)
        output_dir="$2"
        shift 2
        ;;
        *)
        shift
        ;;
    esac
done

echo "Using output directory: $output_dir"

echo "Waiting for segment processing to complete..."

while [ ! -f "${output_dir}/segment_done.txt" ]; do
  sleep 10
done

echo "Starting pointcloud processing..."
python3 process_pointcloud.py "${original_args[@]}"

rm "${output_dir}/segment_done.txt" 
touch "${output_dir}/pointcloud_done.txt"
