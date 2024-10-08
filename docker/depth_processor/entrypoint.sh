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

echo "Waiting for filter processing to complete..."

while [ ! -f "${output_dir}/filter_done.txt" ]; do
  sleep 10
done

echo "Starting depth processing..."
python3 process_depth.py "${original_args[@]}"

rm "${output_dir}/filter_done.txt"
touch "${output_dir}/depth_done.txt"
