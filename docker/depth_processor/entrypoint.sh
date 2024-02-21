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

python3 process_depth.py "${original_args[@]}"

touch "${output_dir}/depth_done.txt"
