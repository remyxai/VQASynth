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

# Start the filtering process
echo "Starting image filtering process..."
python3 process_filter.py "${original_args[@]}"

# Mark filtering as done
touch "${output_dir}/filter_done.txt"
