#!/bin/bash

output_dir="/checkpoint"
original_args=("$@")

if [[ -n "$INCLUDE_TAGS" ]]; then
    original_args+=("--include_tags" "$INCLUDE_TAGS")
fi

if [[ -n "$EXCLUDE_TAGS" ]]; then
    original_args+=("--exclude_tags" "$EXCLUDE_TAGS")
fi

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

python3 process_filter.py "${original_args[@]}"

touch "${output_dir}/filter_done.txt"

