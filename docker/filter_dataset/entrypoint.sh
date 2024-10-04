#!/bin/bash

output_dir="/checkpoint"
filter_dataset=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_dir)
        output_dir="$2"
        shift 2
        ;;
        --classes)
        shift
        classes=()
        while [[ "$#" -gt 0 && "$1" != --* ]]; do
            classes+=("$1")
            shift
        done
        ;;
        --filter_dataset)
        filter_dataset="$2"
        shift 2
        ;;
        --input_dir)
        input_dir="$2"
        shift 2
        ;;
        --hf_dataset)
        hf_dataset="$2"
        shift 2
        ;;
        --hf_token)
        hf_token="$2"
        shift 2
        ;;
        *)
        shift
        ;;
    esac
done

echo "Using output directory: $output_dir"
echo "Using classes: ${classes[@]}"
echo "Filter dataset: $filter_dataset"

echo "Running filter_dataset.py..."
python3 filter_dataset.py --output_dir ${output_dir} --classes "${classes[@]}" ${input_dir:+--input_dir "$input_dir"} ${filter_dataset:+--filter_dataset "$filter_dataset"} ${hf_dataset:+--hf_dataset "$hf_dataset"} ${hf_token:+--hf_token "$hf_token"}

touch "${output_dir}/filter_done.txt"