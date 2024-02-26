#!/bin/bash

output_dir="/checkpoint"
image_dir=""  # Initialize variable for image_dir
original_args=("$@")

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --image_dir)
            image_dir="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done


# Before starting, optionally check if the HUGGINGFACE_HUB_TOKEN is set.
if [ -z "${HUGGINGFACE_HUB_TOKEN}" ]; then
    echo "HUGGINGFACE_HUB_TOKEN is not set. Please provide a Hugging Face authentication token."
    exit 1
fi

echo "Using output directory: $output_dir"
echo "Image directory set to: $image_dir"

echo "Waiting for data processing to complete..."

while [ ! -f "${output_dir}/data_processing_done.txt" ]; do
  sleep 10
done

echo "Starting LoRA fine-tuning..."
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path "${output_dir}/processed_dataset.json" \
    --image_folder "${image_dir}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${output_dir}/checkpoints/llava-v1.5-13b-task-lora" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True

rm "${output_dir}/data_processing_done.txt"
touch "${output_dir}/training_done.txt"
