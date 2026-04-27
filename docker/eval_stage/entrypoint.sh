#!/bin/bash

# The eval stage runs a HuggingFace VLM against spatial reasoning benchmarks
# (SpatialScore, OmniSpatial, SpaCE-10, MindCube) and produces a per-benchmark
# accuracy report. It is a post-training step — run after a user has fine-tuned
# a model on a VQASynth-produced dataset — and does not depend on the synthesis
# pipeline outputs.

set -euo pipefail

CONFIG_FILE=/app/config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
OPENAI_API_KEY=$(yq e '.arguments.openai_key // ""' $CONFIG_FILE)
BENCHMARK=$(yq e '.arguments.eval_benchmark // "all"' $CONFIG_FILE)
HF_MODEL=$(yq e '.arguments.eval_model // ""' $CONFIG_FILE)

if [ -z "${HF_MODEL}" ] || [ "${HF_MODEL}" = "null" ]; then
    echo "ERROR: arguments.eval_model is required in config.yaml" >&2
    echo "Set it to a HuggingFace VLM slug (e.g., remyxai/SpaceThinker-Qwen2.5VL-3B)" >&2
    exit 1
fi

export OUTPUT_DIR

# Default to using the LLM judge fallback whenever an OpenAI key is present;
# OmniSpatial scoring uses LLM judge as its primary path, so without a key
# OmniSpatial will not produce meaningful scores.
JUDGE_FLAG=""
if [ -n "${OPENAI_API_KEY}" ] && [ "${OPENAI_API_KEY}" != "null" ]; then
    JUDGE_FLAG="--use_llm_judge"
fi

echo "Starting evaluation (model: ${HF_MODEL}, benchmark: ${BENCHMARK})..."
python3 process_eval.py \
    --output_dir="${OUTPUT_DIR}" \
    --model="${HF_MODEL}" \
    --benchmark="${BENCHMARK}" \
    --api_key="${OPENAI_API_KEY}" \
    ${JUDGE_FLAG}
