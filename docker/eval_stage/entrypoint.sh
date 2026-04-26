#!/bin/bash

# Parse the config.yaml file using yq or python
CONFIG_FILE=/app/config/config.yaml

OUTPUT_DIR=$(yq e '.directories.output_dir' $CONFIG_FILE)
SOURCE_REPO_ID=$(yq e '.arguments.source_repo_id' $CONFIG_FILE)
TARGET_REPO_NAME=$(yq e '.arguments.target_repo_name' $CONFIG_FILE)
OPENAI_API_KEY=$(yq e '.arguments.openai_key' $CONFIG_FILE)
BENCHMARK=$(yq e '.arguments.eval_benchmark // "all"' $CONFIG_FILE)
# The eval stage scores a column of model predictions (one prediction per QA
# pair) against the ground-truth messages column. The predictions column is
# expected to be populated upstream by a separate model-inference step; the
# preceding r1_reasoning_stage produces CoT training traces, not eval
# predictions, so its `output` column is not a drop-in substitute.
PREDICTION_COLUMN=$(yq e '.arguments.prediction_column // "predictions"' $CONFIG_FILE)
GROUND_TRUTH_COLUMN=$(yq e '.arguments.ground_truth_column // "messages"' $CONFIG_FILE)

# Export these values as environment variables
export OUTPUT_DIR

echo "Using output directory: $OUTPUT_DIR"

echo "Waiting for data processing to complete..."

while [ ! -f "${OUTPUT_DIR}/data_processing_done.txt" ]; do
  sleep 10
done

# Default to using the LLM judge fallback whenever an OpenAI key is present;
# without it, hedged or paraphrased answers (e.g. "Actually, ...") that don't
# pattern-match the rule-based extractors get scored 0.
JUDGE_FLAG=""
if [ -n "${OPENAI_API_KEY}" ]; then
    JUDGE_FLAG="--use_llm_judge"
fi

echo "Starting evaluation (benchmark: ${BENCHMARK}, prediction_column: ${PREDICTION_COLUMN})..."
python3 process_eval.py \
    --output_dir="${OUTPUT_DIR}" \
    --source_repo_id="${SOURCE_REPO_ID}" \
    --target_repo_name="${TARGET_REPO_NAME}" \
    --prediction_column="${PREDICTION_COLUMN}" \
    --ground_truth_column="${GROUND_TRUTH_COLUMN}" \
    --api_key="${OPENAI_API_KEY}" \
    --benchmark="${BENCHMARK}" \
    ${JUDGE_FLAG}

rm "${OUTPUT_DIR}/data_processing_done.txt"
touch "${OUTPUT_DIR}/data_processing_done.txt"
