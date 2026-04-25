#!/bin/bash

set -e

export REPO_DIR="$(pwd)"
export OUTPUT_DIR="${REPO_DIR}/results"

retriever=${1:-"SweRankEmbed-Large"}
RERANKER_MODEL_PATH=${2:-"SweRankLLM-Large"}
RERANKER_TAG=${3:-"SweRankLLM-Large"}
DATASET_DIR=${4:-"./datasets"}
dataset=${5:-"swe-bench-verified"}
split=${6:-"test"}
level=${7:-"function"}
eval_mode=${8:-"default"}

# Default reranking parameters
TOP_K=100
WINDOW_SIZE=10
STEP_SIZE=5

export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

### RETRIEVER OUTPUT PATTERN: model=SweRankEmbed-Large_dataset=swe-bench-lite_split=test_level=function_evalmode=default_results.json

# Reranker output configs
RETRIEVER_OUTPUT_DIR="${OUTPUT_DIR}/model=${retriever}_dataset=${dataset}_split=${split}_level=${level}_evalmode=${eval_mode}_results.json"
DATA_TYPE="${retriever}_${RERANKER_TAG}"

export PYTHONPATH="$(pwd)/src"

# Run the reranker (includes conversion step)
echo "Using Retriever output: ${retriever}"
echo "Running reranker with model: ${RERANKER_MODEL_PATH}"
echo "Reranker tag: ${RERANKER_TAG}"
python src/rerank.py \
    --model ${RERANKER_MODEL_PATH} \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_name ${dataset} \
    --retriever_output_dir ${RETRIEVER_OUTPUT_DIR} \
    --data_type ${DATA_TYPE} \
    --output_dir "${OUTPUT_DIR}" \
    --eval_dir "${OUTPUT_DIR}" \
    --top_k "${TOP_K}" \
    --window_size "${WINDOW_SIZE}" \
    --step_size "${STEP_SIZE}" \
    --use_parallel_reranking

echo "Reranking completed!"

RERANKER_OUTPUT_DIR="${OUTPUT_DIR}/${DATA_TYPE}"

echo "Running evaluation..."
python src/refactored_eval_localization.py \
        --model $retriever \
        --output_dir $OUTPUT_DIR \
        --reranker_output_dir $RERANKER_OUTPUT_DIR \
        --dataset_dir $DATASET_DIR \
        --dataset $dataset
