#!/bin/bash

set -e

export REPO_DIR="$(pwd)"
export OUTPUT_DIR="${REPO_DIR}/results"

retriever=${1:-"SweRankEmbed-Small"}
RERANKER_MODEL_PATH=${2:-"openai/qwen3.5-35b-a3b"}
RERANKER_TAG=${3:-"qwen3.5-35b-a3b"}
DATASET_DIR=${4:-"./datasets/loc-bench"}
dataset=${5:-"loc-bench"}
split=${6:-"test"}
level=${7:-"function"}
eval_mode=${8:-"default"}
NUM_RUNS=2
API_KEY=${9:-"sk-482fa2a1567041ecafa9c3114bf3811d"}
API_BASE=${10:-"${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"}


if [[ "${RERANKER_MODEL_PATH}" == qwen* ]]; then
    RERANKER_MODEL_PATH="openai/${RERANKER_MODEL_PATH}"
fi

if [ -z "${API_KEY}" ]; then
    echo "ERROR: API key is required. Set OPENAI_API_KEY or pass it as the 9th argument."
    exit 1
fi

export OPENAI_API_KEY="${API_KEY}"
export OPENAI_API_BASE="${API_BASE}"

# Default reranking parameters
TOP_K=100
WINDOW_SIZE=10
STEP_SIZE=5
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=32768

export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

### RETRIEVER OUTPUT PATTERN: model=SweRankEmbed-Large_dataset=swe-bench-lite_split=test_level=function_evalmode=default_results.json

# Reranker output configs
RETRIEVER_OUTPUT_DIR="${OUTPUT_DIR}/model=${retriever}_dataset=${dataset}_split=${split}_level=${level}_evalmode=${eval_mode}_results.json"
DATA_TYPE="${retriever}_${RERANKER_TAG}"

export PYTHONPATH="$(pwd)/src"

for i in $(seq 1 $NUM_RUNS); do
    echo "===== Run $i / $NUM_RUNS ====="
    RUN_DATA_TYPE="${DATA_TYPE}_run${i}"
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/${RUN_DATA_TYPE}"

    python src/rerank.py \
        --model ${RERANKER_MODEL_PATH} \
        --dataset_dir "${DATASET_DIR}" \
        --dataset_name ${dataset} \
        --retriever_output_dir ${RETRIEVER_OUTPUT_DIR} \
        --data_type ${RUN_DATA_TYPE} \
        --output_dir "${OUTPUT_DIR}" \
        --eval_dir "${OUTPUT_DIR}" \
        --top_k "${TOP_K}" \
        --window_size "${WINDOW_SIZE}" \
        --step_size "${STEP_SIZE}" \
        --api_key ${API_KEY} \
        --api_base "${API_BASE}" \
        --temperature 0.6
done

echo "Reranking completed!"

RERANKER_OUTPUT_DIR="${OUTPUT_DIR}/${DATA_TYPE}"

echo "Running evaluation..."
python src/refactored_eval_localization.py \
        --model $retriever \
        --output_dir $OUTPUT_DIR \
        --reranker_output_dir $RERANKER_OUTPUT_DIR \
        --dataset_dir $DATASET_DIR \
        --dataset $dataset \
        --tokenizer_model ${RERANKER_TAG} \
        --num_runs $NUM_RUNS
