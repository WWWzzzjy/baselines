#!/bin/bash

set -euo pipefail

export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-482fa2a1567041ecafa9c3114bf3811d}"
export LOCAGENT_MODEL="${LOCAGENT_MODEL:-qwen2.5-14b-instruct}"
export RERANK_TAG="${RERANK_TAG:-rerank-small}"

RETRIEVER_MODEL_NAME="${RETRIEVER_MODEL_NAME:-Salesforce/SweRankEmbed-Small}"
RETRIEVER_MODEL_TAG="${RETRIEVER_MODEL_TAG:-SweRankEmbed-Small}"
RETRIEVER_BATCH_SIZE="${RETRIEVER_BATCH_SIZE:-16}"
RETRIEVER_SEQUENCE_LENGTH="${RETRIEVER_SEQUENCE_LENGTH:-1024}"

DATASET_DIR="${1:-${REPO_DIR}/datasets}"
DATASET_NAME="${2:-swe-bench-lite}"
# 0～all
NUM_INSTANCES="${3:-0}"
NUM_RUNS="${4:-3}"
OUTPUT_DIR="${5:-${REPO_DIR}/outputs}"
EVAL_DIR="${6:-${REPO_DIR}/eval_results}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/baseline_eval_dashscope.log}"

SPLIT="${SPLIT:-test}"
LEVEL="${LEVEL:-function}"
EVAL_MODE="${EVAL_MODE:-default}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TOP_K="${TOP_K:-100}"
WINDOW_SIZE="${WINDOW_SIZE:-10}"
STEP_SIZE="${STEP_SIZE:-5}"
RUN_RETRIEVER="${RUN_RETRIEVER:-1}"

mkdir -p "${OUTPUT_DIR}" "${EVAL_DIR}"
cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "Dataset dir: ${DATASET_DIR}"
echo "Dataset name: ${DATASET_NAME}"
echo "Retriever batch size: ${RETRIEVER_BATCH_SIZE}"
echo "Retriever sequence length: ${RETRIEVER_SEQUENCE_LENGTH}"
echo "Num instances: ${NUM_INSTANCES}"
echo "Num runs: ${NUM_RUNS}"
echo "Rerank tag: ${RERANK_TAG}"

CURRENT_INSTANCE_COUNT=0
if [[ -d "${DATASET_DIR}" ]]; then
  CURRENT_INSTANCE_COUNT=$(find "${DATASET_DIR}" -maxdepth 1 -type d -name "${DATASET_NAME}-function_*" | wc -l | tr -d ' ')
fi
echo "Found ${CURRENT_INSTANCE_COUNT} local ${DATASET_NAME} ${LEVEL}-level instances in ${DATASET_DIR}."

if [[ "${NUM_INSTANCES}" -le 0 ]]; then
  echo "NUM_INSTANCES=${NUM_INSTANCES}; using all available local ${DATASET_NAME} instances under ${DATASET_DIR}."
  if [[ "${CURRENT_INSTANCE_COUNT}" -eq 0 ]]; then
    echo "No local ${DATASET_NAME} instances found. Building the default first instance; this may clone/fetch source repos."
    python src/build_verified_beir_subset.py \
      --dataset_dir "${DATASET_DIR}" \
      --dataset_name "${DATASET_NAME}" \
      --num_instances 1
  else
    echo "Local dataset is available. Skipping dataset build and source repo clone/fetch."
  fi
elif [[ ! -d "${DATASET_DIR}" || "${CURRENT_INSTANCE_COUNT}" -lt "${NUM_INSTANCES}" ]]; then
  echo "Need ${NUM_INSTANCES} ${DATASET_NAME} instances, but found ${CURRENT_INSTANCE_COUNT}. Building up to ${NUM_INSTANCES}; this may clone/fetch source repos."
  python src/build_verified_beir_subset.py \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --num_instances "${NUM_INSTANCES}"
else
  echo "Found enough local ${DATASET_NAME} instances (${CURRENT_INSTANCE_COUNT} >= ${NUM_INSTANCES}). Skipping dataset build and source repo clone/fetch."
fi

RETRIEVER_FILE_TAG="${DATASET_NAME}_${LEVEL}"
RETRIEVER_OUTPUT_FILE="${OUTPUT_DIR}/retriever_${RETRIEVER_FILE_TAG}_eval_summary.json"
RETRIEVER_RESULTS_FILE="${OUTPUT_DIR}/retriever_${RETRIEVER_FILE_TAG}_results.json"
echo "Retriever summary file: ${RETRIEVER_OUTPUT_FILE}"
echo "Retriever results file: ${RETRIEVER_RESULTS_FILE}"

if [[ "${RUN_RETRIEVER}" == "1" ]]; then
  echo "Running retriever with model: ${RETRIEVER_MODEL_NAME}"
  RETRIEVER_START_TS=$(date +%s)
  python src/eval_beir_sbert_canonical.py \
    --dataset_dir "${DATASET_DIR}" \
    --dataset "${DATASET_NAME}" \
    --model "${RETRIEVER_MODEL_NAME}" \
    --batch_size "${RETRIEVER_BATCH_SIZE}" \
    --sequence_length "${RETRIEVER_SEQUENCE_LENGTH}" \
    --output_file "${RETRIEVER_OUTPUT_FILE}" \
    --results_file "${RETRIEVER_RESULTS_FILE}" \
    --eval_mode "${EVAL_MODE}" \
    --split "${SPLIT}" \
    --level "${LEVEL}" \
    --add_prefix
  RETRIEVER_END_TS=$(date +%s)
  RETRIEVER_TIME_SECONDS=$((RETRIEVER_END_TS - RETRIEVER_START_TS))
  echo "Retriever wall-clock time: ${RETRIEVER_TIME_SECONDS}s"
else
  echo "Skipping retriever because RUN_RETRIEVER=${RUN_RETRIEVER}"
  RETRIEVER_TIME_SECONDS="${RETRIEVER_TIME_SECONDS:-0}"
fi

if [[ ! -f "${RETRIEVER_RESULTS_FILE}" ]]; then
  echo "Retriever results file not found: ${RETRIEVER_RESULTS_FILE}" >&2
  echo "Either enable RUN_RETRIEVER=1 or place the retriever output at the expected path." >&2
  exit 1
fi

echo "Running baseline eval with retriever output: ${RETRIEVER_RESULTS_FILE}"
python src/run_swerank_baseline_eval.py \
  --retriever_output_dir "${RETRIEVER_RESULTS_FILE}" \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_dir "${EVAL_DIR}" \
  --model "${LOCAGENT_MODEL}" \
  --api_base "${OPENAI_API_BASE}" \
  --api_key "${OPENAI_API_KEY}" \
  --num_instances "${NUM_INSTANCES}" \
  --num_runs "${NUM_RUNS}" \
  --top_k "${TOP_K}" \
  --window_size "${WINDOW_SIZE}" \
  --step_size "${STEP_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --retriever_time_seconds "${RETRIEVER_TIME_SECONDS}" \
  --rerank_tag "${RERANK_TAG}" \
  --run_tag "dashscope-baseline-${RETRIEVER_MODEL_TAG}"
