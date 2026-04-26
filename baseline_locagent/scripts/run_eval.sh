#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Lite}"
DATASET_SUFFIX="${DATASET_SUFFIX:-SWE-bench_Lite}"

export GRAPH_INDEX_DIR="${GRAPH_INDEX_DIR:-}"
export BM25_INDEX_DIR="${BM25_INDEX_DIR:-}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-482fa2a1567041ecafa9c3114bf3811d}"
export LOCAGENT_MODEL="${LOCAGENT_MODEL:-qwen2.5-14b-instruct}"
export LOCAGENT_TEMPERATURE="${LOCAGENT_TEMPERATURE:-0.6}"
export LOCAGENT_TOP_P="${LOCAGENT_TOP_P:-0.95}"
export LOCAGENT_MAX_TOKENS="${LOCAGENT_MAX_TOKENS:-16384}"
export LOCAGENT_GIT_CLONE_TIMEOUT="${LOCAGENT_GIT_CLONE_TIMEOUT:-600}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_PATH="${1:-$(pwd)/results/lite_full_qwen3coder}"
EVAL_N_LIMIT="${EVAL_N_LIMIT:-0}"
NUM_RUNS="${NUM_RUNS:-3}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
# 指定eval的IDs
INSTANCE_IDS_FILE="${INSTANCE_IDS_FILE:-}"
RERUN_EMPTY_LOCATION="${RERUN_EMPTY_LOCATION:-0}"

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "OPENAI_API_KEY is not set."
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} is not available."
  exit 1
fi

mkdir -p "${RESULT_PATH}"

if [[ -z "${GRAPH_INDEX_DIR}" ]]; then
  echo "GRAPH_INDEX_DIR is not set."
  exit 1
fi
if [[ -z "${BM25_INDEX_DIR}" ]]; then
  echo "BM25_INDEX_DIR is not set."
  exit 1
fi

append_metric_summary() {
  local metrics_file="$1"
  local log_file="$2"

  if [[ ! -f "${metrics_file}" ]]; then
    return
  fi

  {
    echo
    echo "[run_eval] Final aggregate metrics from ${metrics_file}:"
    "${PYTHON_BIN}" - "${metrics_file}" <<'PY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as file:
    payload = json.load(file)

print(json.dumps(payload.get('aggregate', payload), indent=2))
PY
  } >> "${log_file}"
}

echo "[run_eval] Using existing indexes:"
echo "  GRAPH_INDEX_DIR=${GRAPH_INDEX_DIR}"
echo "  BM25_INDEX_DIR=${BM25_INDEX_DIR}"

for ((RUN_IDX=1; RUN_IDX<=NUM_RUNS; RUN_IDX++)); do
  if [[ "${NUM_RUNS}" == "1" ]]; then
    RUN_ROOT="${RESULT_PATH}"
  else
    RUN_ROOT="${RESULT_PATH}/run_${RUN_IDX}"
  fi
  RUN_LOCATION_DIR="${RUN_ROOT}/location"
  mkdir -p "${RUN_LOCATION_DIR}"

  LOCALIZE_ARGS=(
    --dataset "${DATASET_NAME}"
    --split test
    --model "${LOCAGENT_MODEL}"
    --localize
    --merge
    --output_folder "${RUN_LOCATION_DIR}"
    --num_samples "${NUM_SAMPLES}"
    --num_processes "${NUM_PROCESSES}"
    --max_iterations "${MAX_ITERATIONS}"
    --use_function_calling
    --simple_desc
  )

  if [[ "${EVAL_N_LIMIT}" != "0" ]]; then
    LOCALIZE_ARGS+=(--eval_n_limit "${EVAL_N_LIMIT}")
  fi
  if [[ -n "${INSTANCE_IDS_FILE}" ]]; then
    LOCALIZE_ARGS+=(--instance_ids_file "${INSTANCE_IDS_FILE}")
  fi
  if [[ "${RERUN_EMPTY_LOCATION}" == "1" ]]; then
    LOCALIZE_ARGS+=(--rerun_empty_location)
  fi

  echo "[run_eval] Localization run ${RUN_IDX}/${NUM_RUNS}"
  "${PYTHON_BIN}" auto_search_main.py "${LOCALIZE_ARGS[@]}" \
    > "${RUN_LOCATION_DIR}/localize.stdout.log" 2>&1

  echo "[run_eval] Evaluating run ${RUN_IDX}/${NUM_RUNS}"
  "${PYTHON_BIN}" evaluation/eval_verified_baseline.py \
    --loc_file "${RUN_LOCATION_DIR}/merged_loc_outputs_mrr.jsonl" \
    --traj_file "${RUN_LOCATION_DIR}/loc_trajs.jsonl" \
    --dataset "${DATASET_NAME}" \
    --split test \
    --output_file "${RUN_LOCATION_DIR}/verified_metrics.json" \
    > "${RUN_LOCATION_DIR}/eval.log" \
    2> >(tee -a "${RUN_LOCATION_DIR}/eval.log" >&2)

  append_metric_summary "${RUN_LOCATION_DIR}/verified_metrics.json" "${RUN_LOCATION_DIR}/eval.log"
done

if [[ "${NUM_RUNS}" != "1" ]]; then
  echo "[run_eval] Aggregating ${NUM_RUNS} runs"
  "${PYTHON_BIN}" evaluation/eval_verified_baseline.py \
    --loc_file merged_loc_outputs_mrr.jsonl \
    --traj_file loc_trajs.jsonl \
    --dataset "${DATASET_NAME}" \
    --split test \
    --num_runs "${NUM_RUNS}" \
    --runs_root "${RESULT_PATH}" \
    --output_file "${RESULT_PATH}/verified_metrics.json" \
    > "${RESULT_PATH}/eval.log" \
    2> >(tee -a "${RESULT_PATH}/eval.log" >&2)

  append_metric_summary "${RESULT_PATH}/verified_metrics.json" "${RESULT_PATH}/eval.log"
fi
