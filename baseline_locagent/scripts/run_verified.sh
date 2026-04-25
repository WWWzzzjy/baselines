#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-482fa2a1567041ecafa9c3114bf3811d}"
export LOCAGENT_MODEL="${LOCAGENT_MODEL:-qwen3.6-plus}"
export LOCAGENT_TEMPERATURE="${LOCAGENT_TEMPERATURE:-0.6}"
export LOCAGENT_TOP_P="${LOCAGENT_TOP_P:-0.95}"
export LOCAGENT_MAX_TOKENS="${LOCAGENT_MAX_TOKENS:-32768}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set."
  exit 1
fi

if [[ -z "${OPENAI_API_BASE:-}" ]]; then
  echo "OPENAI_API_BASE is not set."
  exit 1
fi

MODEL_NAME="${LOCAGENT_MODEL:-qwen3-coder-plus}"
DATASET_NAME="princeton-nlp/SWE-bench_Verified"
DATASET_SUFFIX="SWE-bench_Verified"
# 0 means no limit: run the full split unless the caller explicitly narrows it.
EVAL_N_LIMIT="${EVAL_N_LIMIT:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NUM_RUNS="${NUM_RUNS:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
INSTANCE_IDS_FILE="${INSTANCE_IDS_FILE:-}"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export GRAPH_INDEX_DIR="${GRAPH_INDEX_DIR:-$(pwd)/index_data/${DATASET_SUFFIX}/graph_index_v2.3}"
export BM25_INDEX_DIR="${BM25_INDEX_DIR:-$(pwd)/index_data/${DATASET_SUFFIX}/BM25_index}"
export LOCAGENT_GIT_CLONE_TIMEOUT="${LOCAGENT_GIT_CLONE_TIMEOUT:-600}"

RESULT_PATH="${1:-$(pwd)/results/verified_full_qwen3coder}"
mkdir -p "${RESULT_PATH}"
PREBUILD_TIME_FILE="${RESULT_PATH}/prebuild_times.json"
PREBUILD_LOG_FILE="${RESULT_PATH}/prebuild.log"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} is not available."
  exit 1
fi

PREBUILD_INDICES="${PREBUILD_INDICES:-1}"
PREBUILD_NUM_PROCESSES="${PREBUILD_NUM_PROCESSES:-1}"
PREBUILD_REPO_PATH="${PREBUILD_REPO_PATH:-$(pwd)/playground/prebuild_verified}"
INSTANCE_ID_JSON_PATH="${INSTANCE_ID_JSON_PATH:-}"

if [[ -n "${INSTANCE_IDS_FILE}" && -z "${INSTANCE_ID_JSON_PATH}" ]]; then
  if [[ "${INSTANCE_IDS_FILE}" == *.json ]]; then
    INSTANCE_ID_JSON_PATH="${INSTANCE_IDS_FILE}"
  else
    INSTANCE_ID_JSON_PATH="${RESULT_PATH}/instance_ids.json"
    "${PYTHON_BIN}" - "${INSTANCE_IDS_FILE}" "${INSTANCE_ID_JSON_PATH}" <<'PY'
import json
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
with open(src_path, "r", encoding="utf-8") as f:
    ids = [line.strip() for line in f if line.strip()]
with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(ids, f, indent=2)
PY
  fi
fi

if [[ -z "${INSTANCE_ID_JSON_PATH}" && "${PREBUILD_INDICES}" == "1" && "${EVAL_N_LIMIT}" != "0" ]]; then
  INSTANCE_ID_JSON_PATH="${RESULT_PATH}/instance_ids.eval_n_limit.json"
  "${PYTHON_BIN}" - "${DATASET_NAME}" "test" "${EVAL_N_LIMIT}" "${INSTANCE_ID_JSON_PATH}" <<'PY'
import json
import sys
from datasets import load_dataset

dataset_name, split, eval_n_limit, output_path = sys.argv[1:5]
eval_n_limit = int(eval_n_limit)
dataset = load_dataset(dataset_name, split=split)
instance_ids = [instance["instance_id"] for instance in dataset.select(range(min(eval_n_limit, len(dataset))))]
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(instance_ids, f, indent=2)
PY
fi

if [[ "${PREBUILD_INDICES}" == "1" ]]; then
  mkdir -p "${PREBUILD_REPO_PATH}"
  PREBUILD_GRAPH_ARGS=(
    --dataset "${DATASET_NAME}"
    --split "test"
    --num_processes "${PREBUILD_NUM_PROCESSES}"
    --download_repo
    --repo_path "${PREBUILD_REPO_PATH}/graph"
    --index_dir "$(pwd)/index_data"
  )
  PREBUILD_BM25_ARGS=(
    --dataset "${DATASET_NAME}" \
    --split "test" \
    --num_processes "${PREBUILD_NUM_PROCESSES}" \
    --download_repo \
    --repo_path "${PREBUILD_REPO_PATH}/bm25" \
    --index_dir "$(pwd)/index_data"
  )

  if [[ -n "${INSTANCE_ID_JSON_PATH}" ]]; then
    PREBUILD_GRAPH_ARGS+=(--instance_id_path "${INSTANCE_ID_JSON_PATH}")
    PREBUILD_BM25_ARGS+=(--instance_id_path "${INSTANCE_ID_JSON_PATH}")
  fi

  echo "[run_verified] Prebuilding graph indexes..." | tee -a "${PREBUILD_LOG_FILE}"
  PREBUILD_START_SECONDS=$(date +%s)
  "${PYTHON_BIN}" dependency_graph/batch_build_graph.py "${PREBUILD_GRAPH_ARGS[@]}" 2>&1 | tee -a "${PREBUILD_LOG_FILE}"
  GRAPH_END_SECONDS=$(date +%s)
  echo "[run_verified] Prebuilding BM25 indexes..." | tee -a "${PREBUILD_LOG_FILE}"
  "${PYTHON_BIN}" build_bm25_index.py "${PREBUILD_BM25_ARGS[@]}" 2>&1 | tee -a "${PREBUILD_LOG_FILE}"
  BM25_END_SECONDS=$(date +%s)
  "${PYTHON_BIN}" - "${PREBUILD_TIME_FILE}" "${PREBUILD_START_SECONDS}" "${GRAPH_END_SECONDS}" "${BM25_END_SECONDS}" <<'PY'
import json
import sys

output_path, start, graph_end, bm25_end = sys.argv[1:5]
start = int(start)
graph_end = int(graph_end)
bm25_end = int(bm25_end)
payload = {
    "graph_seconds": graph_end - start,
    "bm25_seconds": bm25_end - graph_end,
    "total_seconds": bm25_end - start,
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
  echo "[run_verified] Prebuild finished." | tee -a "${PREBUILD_LOG_FILE}"
else
  "${PYTHON_BIN}" - "${PREBUILD_TIME_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump({"graph_seconds": 0, "bm25_seconds": 0, "total_seconds": 0}, f, indent=2)
PY
fi

RUN_EXIT_CODES=()
for ((RUN_IDX=1; RUN_IDX<=NUM_RUNS; RUN_IDX++)); do
  if [[ "${NUM_RUNS}" == "1" ]]; then
    RUN_ROOT="${RESULT_PATH}"
  else
    RUN_ROOT="${RESULT_PATH}/run_${RUN_IDX}"
  fi
  RUN_LOCATION_DIR="${RUN_ROOT}/location"

  LOCALIZE_ARGS=(
    --dataset "${DATASET_NAME}"
    --split "test"
    --model "${MODEL_NAME}"
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

  echo "[run_verified] Starting localization run ${RUN_IDX}/${NUM_RUNS}: ${RUN_LOCATION_DIR}"
  LOCALIZE_EXIT_CODE=0
  if ! "${PYTHON_BIN}" auto_search_main.py "${LOCALIZE_ARGS[@]}"; then
    LOCALIZE_EXIT_CODE=$?
    echo "Run ${RUN_IDX}: localization exited with code ${LOCALIZE_EXIT_CODE}. Will try to merge/evaluate any partial results."
  fi
  RUN_EXIT_CODES+=("${LOCALIZE_EXIT_CODE}")

  LOC_OUTPUT_FILE="${RUN_LOCATION_DIR}/loc_outputs.jsonl"
  MERGED_LOC_FILE="${RUN_LOCATION_DIR}/merged_loc_outputs_mrr.jsonl"
  TRAJ_FILE="${RUN_LOCATION_DIR}/loc_trajs.jsonl"
  METRICS_FILE="${RUN_LOCATION_DIR}/verified_metrics.json"
  RUN_PREBUILD_TIME_FILE="${PREBUILD_TIME_FILE}"
  if [[ "${NUM_RUNS}" != "1" ]]; then
    mkdir -p "${RUN_ROOT}"
    RUN_PREBUILD_TIME_FILE="${RUN_ROOT}/prebuild_times.json"
    "${PYTHON_BIN}" - "${PREBUILD_TIME_FILE}" "${RUN_PREBUILD_TIME_FILE}" "${NUM_RUNS}" <<'PY'
import json
import sys

src_path, dst_path, num_runs = sys.argv[1:4]
num_runs = int(num_runs)
with open(src_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
split_payload = {
    key: value / num_runs for key, value in payload.items()
}
with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(split_payload, f, indent=2)
PY
  fi

  if [[ -f "${LOC_OUTPUT_FILE}" ]]; then
    echo "[run_verified] Merging localization outputs for run ${RUN_IDX}/${NUM_RUNS}."
    "${PYTHON_BIN}" auto_search_main.py \
      --dataset "${DATASET_NAME}" \
      --split "test" \
      --model "${MODEL_NAME}" \
      --merge \
      --output_folder "${RUN_LOCATION_DIR}"

    if [[ -f "${MERGED_LOC_FILE}" ]]; then
      echo "[run_verified] Evaluating run ${RUN_IDX}/${NUM_RUNS}."
      "${PYTHON_BIN}" evaluation/eval_verified_baseline.py \
        --loc_file "${MERGED_LOC_FILE}" \
        --traj_file "${TRAJ_FILE}" \
        --dataset "${DATASET_NAME}" \
        --split "test" \
        --prebuild_time_file "${RUN_PREBUILD_TIME_FILE}" \
        --output_file "${METRICS_FILE}"
    fi
  fi
done

if [[ "${NUM_RUNS}" != "1" ]]; then
  echo "[run_verified] Aggregating ${NUM_RUNS} run-level metrics."
  "${PYTHON_BIN}" evaluation/eval_verified_baseline.py \
    --loc_file "merged_loc_outputs_mrr.jsonl" \
    --traj_file "loc_trajs.jsonl" \
    --dataset "${DATASET_NAME}" \
    --split "test" \
    --num_runs "${NUM_RUNS}" \
    --runs_root "${RESULT_PATH}" \
    --prebuild_time_file "${PREBUILD_TIME_FILE}" \
    --output_file "${RESULT_PATH}/verified_metrics.json"
fi

for EXIT_CODE in "${RUN_EXIT_CODES[@]}"; do
  if [[ "${EXIT_CODE}" != "0" ]]; then
    exit "${EXIT_CODE}"
  fi
done
