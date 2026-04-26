#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cosil
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
if [[ -n "${PROJECT_FILE_LOC:-}" ]]; then
  export PROJECT_FILE_LOC
else
  unset PROJECT_FILE_LOC
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

export temperature="${temperature:-0.6}"
export top_p="${top_p:-0.95}"
export max_tokens="${max_tokens:-32768}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export LOCAGENT_MODEL="${LOCAGENT_MODEL:-qwen3.6-plus}"

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "[baseline] ERROR: OPENAI_API_KEY is required. Export it before running." >&2
  exit 1
fi

DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
BACKEND="${BACKEND:-openai}"
THREADS="${THREADS:-200}"
NUM_RUNS="${NUM_RUNS:-1}"
NUM_INSTANCES="${NUM_INSTANCES:-0}"
TOP_N="${TOP_N:-5}"
MAX_RETRY="${MAX_RETRY:-10}"
RESULT_ROOT="${RESULT_ROOT:-results/baseline-verified}"
MODEL_TAG="${MODEL_TAG:-${LOCAGENT_MODEL//\//_}}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_baseline_verified_$(date +%Y%m%d_%H%M%S).log}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TIME_JSONL="${RUN_TIME_JSONL:-${RESULT_ROOT}/run_wall_times.jsonl}"
EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${RESULT_ROOT}/multi_run_eval.json}"
EVAL_LOG_FILE="${EVAL_LOG_FILE:-${LOG_DIR}/multi_run_eval.log}"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
mkdir -p "${RESULT_ROOT}"
: > "${RUN_TIME_JSONL}"

echo "[baseline] log_file=${LOG_FILE}"
echo "[baseline] dataset=${DATASET}"
echo "[baseline] model=${LOCAGENT_MODEL}"
echo "[baseline] num_runs=${NUM_RUNS} num_instances=${NUM_INSTANCES} threads=${THREADS}"
echo "[baseline] temperature=${temperature} top_p=${top_p} max_tokens=${max_tokens}"
echo "[baseline] started_at=$(date '+%Y-%m-%d %H:%M:%S')"

for run_id in $(seq 1 "${NUM_RUNS}"); do
  run_start=$(date +%s)
  run_root="${RESULT_ROOT}/run_${run_id}"
  file_output="${run_root}/file_level_${MODEL_TAG}"
  func_output="${run_root}/func_level_${MODEL_TAG}"

  mkdir -p "${file_output}" "${func_output}"

  echo "[baseline][run ${run_id}/${NUM_RUNS}] file-level start: ${file_output}"
  python -u afl/fl/AFL_localize_file.py --file_level \
    --output_folder "${file_output}" \
    --num_threads "${THREADS}" \
    --model "${LOCAGENT_MODEL}" \
    --backend "${BACKEND}" \
    --dataset "${DATASET}" \
    --num_instances "${NUM_INSTANCES}" \
    --skip_existing

  echo "[baseline][run ${run_id}/${NUM_RUNS}] function-level start: ${func_output}"
  python -u afl/fl/AFL_localize_func.py \
    --output_folder "${func_output}" \
    --loc_file "${file_output}/loc_outputs.jsonl" \
    --output_file "loc_${MODEL_TAG}_func.jsonl" \
    --temperature "${temperature}" \
    --top_n "${TOP_N}" \
    --max_retry "${MAX_RETRY}" \
    --model "${LOCAGENT_MODEL}" \
    --backend "${BACKEND}" \
    --dataset "${DATASET}" \
    --num_instances "${NUM_INSTANCES}" \
    --skip_existing \
    --num_threads "${THREADS}"
  run_end=$(date +%s)
  run_seconds=$((run_end - run_start))
  printf '{"run_id": %s, "wall_time_seconds": %s}\n' "${run_id}" "${run_seconds}" >> "${RUN_TIME_JSONL}"
  echo "[baseline][run ${run_id}/${NUM_RUNS}] completed"
done

if [[ "${RUN_EVAL}" == "1" ]]; then
  run_ids=""
  for run_id in $(seq 1 "${NUM_RUNS}"); do
    if [[ -z "${run_ids}" ]]; then
      run_ids="${run_id}"
    else
      run_ids="${run_ids},${run_id}"
    fi
  done
  echo "[baseline] evaluation start: ${EVAL_OUTPUT_JSON}"
  python -u evaluation/multi_run_eval.py \
    --dataset "${DATASET}" \
    --num-instances "${NUM_INSTANCES}" \
    --run-ids "${run_ids}" \
    --loc-file-template "${RESULT_ROOT}/run_{run}/func_level_${MODEL_TAG}/loc_${MODEL_TAG}_func.jsonl" \
    --output-json "${EVAL_OUTPUT_JSON}" \
    --log-file "${EVAL_LOG_FILE}"
  echo "[baseline] evaluation completed: ${EVAL_OUTPUT_JSON}"
fi

echo "[baseline] finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
