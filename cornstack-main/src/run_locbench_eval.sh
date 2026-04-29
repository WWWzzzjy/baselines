#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluations/eval_swebench.py"

DATASET_DIR="/Users/Zhuanz/Documents/pyproject/baselines/baseline_swerank/datasets/loc-bench"
OUTPUT_DIR="${SCRIPT_DIR}/results"
DATASET="loc-bench-v1"
SPLIT="test"
LEVEL="function"
MODEL="cornstack/CodeRankEmbed"
BATCH_SIZE=8
SEQUENCE_LENGTH=1024
QUERY_PREFIX="Represent this query for searching relevant code"
PYTHON_BIN="${PYTHON:-python}"
TOK=0
NUM_INSTANCES=1
NUM_RUNS=3


if [[ "${LEVEL}" != "function" && "${LEVEL}" != "file" ]]; then
    echo "Error: --level must be 'function' or 'file'." >&2
    exit 1
fi

if [[ -n "${NUM_INSTANCES}" && ! "${NUM_INSTANCES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num_instances must be a positive integer." >&2
    exit 1
fi

if [[ -n "${NUM_RUNS}" && ! "${NUM_RUNS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_RUNS must be a positive integer." >&2
    exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "Error: dataset directory does not exist: ${DATASET_DIR}" >&2
    exit 1
fi

if [[ "${LEVEL}" == "file" ]]; then
    INSTANCE_PREFIX="${DATASET}"
else
    INSTANCE_PREFIX="${DATASET}-${LEVEL}"
fi

if [[ "${SPLIT}" != "test" ]]; then
    INSTANCE_PREFIX="${DATASET}-${SPLIT}"
    if [[ "${LEVEL}" != "file" ]]; then
        INSTANCE_PREFIX="${INSTANCE_PREFIX}-${LEVEL}"
    fi
fi

if ! find "${DATASET_DIR}" -maxdepth 1 -type d -name "${INSTANCE_PREFIX}_*" -print -quit | grep -q .; then
    echo "Error: no BEIR instance directories found with prefix '${INSTANCE_PREFIX}_' in ${DATASET_DIR}" >&2
    echo "Example expected directory: ${DATASET_DIR}/${INSTANCE_PREFIX}_<instance_id>" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
    "${PYTHON_BIN}" "${EVAL_SCRIPT}"
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --level "${LEVEL}"
    --model "${MODEL}"
    --dataset_dir "${DATASET_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --sequence_length "${SEQUENCE_LENGTH}"
    --query_prefix "${QUERY_PREFIX}"
)

if [[ -n "${NUM_RUNS}" ]]; then
    CMD+=(--num_runs "${NUM_RUNS}")
fi

if [[ -n "${NUM_INSTANCES}" ]]; then
    CMD+=(--num_instances "${NUM_INSTANCES}")
fi

if [[ "${TOK}" -eq 1 ]]; then
    CMD+=(--tok True)
fi

printf 'Running:'
printf '\n'

"${CMD[@]}"
