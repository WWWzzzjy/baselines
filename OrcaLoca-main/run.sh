#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-qwen3-8b}"
DATASET="${DATASET:-czlll/Loc-Bench_V1}"  # "czlll/Loc-Bench_V1" / "princeton-nlp/SWE-bench_Lite"
SPLIT="${SPLIT:-test}"
CFG_PATH="${CFG_PATH:-./key.cfg}"
RUN_ALL="${RUN_ALL:-1}" # 1:跑全部
OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_local}"
CACHE_DIR="${CACHE_DIR:-./repo_cache_v1}"
NUM_RUNS="${NUM_RUNS:-3}"

INSTANCE_IDS=("$@")

if [[ "${#INSTANCE_IDS[@]}" -eq 0 && "$RUN_ALL" != "1" ]]; then
  cat <<'EOF'
Usage:
  ./run.sh <instance_id> [instance_id ...]
  RUN_ALL=1 ./run.sh

Examples:
  ./run.sh astropy__astropy-12907
  ./run.sh django__django-11099 sympy__sympy-20590
  RUN_ALL=1 DATASET=princeton-nlp/SWE-bench_Lite ./run.sh

Environment overrides:
  MODEL, DATASET, SPLIT, CFG_PATH, OUTPUT_DIR, CACHE_DIR, NUM_RUNS
  OPENAI_API_KEY, OPENAI_API_BASE_URL, QWEN_TOKENIZER_MODEL
EOF
  exit 2
fi

if [[ "$RUN_ALL" == "1" ]]; then
  INSTANCE_IDS=()
  while IFS= read -r inst_id; do
    INSTANCE_IDS+=("$inst_id")
  done < <(
    DATASET="$DATASET" SPLIT="$SPLIT" python - <<'PY'
import argparse
import os

from Orcar.load_cache_dataset import load_filter_hf_dataset

args = argparse.Namespace(
    dataset=os.environ["DATASET"],
    split=os.environ["SPLIT"],
    filter_instance=".*",
)

for inst_id in load_filter_hf_dataset(args)["instance_id"]:
    print(inst_id)
PY
  )
fi

echo "Running OrcaLoca localization"
echo "  model:     $MODEL"
echo "  dataset:   $DATASET/$SPLIT"
echo "  instances: ${#INSTANCE_IDS[@]}"
echo "  output:    $OUTPUT_DIR"
echo "  num runs:  $NUM_RUNS"
if [[ -n "$OPENAI_API_BASE_URL" ]]; then
  export OPENAI_API_BASE_URL
  echo "  api base:  $OPENAI_API_BASE_URL"
fi

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -lt 1 ]]; then
  echo "NUM_RUNS must be a positive integer, got: $NUM_RUNS" >&2
  exit 2
fi

for run_idx in $(seq 1 "$NUM_RUNS"); do
  if [[ "$NUM_RUNS" -eq 1 ]]; then
    RUN_OUTPUT_DIR="$OUTPUT_DIR"
  else
    RUN_OUTPUT_DIR="$OUTPUT_DIR/run_${run_idx}"
  fi

  echo
  echo "==== Run ${run_idx}/${NUM_RUNS} ===="
  echo "  run output: $RUN_OUTPUT_DIR"

  python run_local.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --cfg_path "$CFG_PATH" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$RUN_OUTPUT_DIR" \
    --instance_ids "${INSTANCE_IDS[@]}"
done
