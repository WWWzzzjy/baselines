#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-./output_local}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./artifact}"
LOCALIZATION_DATASET="${LOCALIZATION_DATASET:-lite}"  # "loc-bench-v1" / "lite"
LOCALIZATION_SPLIT="${LOCALIZATION_SPLIT:-test}"
FILE_PATH_KEY="${FILE_PATH_KEY:-file_path}"
NUM_RUNS="${NUM_RUNS:-3}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Cannot find OUTPUT_DIR: $OUTPUT_DIR" >&2
  echo "Run ./run.sh first, or set OUTPUT_DIR=/path/to/output" >&2
  exit 2
fi

echo "Evaluating OrcaLoca localization"
echo "  output dir: $OUTPUT_DIR"
echo "  dataset:    $LOCALIZATION_DATASET"
echo "  split:      $LOCALIZATION_SPLIT"
echo "  artifact:   $ARTIFACT_DIR"
echo "  num runs:   $NUM_RUNS"

RUN_DIRS=()
if [[ "$NUM_RUNS" != "0" ]]; then
  if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -lt 1 ]]; then
    echo "NUM_RUNS must be 0 or a positive integer, got: $NUM_RUNS" >&2
    exit 2
  fi
  if [[ "$NUM_RUNS" -eq 1 ]]; then
    RUN_DIRS=("$OUTPUT_DIR")
  else
    for run_idx in $(seq 1 "$NUM_RUNS"); do
      RUN_DIRS+=("$OUTPUT_DIR/run_${run_idx}")
    done
  fi
elif find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'run_*' | grep -q .; then
  while IFS= read -r run_dir; do
    RUN_DIRS+=("$run_dir")
  done < <(
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'run_*' | python -c '
import os
import re
import sys

def key(path):
    match = re.search(r"run_(\d+)$", os.path.basename(path))
    return int(match.group(1)) if match else path

for path in sorted((line.strip() for line in sys.stdin if line.strip()), key=key):
    print(path)
'
  )
else
  RUN_DIRS=("$OUTPUT_DIR")
fi

METRIC_FILES=()
for run_dir in "${RUN_DIRS[@]}"; do
  if [[ ! -d "$run_dir" ]]; then
    echo "Cannot find run output dir: $run_dir" >&2
    exit 2
  fi

  metrics_path="$run_dir/eval_metrics.json"
  METRIC_FILES+=("$metrics_path")
  echo
  echo "---- Evaluating $run_dir ----"
  python artifact/parse_output.py \
    --artifact_dir "$ARTIFACT_DIR" \
    --output_dir "$run_dir" \
    --file_path_key "$FILE_PATH_KEY" \
    --dataset "$LOCALIZATION_DATASET" \
    --split "$LOCALIZATION_SPLIT" \
    --metrics_output "$metrics_path" \
    --no_std_output
done

if [[ "${#METRIC_FILES[@]}" -gt 1 ]]; then
  summary_path="$OUTPUT_DIR/eval_metrics_summary.json"
  echo
  echo "==== Multi-run summary ===="
  python - "$summary_path" "${METRIC_FILES[@]}" <<'PY'
import json
import math
import sys

summary_path = sys.argv[1]
metric_paths = sys.argv[2:]
metrics = []
for path in metric_paths:
    with open(path) as f:
        data = json.load(f)
    data["_path"] = path
    metrics.append(data)

percentage_keys = [
    ("file_match_rate", "File Match"),
    ("mean_file_acc_at_5", "File Acc@5"),
    ("mean_file_f1_at_5", "File F1@5"),
    ("mean_file_precision", "File Precision"),
    ("func_match_rate", "Function Match"),
    ("mean_func_acc_at_5", "Function Acc@5"),
    ("mean_func_f1_at_5", "Function F1@5"),
    ("mean_func_precision", "Function Precision"),
    ("json_not_gen_rate", "Json Not Gen"),
]
numeric_keys = [
    ("mean_total_tokens", "Total Tokens"),
    ("mean_input_tokens", "Input Tokens"),
    ("mean_output_tokens", "Output Tokens"),
    ("mean_wall_time_s", "Wall Time (s)"),
]

def mean(values):
    return sum(values) / len(values) if values else 0.0

def std(values):
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))

summary = {"num_runs": len(metrics), "runs": metric_paths, "metrics": {}}

for key, label in percentage_keys:
    values = [m[key] for m in metrics if key in m]
    if not values:
        continue
    avg = mean(values)
    sd = std(values)
    summary["metrics"][key] = {"mean": avg, "std": sd}
    print(f"{label:<20}: mean {avg * 100:7.2f}%, std {sd * 100:7.2f}%")

for key, label in numeric_keys:
    values = [m[key] for m in metrics if key in m]
    if not values:
        continue
    avg = mean(values)
    sd = std(values)
    summary["metrics"][key] = {"mean": avg, "std": sd}
    print(f"{label:<20}: mean {avg:10.2f}, std {sd:10.2f}")

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"Summary dumped to {summary_path}")
PY
fi

echo
echo "Detailed parsed results:"
echo "  $ARTIFACT_DIR/assets/orcar_parsed_output.json"
if [[ "${#METRIC_FILES[@]}" -gt 1 ]]; then
  echo "Multi-run summary:"
  echo "  $OUTPUT_DIR/eval_metrics_summary.json"
fi
