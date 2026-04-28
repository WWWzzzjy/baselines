#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

dataset="lite" # "lite"/"v1"
result_dir="swe-bench-lite" # "swe-bench-lite" / "loc-bench-v1"
model_tag="qwen3.5-flash-2026-02-23"
run_nums=3

out_file="results/${result_dir}/multi_run_metrics_${model_tag}.txt"
mkdir -p "results/${result_dir}"

python - <<PY | tee "${out_file}"
from pathlib import Path
from statistics import mean, pstdev

from evaluation.FLEvalNew import evaluate_accuracy, load_json, load_jsonl

dataset = "${dataset}"
result_dir = "${result_dir}"
model_tag = "${model_tag}"
run_nums = ${run_nums}

gt_file = "evaluation/gt.json" if dataset == "lite" else "evaluation/gt_v1.json"
gt_data = load_json(gt_file)

metrics = {
    "file_ACC@5": [],
    "file_F1@5": [],
    "func_ACC@5": [],
    "func_F1@5": [],
    "avg_tokens": [],
    "avg_time": [],
}

for run_id in range(1, run_nums + 1):
    loc_file = Path(
        f"results/{result_dir}/run{run_id}/func_level_{model_tag}/loc_{model_tag}_func.jsonl"
    )
    if not loc_file.exists():
        print(f"run{run_id}: missing {loc_file}")
        continue

    result = evaluate_accuracy(load_jsonl(loc_file), gt_data, verbose=False)
    row = {
        "file_ACC@5": result["file_level"]["ACC@5"],
        "file_F1@5": result["file_level"]["F1@5"],
        "func_ACC@5": result["function_level"]["ACC@5"],
        "func_F1@5": result["function_level"]["F1@5"],
        "avg_tokens": result["tokens"]["average"],
        "avg_time": result["inference_time"]["average"],
    }
    for key, value in row.items():
        metrics[key].append(value)

    print(
        f"run{run_id}: "
        f"file_ACC@5={row['file_ACC@5']:.2f}% "
        f"file_F1@5={row['file_F1@5']:.2f}% "
        f"func_ACC@5={row['func_ACC@5']:.2f}% "
        f"func_F1@5={row['func_F1@5']:.2f}% "
        f"avg_tokens={row['avg_tokens']:.2f} "
        f"avg_time={row['avg_time']:.2f}s"
    )

print("\\nmean / std:")
for key, values in metrics.items():
    if not values:
        continue
    suffix = "%" if "ACC" in key or "F1" in key else ("s" if "time" in key else "")
    print(f"{key}: mean={mean(values):.2f}{suffix}, std={pstdev(values):.2f}{suffix}")
PY
