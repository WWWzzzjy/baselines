#!/bin/bash
# eval.sh - Evaluate LocAgent localization results

set -e

WORKSPACE="/Users/Zhuanz/Documents/pyproject/LocAgent"
N_RUNS=3
DATASET="czlll/SWE-bench_Lite"
SPLIT="test"
DATASET_NAME="swe_bench_lite"
cd "${WORKSPACE}"
python - <<EOF
import sys
import numpy as np
import pandas as pd
sys.path.append('${WORKSPACE}')

from evaluation.eval_metric import evaluate_results, calc_token_usage

level2key_dict = {
    'file': 'found_files',
    'module': 'found_modules',
    'function': 'found_entities',
}

loc_files = [f'${WORKSPACE}/outputs/${DATASET_NAME}/run{i}/location/merged_loc_outputs_mrr.jsonl' for i in range(1, $N_RUNS + 1)]
traj_files = [f'${WORKSPACE}/outputs/${DATASET_NAME}/run{i}/location/loc_trajs.jsonl' for i in range(1, $N_RUNS + 1)]

all_results = []
all_tokens = []
all_times = []

for i, (loc_file, traj_file) in enumerate(zip(loc_files, traj_files), 1):
    print(f"=== Run {i} ===")

    # 评估指标
    res = evaluate_results(
        loc_file,
        level2key_dict,
        metrics=['acc', 'f1'],
        k_values_list=[
            [5],   # file level
            [5],   # module level
            [5],   # function level
        ],
        dataset='$DATASET',
        split='$SPLIT',
    )
    print(res)
    all_results.append(res)

    # token 和耗时
    token_usage = calc_token_usage(traj_file, model_name_or_path="Qwen/Qwen2.5-7B-Instruct")
    avg_tokens = sum(s['tokens'] for s in token_usage.values()) / len(token_usage)
    avg_time = sum(s['time'] for s in token_usage.values()) / len(token_usage)
    print(f"平均 token 数: {avg_tokens:.1f}")
    print(f"平均耗时: {avg_time:.1f}s")
    all_tokens.append(avg_tokens)
    all_times.append(avg_time)

# 均值和标准差
print("\n=== Summary ===")
stacked = np.stack([r.values for r in all_results])  # (N_RUNS, n_metrics)
mean_df = pd.DataFrame(
    stacked.mean(axis=0),
    index=all_results[0].index,
    columns=all_results[0].columns
)
std_df = pd.DataFrame(
    stacked.std(axis=0),
    index=all_results[0].index,
    columns=all_results[0].columns
)
print("Mean:")
print(mean_df)
print("\nStd:")
print(std_df)
print(f"\n平均 token 数: mean={np.mean(all_tokens):.1f}, std={np.std(all_tokens):.1f}")
print(f"平均耗时:    mean={np.mean(all_times):.1f}s, std={np.std(all_times):.1f}s")

EOF