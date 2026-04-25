#!/bin/bash

set -e

PR_DATASET="collect/datasets/SWE_PRS_FT_DATASET.jsonl"
premined_data_name="repo_contrastive_premined.jsonl"
mined_data_name="repo_contrastive_mined.jsonl"

# Extract functions from the corresponding codebases
python get_train_by_repo.py \
    --data_path $PR_DATASET \
    --save_file_name $premined_data_name

# Negative mining
python repo_negative_mining.py \
    --data_path ${premined_data_name} \
    --num_workers_per_gpu 1 \
    --save_file_name ${mined_data_name}

# Quality filtering (consistency + hard negatives filtering)
python collect/filter_dataset.py \
    --file_path ${mined_data_name} \