#!/bin/bash

set -e

export REPO_DIR=$PWD

OUT_PATH=${REPO_DIR}/datasets
filtered_dataset="repo_contrastive_mined_filtered.jsonl"

echo "Constructing reranker training data..."

WINDOW_SIZE=10
DATA_FILE_NAME="repo_contrastive_mined_filtered.jsonl"
echo "Filtered dataset path: ${filtered_dataset}"
python reranker/get_rerank_train_data.py \
    --data_type local \
    --data_path ${filtered_dataset} \
    --out_path ${OUT_PATH} \
    --window_size ${WINDOW_SIZE} \
    --random_seed 42 \
    --eval_ratio 0.01 \
    --shuffle_context \
    --first_identifier_only \
    --objective sft