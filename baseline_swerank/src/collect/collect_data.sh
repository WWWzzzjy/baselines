#!/bin/bash

set -e

export REPO_DIR=$PWD
export DATASET_DIR="${REPO_DIR}/datasets"

# Files to save a list of PyPI packages and valid repos
pypi_list_file="pypi_rankings.jsonl"
valid_repo_out_path="valid_top_pypi_gitrepos.jsonl"

# Path to save PRs and tasks from valid repos
prs_path=prs
tasks_path=tasks

# Max number of repos to scrape and process. Increase this number to scrape more repos.
max_num_repos=10

# Scrape PyPI packages
python get_top_pypi.py \
    --max_repos ${max_num_repos} \
    --save_file_name ${pypi_list_file}

# Retrieve source GitHub repos of the scraped PyPI packagesslack
python get_valid_urls.py \
    --pypi_list_path ${pypi_list_file} \
    --save_file_name ${valid_repo_out_path}

# Scrape PRs and create tasks
python generate_tasks.py \
    --repo_list_path ${valid_repo_out_path} \
    --prs_path ${prs_path} \
    --tasks_path ${tasks_path} \
    --start_idx 0 \
    --num_repos_to_process ${max_num_repos}

# Build dataset with scraped PRs
python build_dataset_ft.py \
    --instances_path ${tasks_path} \
    --output_dir ${DATASET_DIR} 
