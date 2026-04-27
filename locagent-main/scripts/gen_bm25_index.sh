export PYTHONPATH=$PYTHONPATH:$(pwd)
N_LIMIT=${N_LIMIT:-3}

python build_bm25_index.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --split 'test' \
        --repo_path playground/build_graph \
        --num_processes 1 \
        --eval_n_limit "$N_LIMIT" \
        --download_repo
