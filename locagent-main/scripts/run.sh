# set api key
# or set api key in `scripts/env/set_env.sh`
# . scripts/env/set_env.sh
export OPENAI_API_KEY="sk-482fa2a1567041ecafa9c3114bf3811d"
export OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

export PYTHONPATH=$PYTHONPATH:$(pwd)
export GRAPH_INDEX_DIR="index_data/SWE-bench_Lite/graph_index_v2.3"
export BM25_INDEX_DIR="index_data/SWE-bench_Lite/BM25_index"

N_RUNS=3

for RUN_ID in $(seq 1 $N_RUNS); do
    result_path="outputs/run${RUN_ID}"
    echo "=== Run ${RUN_ID}, output to: $result_path ==="
    mkdir -p $result_path/location

    python auto_search_main.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --split 'test' \
        --model 'openai/qwen2.5-7b-instruct' \
        --localize \
        --merge \
        --output_folder $result_path/location \
        --eval_n_limit 3 \
        --num_processes 1 \
        --use_function_calling \
        --simple_desc \
        --temperature 0.6 \
        --top_p 0.95 \
        --max_tokens 8192 

done