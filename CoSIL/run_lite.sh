set -e

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PROJECT_FILE_LOC="./repo_structures_lite"
export HF_ENDPOINT=https://hf-mirror.com
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY before running.}"


# Fault Localization
models=("qwen3.5-flash-2026-02-23")
model_names=("qwen3.5-flash-2026-02-23")
data_tap="swe-bench-lite" # "swe-bench-lite" / "loc-bench-v1"
backend=("openai")
threads=1
n_instance=3
run_nums=3
temperature=0.6
top_p=0.95
max_tokens=32768

for run_id in $(seq 1 ${run_nums}); do
  run_dir="results/${data_tap}/run${run_id}"

  for i in "${!models[@]}"; do
    python afl/fl/AFL_localize_file.py \
      --file_level \
      --output_folder "${run_dir}/file_level_${models[$i]}" \
      --num_threads ${threads} \
      --model "${model_names[$i]}" \
      --backend "${backend[$i]}" \
      --skip_existing \
      --temperature ${temperature} \
      --top_p ${top_p} \
      --max_tokens ${max_tokens} \
      --n_instances ${n_instance}
  done

  for i in "${!models[@]}"; do
    python afl/fl/AFL_localize_func.py \
      --output_folder "${run_dir}/func_level_${models[$i]}" \
      --loc_file "${run_dir}/file_level_${models[$i]}/loc_outputs.jsonl" \
      --output_file "loc_${models[$i]}_func.jsonl" \
      --temperature ${temperature} \
      --top_p ${top_p} \
      --max-tokens ${max_tokens} \
      --model "${model_names[$i]}" \
      --backend "${backend[$i]}" \
      --skip_existing \
      --num_threads ${threads} \
      --n_instances ${n_instance}
  done
done


