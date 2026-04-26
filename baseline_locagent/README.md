# LocAgent Lite Eval

这份说明只覆盖当前最简单的运行方式：使用已经构建好的 BM25 和 graph 索引，直接跑 SWE-bench Lite。

## 环境

```bash
conda activate locagent
cd /Users/Zhuanz/Documents/pyproject/baselines/baseline_locagent
```

## Lite 运行命令

把 `GRAPH_INDEX_DIR` 和 `BM25_INDEX_DIR` 换成你本地已经构建好的 Lite 索引目录：

```bash
DATASET_NAME=princeton-nlp/SWE-bench_Lite \
DATASET_SUFFIX=SWE-bench_Lite \
GRAPH_INDEX_DIR=/path/to/lite/graph \
BM25_INDEX_DIR=/path/to/lite/bm25 \
LOCAGENT_MODEL=qwen3-32b \
NUM_RUNS=3 \
bash scripts/run_eval.sh results/lite_eval
```

如果只想快速试几条：

```bash
DATASET_NAME=princeton-nlp/SWE-bench_Lite \
DATASET_SUFFIX=SWE-bench_Lite \
GRAPH_INDEX_DIR=/path/to/lite/graph \
BM25_INDEX_DIR=/path/to/lite/bm25 \
LOCAGENT_MODEL=qwen3-32b \
EVAL_N_LIMIT=3 \
NUM_RUNS=1 \
bash scripts/run_eval.sh results/lite_smoke
```

## 参数说明

- `DATASET_NAME`：Hugging Face 数据集名。Lite 使用 `princeton-nlp/SWE-bench_Lite`。
- `DATASET_SUFFIX`：数据集后缀，只用于标识当前配置；Lite 使用 `SWE-bench_Lite`。
- `GRAPH_INDEX_DIR`：已经构建好的 graph 索引目录。脚本不会假设目录名，传什么就用什么。
- `BM25_INDEX_DIR`：已经构建好的 BM25 索引目录。脚本不会重新构建索引。
- `LOCAGENT_MODEL`：调用的大模型名称，例如 `qwen3-32b`。
- `NUM_RUNS`：重复运行次数。多轮结果会汇总到最终的 `verified_metrics.json`。
- `EVAL_N_LIMIT`：只跑前 N 条样本；不设置时跑完整 split。
- `INSTANCE_IDS_FILE`：可选，只跑文件中列出的 instance id。
- `RERUN_EMPTY_LOCATION=1`：可选，重新跑定位结果为空的样本。
- `results/lite_eval`：输出目录，也就是 `run_eval.sh` 的第一个参数。

## 输出位置

如果 `NUM_RUNS=1`：

```text
results/lite_eval/location/
```

如果 `NUM_RUNS>1`：

```text
results/lite_eval/run_1/location/
results/lite_eval/run_2/location/
results/lite_eval/run_3/location/
results/lite_eval/verified_metrics.json
```

常用文件：

- `localize.stdout.log`：localization 的终端输出日志。
- `localize.log`：LocAgent 运行日志。
- `merged_loc_outputs_mrr.jsonl`：用于评估的定位结果。
- `loc_trajs.jsonl`：模型交互轨迹、token 和推理耗时。
- `verified_metrics.json`：评估指标。

## 当前指标

当前评估脚本输出：

- `files_acc@5`
- `files_f1@5`
- `func_acc@5`
- `func_f1@5`
- `#tokens`
- `localize_time`
- `time`

多轮运行时，标准差按 instance 计算：同一个 instance 先在多轮之间取平均，再在所有 instance 的平均值之间计算标准差。
