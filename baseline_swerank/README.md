# SweRank+ Baseline Eval 说明

这份说明对应当前仓库里已经跑通的 baseline eval 流程。

目标：

- 使用 `SweRankEmbed-Small` 跑 retriever
- 使用阿里云 DashScope 兼容 OpenAI API 的大模型做 rerank
- 输出以下指标：
  - `Func Acc@3`
  - `Func F1@3`
  - `Files Acc@5`
  - `Files F1@5`
  - `retriever_time_seconds`
  - `rerank_time_seconds`
  - `total_time_seconds`
  - token 统计
- 支持多次运行并在 `eval_results` 中计算方差

## 1. 环境

要求在 `swerank` 虚拟环境中运行。

示例：

```bash
conda activate swerank
cd SweRank
```

如果环境里缺包，至少需要确保这些依赖可用：

- `datasets`
- `beir`
- `sentence_transformers`
- `openai`
- `einops`
- `torch`

## 2. 数据

当前默认流程面向：

- `SWE-bench Lite`

也支持显式指定：

- `swe-bench-lite`
- `swe-bench-verified`

脚本使用的数据目录是：

```bash
./datasets
```

仓库现在支持在 `datasets/` 不存在或实例数不足时，自动构建最小可跑的本地 BEIR 子集。

说明：

- 如果本地已有足够的 `datasets/<dataset>-function_*` 目录，脚本会直接使用本地数据，不会重新构建语料。
- 如果本地数据不存在或数量不足，脚本会自动构建缺失语料；构建语料时需要从 GitHub 获取目标 repo。

当前本地子集目录命名格式如下：

```bash
datasets/swe-bench-lite-function_<instance_id>
```

每个实例目录下包含：

- `corpus.jsonl`
- `queries.jsonl`
- `qrels/test.tsv`
- `metadata.json`

## 3. 运行脚本

主入口脚本：

[script/run_baseline_eval_dashscope.sh](/Users/Zhuanz/Documents/pyproject/baselines/baseline_swerank/script/run_baseline_eval_dashscope.sh)

最常用命令：

```bash
bash script/run_baseline_eval_dashscope.sh
```

含义：

- `./datasets`：默认数据目录
- `swe-bench-lite`：默认数据集名

## 4. 默认配置

### Retriever

- `RETRIEVER_MODEL_NAME=Salesforce/SweRankEmbed-Small`
- `RETRIEVER_MODEL_TAG=SweRankEmbed-Small`
- `RETRIEVER_BATCH_SIZE=16`
- `RETRIEVER_SEQUENCE_LENGTH=1024`

### Reranker API

- `OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `OPENAI_API_KEY=<你的 key>`
- `LOCAGENT_MODEL=qwen3-coder-plus`
- `RERANK_TAG=rerank-small`

### LLM 采样参数

- `TEMPERATURE=0.6`
- `TOP_P=0.95`
- `MAX_TOKENS=32768`

### Rerank 参数

- `TOP_K=100`
- `WINDOW_SIZE=10`
- `STEP_SIZE=5`

### 其他

- `NUM_INSTANCES=1`
- `NUM_RUNS=3`

说明：

- `NUM_INSTANCES=0` 表示使用当前数据目录下所有可用 instance
- `NUM_INSTANCES>0` 表示只取前 N 条可用 instance

## 5. 日志

日志文件默认写到：

[logs/baseline_eval_dashscope.log](/Users/Zhuanz/Documents/pyproject/swerank/SweRank/logs/baseline_eval_dashscope.log)

实时查看：

```bash
tail -f logs/baseline_eval_dashscope.log
```

## 6. 输出文件说明

### `outputs/`

这里放中间结果。

#### Retriever 结果

- `outputs/retriever_results.json`
  - retriever 主结果，后续 rerank 直接读取它
- `outputs/retriever_results_per_instance.json`
  - 和上面内容相同，但命名更直观
- `outputs/retriever_results_checkpoint.jsonl`
  - 逐条 checkpoint，用于中途中断后恢复

#### Retriever 评测

- `outputs/retriever_eval_summary.json`
  - retriever 平均评测结果
- `outputs/retriever_eval_per_instance.json`
  - 每条 instance 的 retriever 评测结果
- `outputs/retriever_eval_checkpoint.jsonl`
  - 逐条 checkpoint

#### Rerank 输入

- `outputs/retriever/<instance_dir>/<rerank_tag>-runN_rank_100.json`

这部分是从 retriever 结果转换来的 rerank 输入，不是最终结果。

#### Rerank 输出

- `outputs/rerank-small-run1/<instance_dir>/rerank_100_llm_gen_num.json`
- `outputs/rerank-small-run1/<instance_dir>/rerank_100_llm_gen_num_histories.json`
- `outputs/rerank-small-run2/...`

其中：

- `rerank_100_llm_gen_num.json`
  - rerank 后的最终排序结果
- `rerank_100_llm_gen_num_histories.json`
  - 每次 LLM 调用的 prompt/response/usage/latency

### `eval_results/`

这里放最终评测汇总。

主文件：

- `eval_results/dashscope-baseline-SweRankEmbed-Small_swe-bench-lite_baseline_eval.json`

结构：

- `config`
  - 本次运行配置
- `runs`
  - 每次 run 的单独结果
- `summary`
  - 多次运行的均值和方差

## 7. 指标说明

当前最终关注的指标是：

- `func_acc_at_3`
- `func_f1_at_3`
- `files_acc_at_5`
- `files_f1_at_5`
- `retriever_time_seconds`
- `rerank_time_seconds`
- `total_time_seconds`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

说明：

- `total_time_seconds = retriever_time_seconds + rerank_time_seconds`
- 当前方差是“多次运行之间”的方差，不是一次运行内样本之间的方差

## 8. 可恢复性

当前 retriever 已经改成：

- 逐条 instance 落盘
- 支持 checkpoint
- 再次启动时会跳过已完成 instance

因此如果中途中断，不需要从 0 开始。

## 9. 关于方差

只有当：

```bash
NUM_RUNS >= 2
```

时，`summary.variance` 才有意义。

如果只跑 1 次，方差会是 `0.0`，这不是 bug，而是因为没有跨 run 的波动可计算。

## 10. 当前这份实现的定位

这份仓库当前跑通的是：

- `retrieve + rerank` baseline

不是论文中完整的多轮 agent 系统。

也就是说当前流程是：

1. retriever 召回候选函数
2. reranker 用 LLM 对候选函数做 listwise 重排
3. 输出定位指标

## 11. 建议使用方式

如果只是验证流程是否通，建议先跑：

```bash
bash script/run_baseline_eval_dashscope.sh ./datasets swe-bench-lite 1 1
```

如果要验证方差逻辑，建议跑：

```bash
bash script/run_baseline_eval_dashscope.sh ./datasets swe-bench-lite 3 2
```

如果要正式按要求产出方差，建议跑：

```bash
bash script/run_baseline_eval_dashscope.sh ./datasets swe-bench-lite 0 3
```
