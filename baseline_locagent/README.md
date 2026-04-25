# LocAgent Baseline Eval 说明

这份说明对应当前仓库里已经跑通的 LocAgent baseline eval 流程。

目标：

- 使用 LocAgent 对 `SWE-bench Verified` 做 File / Function 定位
- 使用阿里云 DashScope 兼容 OpenAI API 的大模型
- 输出以下指标：
  - `func_acc@3`
  - `func_f1`
  - `files_acc@5`
  - `files_f1`
  - `#tokens`
  - `time`
- 支持多次运行，并在最终 `verified_metrics.json` 中计算 run 之间的方差

## 1. 环境

要求在 `locagent` 虚拟环境中运行。

示例：

```bash
conda activate locagent
cd baseline_locagent
```

如果需要重新安装环境：

```bash
conda create -n locagent python=3.12
conda activate locagent
pip install -r requirements.txt
```

## 2. 数据

当前流程面向：

- `princeton-nlp/SWE-bench_Verified`
- split: `test`

脚本会从 Hugging Face 读取数据集，并根据 instance 自动准备目标 repo、graph index 和 BM25 index。

## 3. 运行脚本

主入口脚本：

[scripts/run_verified.sh](/Users/Zhuanz/Documents/pyproject/baselines/baseline_locagent/scripts/run_verified.sh)

最常用命令：

```bash
bash scripts/run_verified.sh
```

只跑前 3 条做 smoke test：

```bash
EVAL_N_LIMIT=3 bash scripts/run_verified.sh
```

跑 2 轮并计算 run 间方差：

```bash
EVAL_N_LIMIT=3 NUM_RUNS=2 bash scripts/run_verified.sh
```

限制单条样本的 LLM / tool 交互轮数：

```bash
MAX_ITERATIONS=8 EVAL_N_LIMIT=3 NUM_RUNS=2 bash scripts/run_verified.sh
```

## 4. 默认配置

### API

- `OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `OPENAI_API_KEY=<你的 key>`
- `LOCAGENT_MODEL=qwen3-coder-plus`

### LLM 采样参数

- `LOCAGENT_TEMPERATURE=0.6`
- `LOCAGENT_TOP_P=0.95`
- `LOCAGENT_MAX_TOKENS=32768`

### 运行参数

- `NUM_RUNS=3`
- `NUM_PROCESSES=1`
- `NUM_SAMPLES=1`
- `MAX_ITERATIONS=20`
- `LOCAGENT_GIT_CLONE_TIMEOUT=600`
- `PREBUILD_INDICES=1`
- `PREBUILD_NUM_PROCESSES=1`

默认流程：

1. 预建 graph index
2. 预建 BM25 index
3. 运行 LocAgent localization
4. merge 定位结果
5. 计算评测指标

## 5. 日志

预建阶段日志：

[results/verified_full_qwen3coder/prebuild.log](/Users/Zhuanz/Documents/pyproject/baselines/baseline_locagent/results/verified_full_qwen3coder/prebuild.log)

实时查看：

```bash
tail -f results/verified_full_qwen3coder/prebuild.log
```

localization 阶段日志：

```bash
tail -f results/verified_full_qwen3coder/location/localize.log
```

如果 `NUM_RUNS > 1`：

```bash
tail -f results/verified_full_qwen3coder/run_1/location/localize.log
tail -f results/verified_full_qwen3coder/run_2/location/localize.log
```

## 6. 输出文件说明

默认输出根目录：

```bash
results/verified_full_qwen3coder/
```

### 根目录

- `prebuild.log`
  - graph / BM25 预建日志
- `prebuild_times.json`
  - graph / BM25 离线预建耗时
- `instance_ids.eval_n_limit.json`
  - 设置 `EVAL_N_LIMIT` 时自动生成的 instance 列表
- `verified_metrics.json`
  - 多轮运行时的最终汇总结果

### 单轮输出

如果 `NUM_RUNS=1`，结果在：

```bash
results/verified_full_qwen3coder/location/
```

如果 `NUM_RUNS>1`，每一轮结果在：

```bash
results/verified_full_qwen3coder/run_1/location/
results/verified_full_qwen3coder/run_2/location/
```

每个 `location/` 下包含：

- `args.json`
  - 本轮运行参数
- `localize.log`
  - 本轮 localization 日志
- `loc_outputs.jsonl`
  - 每条 instance 的定位结果
- `loc_trajs.jsonl`
  - 每条 instance 的对话轨迹、usage、token 和耗时
- `merged_loc_outputs_mrr.jsonl`
  - merge 后用于评测的定位结果
- `verified_metrics.json`
  - 本轮评测结果

## 7. 指标说明

单轮 `verified_metrics.json` 结构：

- `per_instance`
  - 每条 instance 的指标
- `aggregate`
  - 本轮成功样本的平均指标

单轮 `aggregate` 不计算方差，避免和 run 间方差混淆。

多轮根目录 `verified_metrics.json` 结构：

- `runs`
  - 每一轮的平均指标
- `aggregate`
  - 对各轮平均指标再计算 `mean` 和 `std`

当前最终关注的指标是：

- `func_acc@3`
- `func_f1`
- `files_acc@5`
- `files_f1`
- `#tokens`
- `localize_time`
- `prebuild_time`
- `time`

说明：

- `time = localize_time + prebuild_time`
- `prebuild_time` 是 graph / BM25 离线预建耗时按 instance 均摊后的结果
- 当前方差是“多次运行之间”的方差，不是一次运行内样本之间的方差
- 只有当 `found_files`、`found_modules`、`found_entities`、`raw_output_loc` 全空时，样本才标记为 `success=false`
- 单轮 `aggregate` 只统计 `success=true` 的样本

## 8. 可恢复性

当前 localization 会逐条写入：

- `loc_outputs.jsonl`
- `loc_trajs.jsonl`

如果中途中断并用同一个输出目录重跑：

- 已写入 `loc_outputs.jsonl` 的 instance 会被跳过
- 中断时正在处理的 instance 可能会重新跑
- merge 和 evaluation 可以重新执行

## 9. 建议使用方式

验证流程是否跑通：

```bash
EVAL_N_LIMIT=3 MAX_ITERATIONS=8 bash scripts/run_verified.sh
```

验证 run 间方差逻辑：

```bash
EVAL_N_LIMIT=3 NUM_RUNS=2 MAX_ITERATIONS=8 bash scripts/run_verified.sh
```

正式跑完整 Verified：

```bash
NUM_RUNS=3 bash scripts/run_verified.sh
```
