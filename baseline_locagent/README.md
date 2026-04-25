# LocAgent Baseline For SWE-bench Verified

This repository is set up to run the `LocAgent` baseline on `princeton-nlp/SWE-bench_Verified` with an OpenAI-compatible API endpoint.

The main runnable entrypoint is:

```bash
bash scripts/run_verified.sh
```

## Setup

```bash
git clone <your-repo-url>
cd baseline_locagent

conda create -n locagent python=3.11
conda activate locagent
pip install -r requirements.txt
```

## API Configuration

Default API settings are defined in `scripts/run_verified.sh`. You can edit that file directly, or override values at runtime:

```bash
OPENAI_API_KEY=your_key OPENAI_API_BASE=your_base LOCAGENT_MODEL=your_model bash scripts/run_verified.sh
```

Default variables used by the script:

- `OPENAI_API_BASE`
- `OPENAI_API_KEY`
- `LOCAGENT_MODEL`
- `LOCAGENT_TEMPERATURE`
- `LOCAGENT_TOP_P`
- `LOCAGENT_MAX_TOKENS`

## Run

After activating `conda locagent`, run:

```bash
bash scripts/run_verified.sh
```

Default behavior:

- dataset: `princeton-nlp/SWE-bench_Verified`
- split: `test`
- model: `qwen3-coder-plus`
- output dir: `results/verified_full_qwen3coder`
- localization workers: `NUM_PROCESSES=1`
- sample count per issue: `NUM_SAMPLES=1`
- max LLM/tool rounds per attempt: `MAX_ITERATIONS=20`
- git clone timeout: `LOCAGENT_GIT_CLONE_TIMEOUT=600`

Most importantly, the script now defaults to:

1. prebuild graph indexes
2. prebuild BM25 indexes
3. run localization
4. merge outputs
5. run evaluation

This avoids the unstable path where localization triggers repo cloning and BM25 construction in the middle of agent search.

During prebuild, the script prints progress lines such as `[graph] progress 2/300` and `[bm25] progress 2/300`. During clone, it prints periodic size updates, so a quiet long-running step is easier to distinguish from a stuck process.

## Useful Overrides

Run only the first 3 verified instances:

```bash
EVAL_N_LIMIT=3 bash scripts/run_verified.sh
```

Run 2 repeated rounds and compute run-level variance:

```bash
NUM_RUNS=2 bash scripts/run_verified.sh
```

Limit LLM/tool interaction rounds for faster smoke tests:

```bash
MAX_ITERATIONS=8 bash scripts/run_verified.sh
```

Run a custom instance subset from a file:

```bash
INSTANCE_IDS_FILE=path/to/instance_ids.txt bash scripts/run_verified.sh
```

Use a different output directory:

```bash
bash scripts/run_verified.sh results/my_verified_run
```

Disable prebuild if you explicitly want to test runtime indexing:

```bash
PREBUILD_INDICES=0 bash scripts/run_verified.sh
```

Change prebuild worker count:

```bash
PREBUILD_NUM_PROCESSES=1 bash scripts/run_verified.sh
```

Change localization worker count:

```bash
NUM_PROCESSES=1 bash scripts/run_verified.sh
```

Change clone timeout:

```bash
LOCAGENT_GIT_CLONE_TIMEOUT=1200 bash scripts/run_verified.sh
```

## Output Files

Results are written under:

```bash
results/verified_full_qwen3coder/location/
```

Important files:

- `args.json`: run arguments
- `localize.log`: localization log
- `prebuild_times.json`: offline graph and BM25 prebuild time
- `loc_outputs.jsonl`: per-instance localization outputs
- `loc_trajs.jsonl`: per-instance trajectory and token/time usage for successful outputs
- `merged_loc_outputs_mrr.jsonl`: merged localization outputs
- `verified_metrics.json`: final metrics

## Evaluation Metrics

The evaluation script is:

```bash
evaluation/eval_verified_baseline.py
```

It reports:

- `func_acc@3`
- `func_f1`
- `files_acc@5`
- `files_f1`
- `#tokens`
- `time`

`time` includes localization time plus the offline graph/BM25 prebuild time distributed over the evaluated instances. The metrics file also keeps `localize_time` and `prebuild_time` separately.

Current evaluation behavior:

- failed localization samples are kept in `per_instance`
- failed samples are marked with `success: false`
- failed samples can show `#tokens = 0` and `time = 0`
- aggregate metrics only include `success: true` samples

## Rerun Evaluation Only

If localization already finished and you only want to recompute metrics:

```bash
cd baseline_locagent
PYTHONPATH=$(pwd) python evaluation/eval_verified_baseline.py \
  --loc_file results/verified_full_qwen3coder/location/merged_loc_outputs_mrr.jsonl \
  --traj_file results/verified_full_qwen3coder/location/loc_trajs.jsonl \
  --dataset princeton-nlp/SWE-bench_Verified \
  --split test \
  --output_file results/verified_full_qwen3coder/location/verified_metrics.json
```

## Resume Behavior

If a run is interrupted and you restart with the same output directory:

- completed instances already written to `loc_outputs.jsonl` are skipped
- the currently interrupted instance may rerun
- merge and evaluation can still be rerun afterward
