import json
import time
import traceback
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "qwen_token_per_instance_stats.json"
TOKENIZER_NAME = "Qwen/Qwen3-8B"
DATASETS = ["swe-bench-lite", "loc-bench"]
RUNS = [1, 2]


def log(message):
    print(message, flush=True)


def iter_entries(obj):
    if isinstance(obj, dict):
        if "prompt" in obj and "response" in obj:
            yield obj
        else:
            for value in obj.values():
                yield from iter_entries(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_entries(value)


def count_prompt(tokenizer, prompt):
    if isinstance(prompt, list):
        return len(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return len(tokenizer.encode(str(prompt), add_special_tokens=False))


def count_response(tokenizer, response):
    return len(tokenizer.encode(response or "", add_special_tokens=False))


def summarize_instance_values(values):
    return {
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
    }


def load_retriever_times(dataset):
    path = (
        ROOT
        / f"model=SweRankEmbed-Large_dataset={dataset}_split=test_level=function_evalmode=default_output.json"
    )
    if not path.exists():
        return 0.0, 0.0
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    avg_time = float(data.get("time", 0.0))

    all_path = Path(str(path) + "_all")
    if all_path.exists():
        with all_path.open(encoding="utf-8") as file:
            all_data = json.load(file)
        total_time = sum(float(item.get("time", 0.0)) for item in all_data)
    else:
        total_time = avg_time
    return avg_time, total_time


def load_instance_time(instance_dir):
    path = instance_dir / "time_metrics.json"
    if not path.exists():
        return 0.0
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return float(data.get("inference_time_secs", 0.0))


def main():
    log(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"root={ROOT}")
    log(f"tokenizer={TOKENIZER_NAME}")
    log(
        "granularity=per instance; each instance token is the sum of "
        "no-context prompt messages plus response tokens over all windows."
    )

    log("[import transformers]")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        local_files_only=True,
        trust_remote_code=True,
    )
    log("[loaded tokenizer]")

    run_rows = []

    for dataset in DATASETS:
        retriever_avg_time, retriever_total_time = load_retriever_times(dataset)
        for run_idx in RUNS:
            run_dir = ROOT / f"SweRankEmbed-Large_Qwen3-Coder-30B_run{run_idx}_{dataset}"
            files = sorted(run_dir.glob("*/rerank_100_llm_gen_num_histories.json"))
            log(f"\n[run] dataset={dataset} run={run_idx} files={len(files)}")

            instance_inputs = []
            instance_outputs = []
            instance_totals = []
            instance_windows = []
            instance_reranker_times = []

            run_start = time.time()
            for index, path in enumerate(files, 1):
                input_tokens = 0
                output_tokens = 0
                windows = 0
                reranker_time = load_instance_time(path.parent)

                with path.open(encoding="utf-8") as file:
                    data = json.load(file)

                for entry in iter_entries(data.get("run_history", [])):
                    windows += 1
                    input_tokens += count_prompt(tokenizer, entry.get("prompt"))
                    output_tokens += count_response(tokenizer, entry.get("response"))

                instance_inputs.append(input_tokens)
                instance_outputs.append(output_tokens)
                instance_totals.append(input_tokens + output_tokens)
                instance_windows.append(windows)
                instance_reranker_times.append(reranker_time)

                if index == 1 or index % 10 == 0 or index == len(files):
                    elapsed = time.time() - run_start
                    avg_per_file = elapsed / index
                    remaining = avg_per_file * (len(files) - index)
                    log(
                        f"[progress] {dataset} run{run_idx} {index}/{len(files)} "
                        f"elapsed={elapsed:.1f}s eta={remaining:.1f}s "
                        f"last={path.parent.name} last_total={input_tokens + output_tokens} "
                        f"last_windows={windows}"
                    )

            num_instances = len(instance_totals)
            reranker_total_time = sum(instance_reranker_times)
            reranker_avg_time = reranker_total_time / max(num_instances, 1)
            pipeline_total_time = retriever_total_time + reranker_total_time
            pipeline_avg_time = retriever_avg_time + reranker_avg_time

            row = {
                "dataset": dataset,
                "run": run_idx,
                "num_instances": num_instances,
                "windows_total": sum(instance_windows),
                "input_total": sum(instance_inputs),
                "output_total": sum(instance_outputs),
                "total_tokens": sum(instance_totals),
                "per_instance_input": summarize_instance_values(instance_inputs),
                "per_instance_output": summarize_instance_values(instance_outputs),
                "per_instance_total": summarize_instance_values(instance_totals),
                "windows_per_instance": summarize_instance_values(instance_windows),
                "time": {
                    "retriever_avg_secs_per_instance": retriever_avg_time,
                    "retriever_total_secs": retriever_total_time,
                    "reranker_avg_secs_per_instance": reranker_avg_time,
                    "reranker_total_secs": reranker_total_time,
                    "pipeline_avg_secs_per_instance": pipeline_avg_time,
                    "pipeline_total_secs": pipeline_total_time,
                },
            }
            run_rows.append(row)
            log("[result-run] " + json.dumps(row, ensure_ascii=False))

    summary_rows = []
    for dataset in DATASETS:
        rows = [row for row in run_rows if row["dataset"] == dataset]
        summary = {"dataset": dataset}
        for field in [
            "per_instance_input",
            "per_instance_output",
            "per_instance_total",
        ]:
            run_means = [row[field]["mean"] for row in rows]
            summary[f"{field}_mean_across_runs"] = mean(run_means)
            summary[f"{field}_std_across_runs"] = pstdev(run_means)
        for field in [
            "retriever_avg_secs_per_instance",
            "reranker_avg_secs_per_instance",
            "pipeline_avg_secs_per_instance",
        ]:
            run_means = [row["time"][field] for row in rows]
            summary[f"time_{field}_mean_across_runs"] = mean(run_means)
            summary[f"time_{field}_std_across_runs"] = pstdev(run_means)
        summary_rows.append(summary)
        log("[result-summary] " + json.dumps(summary, ensure_ascii=False))

    result = {
        "tokenizer": TOKENIZER_NAME,
        "note": (
            "Offline token estimate from saved histories. Input tokens use "
            "Qwen chat template per no-context window; output tokens encode "
            "the saved response text."
        ),
        "runs": run_rows,
        "two_run_summary_of_run_means": summary_rows,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[wrote] {OUT_JSON}")
    log(f"[done] {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("[error]")
        traceback.print_exc()
        raise
