import json
import time
from pathlib import Path
from statistics import mean, pstdev


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent / "results"
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
            if not files:
                raise FileNotFoundError(
                    f"No history files found under {run_dir}. "
                    "Check that ROOT points to the results directory."
                )

            instance_inputs = []
            instance_outputs = []
            instance_totals = []
            instance_reranker_times = []

            for path in files:
                input_tokens = 0
                output_tokens = 0
                reranker_time = load_instance_time(path.parent)

                with path.open(encoding="utf-8") as file:
                    data = json.load(file)

                for entry in iter_entries(data.get("run_history", [])):
                    input_tokens += count_prompt(tokenizer, entry.get("prompt"))
                    output_tokens += count_response(tokenizer, entry.get("response"))

                instance_inputs.append(input_tokens)
                instance_outputs.append(output_tokens)
                instance_totals.append(input_tokens + output_tokens)
                instance_reranker_times.append(reranker_time)

            num_instances = len(instance_totals)
            input_total = sum(instance_inputs)
            output_total = sum(instance_outputs)
            token_total = sum(instance_totals)
            reranker_total_time = sum(instance_reranker_times)
            reranker_avg_time = reranker_total_time / max(num_instances, 1)

            row = {
                "dataset": dataset,
                "run": run_idx,
                "num_instances": num_instances,
                "token": {
                    "input_total": input_total,
                    "output_total": output_total,
                    "total": token_total,
                    "input_avg_per_instance": input_total / num_instances,
                    "output_avg_per_instance": output_total / num_instances,
                    "total_avg_per_instance": token_total / num_instances,
                },
                "time": {
                    "retriever_avg_secs_per_instance": retriever_avg_time,
                    "retriever_total_secs": retriever_total_time,
                    "reranker_avg_secs_per_instance": reranker_avg_time,
                    "reranker_total_secs": reranker_total_time,
                },
            }
            run_rows.append(row)
            log("[result-run] " + json.dumps(row, ensure_ascii=False))

    summary_rows = []
    for dataset in DATASETS:
        rows = [row for row in run_rows if row["dataset"] == dataset]
        summary = {"dataset": dataset}
        for field in [
            "input_avg_per_instance",
            "output_avg_per_instance",
            "total_avg_per_instance",
        ]:
            run_values = [row["token"][field] for row in rows]
            summary[f"token_{field}_mean"] = mean(run_values)
            summary[f"token_{field}_std"] = pstdev(run_values)
        for field in [
            "retriever_avg_secs_per_instance",
            "reranker_avg_secs_per_instance",
        ]:
            run_values = [row["time"][field] for row in rows]
            summary[f"time_{field}_mean"] = mean(run_values)
            summary[f"time_{field}_std"] = pstdev(run_values)
        summary_rows.append(summary)
        log("[result-summary] " + json.dumps(summary, ensure_ascii=False))

    result = {
        "tokenizer": TOKENIZER_NAME,
        "note": (
            "Offline token estimate from saved histories. Input tokens use "
            "Qwen chat template per no-context window; output tokens encode "
            "the saved response text. Retriever time is read from retriever "
            "output files; reranker time is read from per-instance time_metrics.json. "
            "Std is computed only across run-level per-instance averages."
        ),
        "runs": run_rows,
        "summary": summary_rows,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[wrote] {OUT_JSON}")
    log(f"[done] {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
