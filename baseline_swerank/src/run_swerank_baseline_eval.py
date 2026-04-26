import collections
import copy
import json
import os
import re
import statistics
import time
from pathlib import Path

from reranker import convert_format, rerank_llm
from refactored_eval_localization import (
    cal_metrics_w_dataset,
    get_sorted_documents_func,
    load_json,
    load_qrels,
)


DEFAULT_OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENAI_API_KEY = "sk-482fa2a1567041ecafa9c3114bf3811d"
DEFAULT_LOCAGENT_MODEL = "qwen3-coder-plus"


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
    if not content:
        return []
    if content[0] == "[":
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def dump_jsonl(rows, file_path):
    with open(file_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def normalize_dataset_name(dataset_name):
    normalized = dataset_name.strip().lower().replace("_", "-")
    aliases = {
        "swe-bench-lite": "swe-bench-lite",
        "swebench-lite": "swe-bench-lite",
        "swe-bench-verified": "swe-bench-verified",
        "swebench-verified": "swe-bench-verified",
        "loc-bench": "loc-bench",
        "locbench": "loc-bench",
    }
    return aliases.get(normalized, normalized)


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def compute_variance(values):
    if len(values) <= 1:
        return 0.0
    return round(statistics.variance(values), 8)


def compute_std(values):
    if len(values) <= 1:
        return 0.0
    return round(statistics.stdev(values), 8)


def metric_at(result_dict, prefix, k):
    return float(result_dict[f"{prefix}@{k}"])


def collect_history_stats(history_path):
    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_latency_seconds = 0.0

    for run_history in history.get("run_history", []):
        for step in run_history:
            usage = step.get("usage", {})
            total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)
            total_latency_seconds += float(step.get("latency_seconds", 0.0) or 0.0)

    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "rerank_api_latency_seconds": round(total_latency_seconds, 4),
    }


def load_retriever_eval_times(retriever_output_dir, selected_ids):
    results_path = Path(retriever_output_dir)
    eval_path = results_path.with_name(
        results_path.stem.replace("_results", "_eval_per_instance") + ".json"
    )
    if not eval_path.exists():
        return {instance_id: 0.0 for instance_id in selected_ids}

    rows = json.loads(eval_path.read_text(encoding="utf-8"))
    return {
        instance_id: round(float(rows[index].get("time", 0.0) or 0.0), 4)
        for index, instance_id in enumerate(selected_ids)
        if index < len(rows)
    }


def select_instances(retriever_rows, dataset_dir, dataset_name, limit):
    selected = []
    for row in retriever_rows:
        instance_id = row["instance_id"]
        dataset_path = Path(dataset_dir) / f"{dataset_name}-function_{instance_id}"
        if dataset_path.exists():
            selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    if limit > 0 and len(selected) < limit:
        raise ValueError(
            f"Only found {len(selected)} valid instances under {dataset_dir} for dataset {dataset_name}, "
            f"but at least {limit} are required."
        )
    if limit <= 0 and not selected:
        raise ValueError(
            f"No valid instances found under {dataset_dir} for dataset {dataset_name}."
        )
    return selected


def evaluate_subset(dataset_name, split, dataset_dir, reranker_results, selected_ids, retriever_subset_path):
    qrels = load_qrels(dataset_name, split, dataset_dir)

    file_res = cal_metrics_w_dataset(
        str(retriever_subset_path),
        "docs",
        "file",
        dataset_name,
        split,
        k_values=[5],
        metrics=["acc", "precision", "recall"],
        selected_list=selected_ids,
        qrels=qrels,
        reranker_results=copy.deepcopy(reranker_results),
    )
    func_res = cal_metrics_w_dataset(
        str(retriever_subset_path),
        "docs",
        "function",
        dataset_name,
        split,
        k_values=[5],
        metrics=["acc", "precision", "recall"],
        selected_list=selected_ids,
        qrels=qrels,
        reranker_results=copy.deepcopy(reranker_results),
    )

    file_precision = metric_at(file_res, "P", 5)
    file_recall = metric_at(file_res, "Recall", 5)
    func_precision = metric_at(func_res, "P", 5)
    func_recall = metric_at(func_res, "Recall", 5)

    return {
        "func_acc_at_5": metric_at(func_res, "Acc", 5),
        "func_f1_at_5": compute_f1(func_precision, func_recall),
        "files_acc_at_5": metric_at(file_res, "Acc", 5),
        "files_f1_at_5": compute_f1(file_precision, file_recall),
        "func_precision_at_5": func_precision,
        "func_recall_at_5": func_recall,
        "files_precision_at_5": file_precision,
        "files_recall_at_5": file_recall,
    }


REQUESTED_METRIC_KEYS = [
    ("files_acc@5", "files_acc_at_5"),
    ("files_f1@5", "files_f1_at_5"),
    ("func_acc@5", "func_acc_at_5"),
    ("func_f1@5", "func_f1_at_5"),
    ("time", "total_time_seconds"),
    ("tokens", "total_tokens"),
]


def aggregate_runs(run_metrics):
    summary = {}
    for key in [
        "func_acc_at_5",
        "func_f1_at_5",
        "files_acc_at_5",
        "files_f1_at_5",
        "total_tokens",
        "retriever_time_seconds",
        "rerank_time_seconds",
        "total_time_seconds",
        "rerank_api_latency_seconds",
    ]:
        values = [float(run[key]) for run in run_metrics]
        summary[key] = {
            "mean": round(sum(values) / len(values), 4),
            "variance": compute_variance(values),
            "std": compute_std(values),
            "values": values,
        }
    return summary


def build_requested_summary(run_metrics):
    instance_metric_values = collections.defaultdict(lambda: collections.defaultdict(list))
    for run in run_metrics:
        for instance_row in run.get("instances", []):
            instance_id = instance_row["instance_id"]
            for output_key, _source_key in REQUESTED_METRIC_KEYS:
                instance_metric_values[output_key][instance_id].append(float(instance_row[output_key]))

    summary = {}
    for output_key, _source_key in REQUESTED_METRIC_KEYS:
        instance_means = {
            instance_id: round(sum(values) / len(values), 4)
            for instance_id, values in instance_metric_values[output_key].items()
        }
        values = list(instance_means.values())
        summary[output_key] = {
            "mean": round(sum(values) / len(values), 4) if values else 0.0,
            "std": compute_std(values),
            "values": values,
            "instance_means": instance_means,
        }
    return summary


def run_eval(
    retriever_output_dir,
    dataset_dir="datasets",
    dataset_name="swe-bench-verified",
    split="test",
    output_dir="outputs",
    eval_dir="eval_results",
    model=os.environ.get("LOCAGENT_MODEL", DEFAULT_LOCAGENT_MODEL),
    api_base=os.environ.get("OPENAI_API_BASE", DEFAULT_OPENAI_API_BASE),
    api_key=os.environ.get("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY),
    api_type=None,
    api_version=None,
    num_instances=3,
    num_runs=3,
    top_k=100,
    window_size=10,
    step_size=5,
    context_size=32768,
    temperature=0.6,
    top_p=0.95,
    max_tokens=32768,
    run_tag="dashscope-baseline",
    retriever_time_seconds=0.0,
    rerank_tag="rerank-small",
):
    dataset_name = normalize_dataset_name(dataset_name)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    retriever_rows = load_jsonl(retriever_output_dir)
    selected_rows = select_instances(retriever_rows, dataset_dir, dataset_name, num_instances)
    selected_ids = [row["instance_id"] for row in selected_rows]
    retriever_eval_times = load_retriever_eval_times(retriever_output_dir, selected_ids)

    subset_dir = Path(output_dir) / "baseline_eval_inputs"
    subset_dir.mkdir(parents=True, exist_ok=True)
    retriever_work_dir = Path(output_dir) / "retriever"
    retriever_work_dir.mkdir(parents=True, exist_ok=True)
    subset_name = f"{sanitize_name(dataset_name)}_{sanitize_name(run_tag)}_{num_instances}instances.jsonl"
    retriever_subset_path = subset_dir / subset_name
    dump_jsonl(selected_rows, retriever_subset_path)

    dataset_instances = [f"{dataset_name}-function_{instance_id}" for instance_id in selected_ids]
    run_metrics = []

    for run_idx in range(num_runs):
        run_name = f"{sanitize_name(run_tag)}-run{run_idx + 1}"
        data_type = f"{sanitize_name(rerank_tag)}-run{run_idx + 1}"
        rerank_start_time = time.perf_counter()

        convert_format.convert_results_single_pass(
            prefix=dataset_name,
            data_dir=dataset_dir,
            data_type=data_type,
            output_path=str(retriever_work_dir),
            retriever_results_path=str(retriever_subset_path),
            top_k=top_k,
            rerank_type="code",
        )

        rerank_llm.process_dataset(
            model=model,
            output_path=output_dir,
            data_dir=dataset_dir,
            dataset=dataset_instances,
            data_type=data_type,
            eval_dir=eval_dir,
            use_logits=0,
            use_alpha=0,
            llm_top_k=top_k,
            window_size=window_size,
            step_size=step_size,
            batched=0,
            context_size=context_size,
            rerank_type="code",
            code_prompt_type="github_issue",
            api_config={
                "keys": api_key,
                "api_type": api_type,
                "api_base": api_base,
                "api_version": api_version,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            converted_input_root=str(retriever_work_dir),
        )
        rerank_time_seconds = round(time.perf_counter() - rerank_start_time, 4)

        reranker_results = {}
        instance_history_totals = {}
        history_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "rerank_api_latency_seconds": 0.0,
        }

        for dataset_instance in dataset_instances:
            result_path = (
                Path(output_dir)
                / data_type
                / dataset_instance
                / f"rerank_{top_k}_llm_gen_num.json"
            )
            history_path = (
                Path(output_dir)
                / data_type
                / dataset_instance
                / f"rerank_{top_k}_llm_gen_num_histories.json"
            )
            if not result_path.exists():
                raise FileNotFoundError(f"Missing rerank result: {result_path}")
            if not history_path.exists():
                raise FileNotFoundError(f"Missing rerank history: {history_path}")

            retrieved_results = load_json(str(result_path))[0]
            get_sorted_documents_func(retrieved_results, reranker_results)

            instance_history = collect_history_stats(str(history_path))
            instance_id = dataset_instance.split("-function_", 1)[-1]
            instance_history_totals[instance_id] = instance_history
            for key in history_totals:
                history_totals[key] += instance_history[key]

        total_tokens = int(history_totals["total_tokens"])
        retriever_time_seconds = round(float(retriever_time_seconds), 4)
        total_time_seconds = round(retriever_time_seconds + rerank_time_seconds, 4)

        metric_row = evaluate_subset(
            dataset_name=dataset_name,
            split=split,
            dataset_dir=dataset_dir,
            reranker_results=reranker_results,
            selected_ids=selected_ids,
            retriever_subset_path=retriever_subset_path,
        )
        instance_metrics = []
        for instance_id in selected_ids:
            instance_metric_row = evaluate_subset(
                dataset_name=dataset_name,
                split=split,
                dataset_dir=dataset_dir,
                reranker_results=reranker_results,
                selected_ids=[instance_id],
                retriever_subset_path=retriever_subset_path,
            )
            instance_history = instance_history_totals.get(
                instance_id,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "rerank_api_latency_seconds": 0.0,
                },
            )
            instance_retriever_time = round(float(retriever_eval_times.get(instance_id, 0.0)), 4)
            instance_time = round(
                instance_retriever_time
                + float(instance_history.get("rerank_api_latency_seconds", 0.0) or 0.0),
                4,
            )
            instance_metric_row.update(
                {
                    "instance_id": instance_id,
                    "run_index": run_idx + 1,
                    "prompt_tokens": instance_history["prompt_tokens"],
                    "completion_tokens": instance_history["completion_tokens"],
                    "total_tokens": int(instance_history["total_tokens"]),
                    "retriever_time_seconds": instance_retriever_time,
                    "rerank_api_latency_seconds": round(
                        float(instance_history["rerank_api_latency_seconds"]), 4
                    ),
                    "time": instance_time,
                    "tokens": int(instance_history["total_tokens"]),
                    "files_acc@5": instance_metric_row["files_acc_at_5"],
                    "files_f1@5": instance_metric_row["files_f1_at_5"],
                    "func_acc@5": instance_metric_row["func_acc_at_5"],
                    "func_f1@5": instance_metric_row["func_f1_at_5"],
                }
            )
            instance_metrics.append(instance_metric_row)

        metric_row.update(
            {
                "run_index": run_idx + 1,
                "data_type": data_type,
                "selected_ids": selected_ids,
                "prompt_tokens": history_totals["prompt_tokens"],
                "completion_tokens": history_totals["completion_tokens"],
                "total_tokens": total_tokens,
                "retriever_time_seconds": retriever_time_seconds,
                "rerank_time_seconds": rerank_time_seconds,
                "total_time_seconds": total_time_seconds,
                "rerank_api_latency_seconds": round(history_totals["rerank_api_latency_seconds"], 4),
                "instances": instance_metrics,
            }
        )
        metric_row.update(
            {
                "files_acc@5": metric_row["files_acc_at_5"],
                "files_f1@5": metric_row["files_f1_at_5"],
                "func_acc@5": metric_row["func_acc_at_5"],
                "func_f1@5": metric_row["func_f1_at_5"],
                "time": metric_row["total_time_seconds"],
                "tokens": metric_row["total_tokens"],
            }
        )
        run_metrics.append(metric_row)

    summary = aggregate_runs(run_metrics)
    requested_summary = build_requested_summary(run_metrics)
    payload = {
        "config": {
            "retriever_output_dir": retriever_output_dir,
            "dataset_dir": dataset_dir,
            "dataset_name": dataset_name,
            "split": split,
            "model": model,
            "api_base": api_base,
            "num_instances": num_instances,
            "num_runs": num_runs,
            "top_k": top_k,
            "window_size": window_size,
            "step_size": step_size,
            "context_size": context_size,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "retriever_time_seconds": round(float(retriever_time_seconds), 4),
            "rerank_tag": rerank_tag,
            "selected_ids": selected_ids,
        },
        "runs": run_metrics,
        "summary": summary,
        "requested_summary": requested_summary,
    }

    report_path = Path(eval_dir) / f"{sanitize_name(run_tag)}_{sanitize_name(dataset_name)}_baseline_eval.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"\nSaved report to: {report_path}")
    print("\nRequested metrics summary:")
    print(json.dumps(requested_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire(run_eval)
