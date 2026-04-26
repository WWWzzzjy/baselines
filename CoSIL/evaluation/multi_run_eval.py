import argparse
import json
import os
import shlex
import sys
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


TIME_KEYS = {
    "elapsed",
    "elapsed_time",
    "inference_time",
    "latency",
    "runtime",
    "seconds",
    "time",
    "time_cost",
    "total_time",
    "wall_time",
    "wall_time_seconds",
}


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_variance(values: list[float]) -> float:
    return statistics.variance(values) if len(values) > 1 else 0.0


def parse_gt_entries(gt_entries: list[str]) -> tuple[set[str], set[str]]:
    files, funcs = set(), set()
    for entry in gt_entries:
        parts = entry.split("::", maxsplit=1)
        if len(parts) == 1:
            files.add(parts[0])
        else:
            file_name, func_name = parts
            files.add(file_name)
            funcs.add(func_name)
    return files, funcs


def extract_predicted_funcs(found_related_locs: Any) -> list[str]:
    if not isinstance(found_related_locs, dict):
        return []

    predicted = []
    for locs in found_related_locs.values():
        if isinstance(locs, str):
            locs = [locs]
        if not isinstance(locs, list):
            continue
        for loc in locs:
            if not isinstance(loc, str):
                continue
            for entry in loc.splitlines():
                entry = entry.strip()
                if entry.startswith("function:") or entry.startswith("class:"):
                    _, value = entry.split(":", maxsplit=1)
                    value = value.strip()
                    if value:
                        predicted.append(value)
    return predicted


def f1_score(gt: set[str], pred: list[str]) -> float:
    pred_set = set(x for x in pred if x)
    if not gt and not pred_set:
        return 1.0
    if not gt or not pred_set:
        return 0.0
    true_positive = len(gt & pred_set)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gt)
    return 2 * precision * recall / (precision + recall)


def acc_at_k(gt: set[str], pred: list[str], k: int) -> float:
    return 1.0 if any(item in gt for item in pred[:k]) else 0.0


def walk_numbers(obj: Any, key_hint: str | None = None) -> list[tuple[str | None, float]]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(walk_numbers(value, str(key)))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(walk_numbers(value, key_hint))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        found.append((key_hint, float(obj)))
    return found


def extract_usage(row: dict[str, Any]) -> dict[str, float]:
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0

    def visit(obj: Any) -> None:
        nonlocal prompt_tokens, completion_tokens, total_tokens
        if isinstance(obj, dict):
            usage = obj.get("usage")
            if isinstance(usage, dict):
                prompt_tokens += float(usage.get("prompt_tokens", 0) or 0)
                completion_tokens += float(usage.get("completion_tokens", 0) or 0)
                total_tokens += float(usage.get("total_tokens", 0) or 0)
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)

    visit(row)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def extract_time(row: dict[str, Any]) -> float:
    if "inference_time_seconds" in row:
        return float(row.get("inference_time_seconds", 0.0) or 0.0)
    total = 0.0
    for key, value in walk_numbers(row):
        if key and key.lower() in TIME_KEYS:
            total += value
    return total


def load_dataset_instance_ids(dataset_name: str, num_instances: int = 0) -> list[str] | None:
    local_path = Path("datasets") / dataset_name
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        return None

    if local_path.exists():
        dataset = load_from_disk(str(local_path))
    elif "/" in dataset_name:
        dataset = load_dataset(dataset_name, split="test")
    else:
        return None

    if num_instances > 0:
        dataset = dataset.select(range(min(num_instances, len(dataset))))
    return [item["instance_id"] for item in dataset if "instance_id" in item]


def load_run_times(path: str | Path | None) -> dict[int, float]:
    if not path:
        return {}
    run_times = {}
    for row in load_jsonl(path):
        if "run_id" in row and "inference_time_seconds" in row:
            run_times[int(row["run_id"])] = float(row["inference_time_seconds"])
    return run_times


def default_gt_file(dataset_name: str) -> Path:
    dataset_lower = dataset_name.lower()
    if "verified" in dataset_lower:
        return Path("evaluation") / "gt_verified.json"
    return Path("evaluation") / "gt.json"


def evaluate_one_run(
    loc_file: str | Path,
    gt_data: dict[str, list[str]],
    instance_ids: list[str] | None = None,
    measured_wall_time: float | None = None,
) -> dict[str, Any]:
    rows = load_jsonl(loc_file)
    rows_by_id = {row["instance_id"]: row for row in rows if "instance_id" in row}
    eval_ids = instance_ids if instance_ids is not None else sorted(gt_data.keys())

    file_acc5 = []
    file_f1 = []
    func_acc3 = []
    func_f1 = []
    token_usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    extracted_time = 0.0

    missing = 0
    for instance_id in eval_ids:
        gt_files, gt_funcs = parse_gt_entries(gt_data.get(instance_id, []))
        row = rows_by_id.get(instance_id)
        if row is None:
            missing += 1
            pred_files = []
            pred_funcs = []
        else:
            pred_files = row.get("found_files", []) or []
            pred_funcs = extract_predicted_funcs(row.get("found_related_locs", {}))
            usage = extract_usage(row)
            for key in token_usage:
                token_usage[key] += usage[key]
            extracted_time += extract_time(row)

        file_acc5.append(acc_at_k(gt_files, pred_files, 5))
        file_f1.append(f1_score(gt_files, pred_files[:5]))
        func_acc3.append(acc_at_k(gt_funcs, pred_funcs, 3))
        func_f1.append(f1_score(gt_funcs, pred_funcs[:3]))

    inference_time = measured_wall_time if measured_wall_time is not None else extracted_time
    return {
        "loc_file": str(loc_file),
        "num_instances": len(eval_ids),
        "num_predictions": len(rows_by_id),
        "missing_predictions": missing,
        "file_acc_at_5": safe_mean(file_acc5),
        "file_f1": safe_mean(file_f1),
        "func_acc_at_3": safe_mean(func_acc3),
        "func_f1": safe_mean(func_f1),
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
        "inference_time_seconds": inference_time,
    }


def summarize_runs(run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "file_acc_at_5",
        "file_f1",
        "func_acc_at_3",
        "func_f1",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "inference_time_seconds",
    ]
    summary = {"num_runs": len(run_metrics)}
    for key in metric_keys:
        values = [float(run[key]) for run in run_metrics]
        summary[f"{key}_mean"] = safe_mean(values)
        summary[f"{key}_variance"] = safe_variance(values)
    return summary


def format_template(template: str, run_id: int, run_index: int) -> str:
    return template.format(run=run_id, run_id=run_id, i=run_index, idx=run_index)


def parse_run_ids(run_ids: str | None, start_run: int, num_runs: int) -> list[int]:
    if not run_ids:
        return [start_run + run_index for run_index in range(num_runs)]
    return [int(item.strip()) for item in run_ids.split(",") if item.strip()]


def run_command(command: str) -> float:
    start = time.perf_counter()
    subprocess.run(command, shell=True, check=True)
    return time.perf_counter() - start


def print_run_metrics(run_id: int, metrics: dict[str, Any]) -> None:
    print(f"Run {run_id}: {metrics['loc_file']}", flush=True)
    print(f"  File Acc@5: {metrics['file_acc_at_5'] * 100:.2f}%", flush=True)
    print(f"  File F1: {metrics['file_f1'] * 100:.2f}%", flush=True)
    print(f"  Func Acc@3: {metrics['func_acc_at_3'] * 100:.2f}%", flush=True)
    print(f"  Func F1: {metrics['func_f1'] * 100:.2f}%", flush=True)
    print(f"  Tokens: {metrics['total_tokens']:.0f}", flush=True)
    print(f"  Inference time: {metrics['inference_time_seconds']:.2f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-run CoSIL localization outputs."
    )
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--gt-file", default=None)
    parser.add_argument("--num-runs", type=int, default=int(os.environ.get("NUM_RUNS", 1)))
    parser.add_argument("--num-instances", type=int, default=int(os.environ.get("NUM_INSTANCES", 0)))
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument(
        "--run-ids",
        default=None,
        help="Comma-separated run ids, e.g. 1,3,5. Overrides --num-runs/--start-run.",
    )
    parser.add_argument(
        "--loc-file-template",
        required=True,
        help="Path template for per-run JSONL. Supports {run}, {run_id}, {i}, {idx}.",
    )
    parser.add_argument(
        "--run-command-template",
        default=None,
        help="Optional shell command template to execute before evaluating each run.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--run-time-jsonl", default=None)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    log_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(args.log_file, "a")
        sys.stdout = Tee(sys.stdout, log_handle)
        sys.stderr = Tee(sys.stderr, log_handle)
        print(f"[multi-run-eval] log_file={args.log_file}", flush=True)

    gt_file = Path(args.gt_file) if args.gt_file else default_gt_file(args.dataset)
    print(f"[multi-run-eval] loading gt_file={gt_file}", flush=True)
    gt_data = load_json(gt_file)
    print(f"[multi-run-eval] loaded gt entries={len(gt_data)}", flush=True)
    print(f"[multi-run-eval] resolving dataset filter dataset={args.dataset}", flush=True)
    instance_ids = load_dataset_instance_ids(args.dataset, args.num_instances)
    if instance_ids is not None:
        gt_data = {k: v for k, v in gt_data.items() if k in instance_ids}
        print(
            f"[multi-run-eval] filtered by dataset order instances={len(instance_ids)} gt_entries={len(gt_data)}",
            flush=True,
        )
    else:
        print("[multi-run-eval] no local dataset filter found; using all gt entries", flush=True)

    run_ids = parse_run_ids(args.run_ids, args.start_run, args.num_runs)
    run_times = load_run_times(args.run_time_jsonl)
    if args.run_time_jsonl:
        print(f"[multi-run-eval] loaded run times from {args.run_time_jsonl}: {run_times}", flush=True)
    print(f"[multi-run-eval] run_ids={run_ids}", flush=True)
    run_metrics = []
    for run_index, run_id in enumerate(run_ids):
        wall_time = None
        if args.run_command_template:
            command = format_template(args.run_command_template, run_id, run_index)
            print(f"[multi-run-eval] running command for run {run_id}: {command}", flush=True)
            wall_time = run_command(command)
        elif run_id in run_times:
            wall_time = run_times[run_id]

        loc_file = format_template(args.loc_file_template, run_id, run_index)
        print(f"[multi-run-eval] evaluating run {run_id} loc_file={loc_file}", flush=True)
        metrics = evaluate_one_run(
            loc_file=loc_file,
            gt_data=gt_data,
            instance_ids=instance_ids,
            measured_wall_time=wall_time,
        )
        metrics["run_id"] = run_id
        run_metrics.append(metrics)
        print_run_metrics(run_id, metrics)

    summary = summarize_runs(run_metrics)
    print("\nSummary:", flush=True)
    for key, value in summary.items():
        if key == "num_runs":
            print(f"  {key}: {value}", flush=True)
        elif key.endswith("_mean") and any(metric in key for metric in ["acc", "f1"]):
            print(f"  {key}: {value * 100:.2f}%", flush=True)
        else:
            print(f"  {key}: {value:.4f}", flush=True)

    if args.output_json:
        output = {
            "dataset": args.dataset,
            "gt_file": str(gt_file),
            "instance_count": len(gt_data),
            "runs": run_metrics,
            "summary": summary,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[multi-run-eval] wrote output_json={args.output_json}", flush=True)

    if log_handle is not None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    main()
