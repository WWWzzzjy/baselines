#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
import traceback
from typing import Any

from llama_index.core.chat_engine.types import AgentChatResponse

from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout
from Orcar.search_agent import SearchAgent
from Orcar.types import SearchInput, TraceAnalysisOutput

logger = get_logger(__name__)


def get_repo_dir(repo: str) -> str:
    return repo.replace("/", "__")


def run_cmd(cmd: list[str], cwd: str | None = None) -> str:
    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.info(result.stderr)
    return result.stdout


def reset_repo(repo_path: str, base_commit: str) -> None:
    run_cmd(["git", "reset", "--hard", base_commit], cwd=repo_path)
    run_cmd(["git", "submodule", "update", "--init", "--recursive", "--force"], cwd=repo_path)
    run_cmd(["git", "submodule", "deinit", "-f", "--all"], cwd=repo_path)
    modules_dir = os.path.join(repo_path, ".git", "modules")
    if os.path.isdir(modules_dir):
        run_cmd(["rm", "-rf", modules_dir])
    run_cmd(["git", "diff", "--quiet"], cwd=repo_path)


def ensure_repo(inst: dict[str, Any], cache_dir: str) -> str:
    repo_name = inst["repo"]
    repo_path = os.path.join(cache_dir, get_repo_dir(repo_name))
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.isdir(repo_path):
        run_cmd(["git", "clone", f"https://github.com/{repo_name}.git", repo_path])

    reset_repo(repo_path, inst["base_commit"])
    return repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run static OrcaLoca localization without Docker/conda/reproducer execution. "
            "This evaluates localization only: problem statement + local repo graph -> searcher JSON."
        )
    )
    parser.add_argument("--model", default=os.environ.get("MODEL", "claude-3-5-sonnet-20241022"))
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "princeton-nlp/SWE-bench_Lite"))
    parser.add_argument("--split", default=os.environ.get("SPLIT", "test"))
    parser.add_argument("--cfg_path", default=os.environ.get("CFG_PATH", "./key.cfg"))
    parser.add_argument("--cache_dir", default=os.environ.get("CACHE_DIR", "~/.orcar"))
    parser.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR", "./output_local"))
    parser.add_argument("--filter_instance", default=None)
    parser.add_argument("--instance_ids", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stdout_log", action="store_true")
    args = parser.parse_args()

    if args.instance_ids:
        args.filter_instance = "^(" + "|".join(args.instance_ids) + ")$"
    elif args.filter_instance is None:
        raise SystemExit(
            "Please pass --instance_ids ID [ID ...] or --filter_instance 'regex'. "
            "Use --filter_instance '.*' only when you really want the full split."
        )

    return args


def run_instance(
    inst: dict[str, Any], llm: Any, repo_path: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    inference_start = time.perf_counter()
    agent_init_start = time.perf_counter()
    search_input = SearchInput(
        problem_statement=inst["problem_statement"],
        trace_analysis_output=TraceAnalysisOutput(),
    )
    search_agent = SearchAgent(
        repo_path=repo_path,
        llm=llm,
        search_input=search_input,
        verbose=False,
        config_path="search.cfg",
    )
    agent_init_time_s = time.perf_counter() - agent_init_start
    response: AgentChatResponse = search_agent.chat(search_input.get_content())
    wall_time_s = time.perf_counter() - inference_start

    agent_worker = getattr(search_agent, "agent_worker", None) or getattr(
        search_agent, "_agent_worker", None
    )
    inference_stats = dict(getattr(agent_worker, "last_inference_stats", {}) or {})
    inference_stats.update(
        {
            "agent_init_time_s": agent_init_time_s,
            "wall_time_s": wall_time_s,
        }
    )
    message_records = list(getattr(agent_worker, "last_message_records", []) or [])
    return json.loads(response.response), inference_stats, message_records


def dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    args = parse_args()
    cache_dir = os.path.expanduser(args.cache_dir)
    output_base = os.path.abspath(args.output_dir)
    os.makedirs(output_base, exist_ok=True)

    if args.stdout_log:
        switch_log_to_stdout()
    else:
        switch_log_to_file()

    cfg = Config(args.cfg_path)
    llm = get_llm(model=args.model, max_tokens=4096, orcar_config=cfg)
    ds = load_filter_hf_dataset(args)
    if args.instance_ids:
        found_ids = set(ds["instance_id"])
        missing_ids = [inst_id for inst_id in args.instance_ids if inst_id not in found_ids]
        if missing_ids:
            raise SystemExit(
                "The following instance_ids were not found in the selected dataset: "
                + ", ".join(missing_ids)
            )

    print("Running static OrcaLoca localization")
    print(f"  model:     {args.model}")
    print(f"  dataset:   {args.dataset}/{args.split}")
    print(f"  instances: {len(ds)}")
    print(f"  cache:     {cache_dir}")
    print(f"  output:    {output_base}")

    for i, inst_raw in enumerate(ds):
        inst = dict(inst_raw)
        inst_id = inst["instance_id"]
        inst_output_dir = os.path.join(output_base, inst_id)
        output_path = os.path.join(inst_output_dir, f"searcher_{inst_id}.json")
        stats_path = os.path.join(inst_output_dir, f"inference_stats_{inst_id}.json")
        messages_path = os.path.join(inst_output_dir, f"messages_{inst_id}.jsonl")

        print(f"({i + 1:03d}/{len(ds):03d}) Current inst: {inst_id}")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"  skip existing: {output_path}")
            continue

        os.makedirs(inst_output_dir, exist_ok=True)
        log_dir = os.path.join("./log_local", inst_id)
        os.makedirs(log_dir, exist_ok=True)
        set_log_dir(log_dir)

        try:
            repo_path = ensure_repo(inst, cache_dir)
            # continue
            search_output, inference_stats, message_records = run_instance(
                inst, llm, repo_path
            )
            with open(output_path, "w") as f:
                json.dump(search_output, f, indent=4)
            with open(stats_path, "w") as f:
                json.dump(inference_stats, f, indent=4)
            dump_jsonl(messages_path, message_records)
            print(
                "  message tokens/time: "
                f"tokens={inference_stats.get('message_total_tokens', 0)}, "
                f"wall={inference_stats.get('wall_time_s', 0):.2f}s"
            )
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
