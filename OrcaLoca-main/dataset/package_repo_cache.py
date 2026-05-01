#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from Orcar.load_cache_dataset import load_filter_hf_dataset_explicit


DATASET_CHOICES = {
    "lite": {
        "dataset": "princeton-nlp/SWE-bench_Lite",
        "cache_dir": "repo_cache_lite",
    },
    "v1": {
        "dataset": "czlll/Loc-Bench_V1",
        "cache_dir": "repo_cache_v1",
    },
}


def get_repo_dir(repo: str) -> str:
    return repo.replace("/", "__")


def run_cmd(
    cmd: list[str],
    cwd: str | Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cwd_str = str(cwd) if cwd is not None else None
    print("$ " + (f"(cd {cwd_str} && " if cwd_str else "") + " ".join(cmd) + (")" if cwd_str else ""))
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def git_exists(repo_path: Path, treeish_path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", treeish_path],
        cwd=repo_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0


def load_instances(dataset_key: str, split: str, filter_instance: str) -> list[dict[str, Any]]:
    dataset_name = DATASET_CHOICES[dataset_key]["dataset"]
    ds = load_filter_hf_dataset_explicit(dataset_name, filter_instance, split)
    return [dict(row) for row in ds]


def collect_submodule_commits(
    instances: list[dict[str, Any]], cache_dir: Path
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]], list[dict[str, str]]]:
    commits_by_repo: dict[str, set[str]] = defaultdict(set)
    inst_by_repo_commit: dict[tuple[str, str], list[str]] = defaultdict(list)
    missing_repos: list[dict[str, str]] = []
    missing_commits: list[dict[str, str]] = []

    for inst in instances:
        repo = inst["repo"]
        repo_dir = get_repo_dir(repo)
        commit = inst["base_commit"]
        commits_by_repo[repo_dir].add(commit)
        inst_by_repo_commit[(repo_dir, commit)].append(inst["instance_id"])

    needed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for repo_dir, commits in sorted(commits_by_repo.items()):
        repo_path = cache_dir / repo_dir
        if not (repo_path / ".git").exists():
            missing_repos.append({"repo_dir": repo_dir, "path": str(repo_path)})
            continue

        for commit in sorted(commits):
            if not git_exists(repo_path, f"{commit}^{{commit}}"):
                missing_commits.append(
                    {
                        "repo_dir": repo_dir,
                        "commit": commit,
                        "instances": ",".join(inst_by_repo_commit[(repo_dir, commit)]),
                    }
                )
                continue
            if git_exists(repo_path, f"{commit}:.gitmodules"):
                needed[repo_dir].append(
                    {
                        "commit": commit,
                        "instances": ",".join(inst_by_repo_commit[(repo_dir, commit)]),
                    }
                )
    return needed, missing_repos, missing_commits


def hydrate_submodules(cache_dir: Path, needed: dict[str, list[dict[str, str]]]) -> None:
    for repo_dir, entries in sorted(needed.items()):
        repo_path = cache_dir / repo_dir
        print(f"\n== {repo_dir}: {len(entries)} commit(s) with submodules ==")
        for entry in entries:
            commit = entry["commit"]
            print(f"-- hydrate {repo_dir} at {commit} ({entry['instances']})")
            run_cmd(["git", "reset", "--hard", commit], cwd=repo_path)
            run_cmd(["git", "submodule", "sync", "--recursive"], cwd=repo_path)
            run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_path)


def verify_no_fetch(cache_dir: Path, needed: dict[str, list[dict[str, str]]]) -> None:
    for repo_dir, entries in sorted(needed.items()):
        repo_path = cache_dir / repo_dir
        print(f"\n== verify offline submodules for {repo_dir} ==")
        for entry in entries:
            commit = entry["commit"]
            print(f"-- verify --no-fetch at {commit} ({entry['instances']})")
            run_cmd(["git", "reset", "--hard", commit], cwd=repo_path)
            run_cmd(["git", "submodule", "sync", "--recursive"], cwd=repo_path)
            run_cmd(
                [
                    "git",
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--force",
                    "--no-fetch",
                ],
                cwd=repo_path,
            )
            run_cmd(["git", "submodule", "status", "--recursive"], cwd=repo_path)


def package_cache(cache_dir: Path, archive_path: Path) -> None:
    cache_dir = cache_dir.resolve()
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "tar",
            "-czf",
            str(archive_path),
            "-C",
            str(cache_dir.parent),
            cache_dir.name,
        ]
    )
    print(f"\nArchive written: {archive_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hydrate git submodules for a prepared repo cache and package it for "
            "offline evaluation. This script never clones missing main repos."
        )
    )
    parser.add_argument("dataset", choices=sorted(DATASET_CHOICES))
    parser.add_argument("--split", default="test")
    parser.add_argument("--filter_instance", default=".*")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--skip_package", action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir or DATASET_CHOICES[args.dataset]["cache_dir"])
    archive_path = Path(args.archive or f"{cache_dir.name}.tar.gz")

    if not cache_dir.is_dir():
        raise SystemExit(f"Cannot find cache_dir: {cache_dir}")
    re.compile(args.filter_instance)

    print("Preparing repo cache package")
    print(f"  dataset: {DATASET_CHOICES[args.dataset]['dataset']}/{args.split}")
    print(f"  cache:   {cache_dir}")
    print(f"  archive: {archive_path}")

    instances = load_instances(args.dataset, args.split, args.filter_instance)
    needed, missing_repos, missing_commits = collect_submodule_commits(
        instances, cache_dir
    )

    print(f"\nInstances: {len(instances)}")
    print(f"Repos needing submodule hydration: {len(needed)}")
    for repo_dir, entries in sorted(needed.items()):
        print(f"  {repo_dir}: {len(entries)} commit(s)")
        for entry in entries:
            print(f"    {entry['commit']}  {entry['instances']}")

    if missing_repos:
        print("\nMissing repos, skipped:")
        for item in missing_repos:
            print(f"  {item['repo_dir']}  {item['path']}")

    if missing_commits:
        print("\nMissing base commits, skipped:")
        for item in missing_commits:
            print(f"  {item['repo_dir']} {item['commit']}  {item['instances']}")

    if args.check_only:
        return

    hydrate_submodules(cache_dir, needed)
    if not args.skip_verify:
        verify_no_fetch(cache_dir, needed)
    if not args.skip_package:
        package_cache(cache_dir, archive_path)


if __name__ == "__main__":
    main()
