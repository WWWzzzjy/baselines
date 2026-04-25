import csv
import json
import os
from pathlib import Path

from datasets import load_dataset

from get_repo_structure.get_patch_info import find_py_or_non_dict_with_path, parse_patch_full
from get_repo_structure.get_repo_structure import get_project_structure_from_scratch


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower().replace("_", "-")
    aliases = {
        "swe-bench-verified": "swe-bench-verified",
        "swebench-verified": "swe-bench-verified",
    }
    return aliases.get(normalized, normalized)


def changed_functions_from_patch(instance, structure):
    patch_info = parse_patch_full(instance["patch"], structure)
    changed_funcs = set()

    for file_path, hunks in patch_info.items():
        for hunk in hunks:
            function_name = hunk.get("function_changed")
            if not function_name or hunk.get("newly_added"):
                continue
            class_name = hunk.get("class_changed")
            if class_name:
                changed_funcs.add(f"{file_path}/{class_name}/{function_name}")
            else:
                changed_funcs.add(f"{file_path}/{function_name}")

    return changed_funcs


def write_jsonl(rows, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_instance_dataset(instance, dataset_root):
    print(f"[build] instance={instance['instance_id']} repo={instance['repo']} commit={instance['base_commit']}", flush=True)
    structure = get_project_structure_from_scratch(
        instance["repo"],
        instance["base_commit"],
        instance["instance_id"],
        "playground",
    )
    print(f"[build] extracted structure for {instance['instance_id']}", flush=True)
    corpus_map = find_py_or_non_dict_with_path(
        structure["structure"],
        cond=instance["instance_id"].startswith("pytest-dev__"),
    )
    print(f"[build] corpus docs={len(corpus_map)} for {instance['instance_id']}", flush=True)
    changed_funcs = changed_functions_from_patch(instance, structure)
    print(f"[build] changed funcs={len(changed_funcs)} for {instance['instance_id']}", flush=True)

    if not changed_funcs:
        return False, "no_changed_functions_detected"

    relevant_doc_ids = [doc_id for doc_id in changed_funcs if doc_id in corpus_map]
    if not relevant_doc_ids:
        return False, "changed_functions_missing_from_corpus"

    instance_dir = dataset_root / f"swe-bench-verified-function_{instance['instance_id']}"
    qrels_dir = instance_dir / "qrels"
    instance_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    corpus_rows = [
        {"_id": doc_id, "title": "", "text": corpus_map[doc_id]}
        for doc_id in sorted(corpus_map.keys())
    ]
    queries_rows = [{"_id": instance["instance_id"], "text": instance["problem_statement"]}]

    write_jsonl(corpus_rows, instance_dir / "corpus.jsonl")
    write_jsonl(queries_rows, instance_dir / "queries.jsonl")

    with open(qrels_dir / "test.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for doc_id in sorted(relevant_doc_ids):
            writer.writerow([instance["instance_id"], doc_id, 1])

    metadata = {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": instance["base_commit"],
        "num_corpus_docs": len(corpus_rows),
        "num_relevant_docs": len(relevant_doc_ids),
        "relevant_doc_ids": sorted(relevant_doc_ids),
    }
    with open(instance_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"[build] wrote dataset to {instance_dir}", flush=True)
    return True, metadata


def main(
    dataset_dir="datasets",
    num_instances=3,
    hf_split="test",
    max_candidates=20,
):
    dataset_name = normalize_dataset_name("swe-bench-verified")
    dataset_root = Path(dataset_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split=hf_split)

    built = []
    skipped = []
    for instance in ds:
        print(f"[scan] trying instance={instance['instance_id']}", flush=True)
        success, payload = build_instance_dataset(instance, dataset_root)
        if success:
            built.append(payload)
            print(f"[scan] built {len(built)}/{num_instances}: {instance['instance_id']}", flush=True)
        else:
            skipped.append({"instance_id": instance["instance_id"], "reason": payload})
            print(f"[scan] skipped {instance['instance_id']}: {payload}", flush=True)

        if len(built) >= num_instances:
            break
        if len(built) + len(skipped) >= max_candidates:
            break

    report = {
        "dataset_name": dataset_name,
        "requested_instances": num_instances,
        "built_instances": built,
        "skipped_instances": skipped,
    }

    with open(dataset_root / "build_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if len(built) < num_instances:
        raise RuntimeError(
            f"Only built {len(built)} instances under {dataset_root}, fewer than requested {num_instances}."
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
