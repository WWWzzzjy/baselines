import json
from datasets import load_dataset

ds = load_dataset("czlll/Loc-Bench_V1", split="test")

gt = {}

for row in ds:
    instance_id = row["instance_id"]
    funcs = (row.get("edit_functions") or []) + (row.get("added_functions") or [])

    entries = []
    for item in funcs:
        file_path, symbol = item.rsplit(":", 1)
        entries.append(f"{file_path}::{symbol}")

    gt[instance_id] = list(dict.fromkeys(entries))

with open("evaluation/gt_v1.json", "w") as f:
    json.dump(gt, f, indent=4)
