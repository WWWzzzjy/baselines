import os
import re
import subprocess
from pathlib import Path

import datasets

from .log_utils import get_logger

logger = get_logger(__name__)
DATASET_ALIASES = {
    "loc-bench-v1": "czlll/Loc-Bench_V1",
    "loc-bench_v1": "czlll/Loc-Bench_V1",
    "locbench-v1": "czlll/Loc-Bench_V1",
    "locbench_v1": "czlll/Loc-Bench_V1",
}


def load_filter_hf_dataset(args) -> datasets.arrow_dataset.Dataset:

    ret = load_filter_hf_dataset_explicit(
        dataset=args.dataset, filter_instance=args.filter_instance, split=args.split
    )
    # Cannot has both idx_list and idx_range
    assert not (
        hasattr(args, "idx_list") and hasattr(args, "idx_range")
    ), "Cannot has both idx_list and idx_range in arguments"
    if hasattr(args, "idx_list"):
        if args.filter_instance != ".*":
            logger.info(
                (
                    "Running idx_list on a filtered (non-full) dataset."
                    "Please make sure this is expected."
                )
            )
        return ret.select(args.idx_list)
    elif hasattr(args, "idx_range"):
        if args.filter_instance != ".*":
            logger.info(
                (
                    "Running idx_range on a filtered (non-full) dataset."
                    "Please make sure this is expected."
                )
            )
        start_idx = args.idx_range[0]
        end_idx = args.idx_range[1]
        assert start_idx < end_idx, "start_idx should be smaller than end_idx"
        return ret.select(range(start_idx, end_idx))
    else:
        return ret


def load_filter_hf_dataset_explicit(
    dataset: str, filter_instance: str, split: str
) -> datasets.arrow_dataset.Dataset:
    dataset = DATASET_ALIASES.get(dataset.lower(), dataset)
    cache_dir = str(Path.home()) + "/.cache/orcar"
    subprocess.run(f"mkdir -p {cache_dir}", shell=True, check=True)
    dataset_file = f'{dataset.replace("/", "__")}_{split}.json'
    dataset_path = f"{cache_dir}/{dataset_file}"
    if not os.path.exists(dataset_path):
        if dataset == "SWE-bench_common":
            ds_lite: datasets.arrow_dataset.Dataset = datasets.load_dataset(
                "princeton-nlp/SWE-bench_Lite", split=split
            )
            ds_verified: datasets.arrow_dataset.Dataset = datasets.load_dataset(
                "princeton-nlp/SWE-bench_Verified", split=split
            )
            ds = ds_verified.filter(
                input_columns=["instance_id"],
                function=lambda x: x in ds_lite["instance_id"],
            )
        elif dataset == "SWE-bench_Lite_Diff_common":
            ds_lite: datasets.arrow_dataset.Dataset = datasets.load_dataset(
                "princeton-nlp/SWE-bench_Lite", split=split
            )
            ds_verified: datasets.arrow_dataset.Dataset = datasets.load_dataset(
                "princeton-nlp/SWE-bench_Verified", split=split
            )
            ds = ds_lite.filter(
                input_columns=["instance_id"],
                function=lambda x: x not in ds_verified["instance_id"],
            )
        elif dataset == "SWE-bench_Verified_Diff_common":
            ds_lite: datasets.arrow_dataset.Dataset = datasets.load_dataset(
                "princeton-nlp/SWE-bench_Lite", split=split
            )
            ds_verified: datasets.arrow_dataset.Dataset = datasets.load_dataset(
                "princeton-nlp/SWE-bench_Verified", split=split
            )
            ds = ds_verified.filter(
                input_columns=["instance_id"],
                function=lambda x: x not in ds_lite["instance_id"],
            )
        else:
            ds = datasets.load_dataset(dataset, split=split)
        ds.to_json(dataset_path)
    else:
        data_files = {split: dataset_path}
        ds = datasets.load_dataset("json", data_files=data_files, split=split)
    return ds.filter(
        input_columns=["instance_id"],
        function=lambda x: bool(re.match(filter_instance, x)),
    )
