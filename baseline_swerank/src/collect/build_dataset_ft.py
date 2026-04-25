#!/usr/bin/env python3

import argparse
import glob
import json
import os
import random

from tqdm import tqdm
from datetime import datetime


def main(instances_path: str, output_dir: str, seed: int):
    """
    Combine all non-eval task instances into a single fine tuning dataset

    Args:
        instances_path (str): Path to directory containing all candidate task instances
        output_path (str): Path to save output fine tuning dataset to
        eval_path (str): Path to directory containing all eval task instances
        seed (int): Random seed
    """
    # Define output file name
    random.seed(seed)
    SWE_PRS_FT_DATASET = "SWE_PRS_FT_DATASET.jsonl"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    destination = os.path.join(output_dir, SWE_PRS_FT_DATASET)
    total_insts, total_repos = 0, 0


    # Create fine tuning dataset
    with open(destination, "w") as f_out:
        for dataset_path in tqdm(
            glob.glob(os.path.join(instances_path, "*-task-instances.jsonl.all"))
        ):
            total_repos += 1
            with open(dataset_path) as f:
                lines = f.readlines()

                # Shuffle lines
                random.shuffle(lines)

                # Keep 500 lines per dataset
                for line in lines:
                    line = json.loads(line)
                    if "test_patch" in line:
                        del line["test_patch"]
                    f_out.write(json.dumps(line) + "\n")
                    total_insts += 1

    print(
        f"Fine tuning dataset saved to {destination} ({total_insts} instances from {total_repos} repos)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances_path",
        type=str,
        help="Path to directory containing all candidate task instances",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(**vars(args))
