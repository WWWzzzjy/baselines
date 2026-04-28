import argparse
import json

import numpy as np
import pandas as pd
from parse_output import (
    ParsedPatch,
    add_missing_top_k_scores,
    build_golden_sets,
    build_model_func_candidates,
    download_golden_data,
    normalize_file_path,
    top_k_scores,
    unique_preserve_order,
)


def parse_output_json(ds_golden: pd.DataFrame, args) -> None:
    output_json = json.load(open(args.output_json))
    artifact_dir: str = args.artifact_dir
    file_path_key: str = args.file_path_key
    file_match = 0
    func_match = 0
    notgen_cnt = 0
    # extractor_file_match = 0
    # extractor_notgen_cnt = 0
    output_dict = dict()
    issues = set(output_json.keys())
    file_prec_list = []
    func_prec_list = []
    file_acc_at_5_list = []
    file_f1_at_5_list = []
    func_acc_at_5_list = []
    func_f1_at_5_list = []
    output_json = json.load(open(args.output_json))
    for inst_id in sorted(issues):
        inst = ds_golden[ds_golden["instance_id"] == inst_id].iloc[0]
        output_dict[inst_id] = dict()

        parsed_patch = ParsedPatch.model_validate_json(inst["parsed_patch"])
        file_set, func_set = build_golden_sets(parsed_patch, inst_id)
        if not file_set:
            print("No file found", inst_id)
            print(parsed_patch)

        model_file_set = set()
        model_func_set = set()
        model_file_ranked = []
        model_func_ranked = []
        # print(inst_id)

        instance_info = output_json[inst_id]
        if "bug_locations" not in instance_info:
            notgen_cnt += 1
            output_dict[inst_id]["status"] = "Json Not Gen"
            add_missing_top_k_scores(
                file_acc_at_5_list,
                file_f1_at_5_list,
                func_acc_at_5_list,
                func_f1_at_5_list,
            )
            continue
        else:
            model_searcher_output = instance_info
            for loc in model_searcher_output["bug_locations"]:
                file_path = normalize_file_path(loc[file_path_key])
                model_file_ranked.append(file_path)
                model_file_set.add(file_path)
                func_candidates = build_model_func_candidates(loc, file_path_key)
                model_func_ranked.extend(func_candidates)
                model_func_set.update(func_candidates)
            output_dict[inst_id]["file"] = dict()
            if file_set.issubset(model_file_set):
                file_match += 1
                output_dict[inst_id]["file"]["file_status"] = "Matched"
            else:
                output_dict[inst_id]["file"]["file_status"] = "Not Matched"
            file_acc_at_5, file_f1_at_5 = top_k_scores(file_set, model_file_ranked)
            file_acc_at_5_list.append(file_acc_at_5)
            file_f1_at_5_list.append(file_f1_at_5)
            file_prec_list.append(
                len(file_set.intersection(model_file_set)) / len(model_file_set)
                if len(model_file_set)
                else 1
            )

            output_dict[inst_id]["file"]["golden"] = list(file_set)
            output_dict[inst_id]["file"]["model"] = unique_preserve_order(
                model_file_ranked
            )
            output_dict[inst_id]["file"]["acc@5"] = file_acc_at_5
            output_dict[inst_id]["file"]["f1@5"] = file_f1_at_5

            output_dict[inst_id]["func"] = dict()
            if func_set.issubset(model_func_set):
                func_match += 1

                output_dict[inst_id]["func"]["func_status"] = "Matched"
            else:
                output_dict[inst_id]["func"]["func_status"] = "Not Matched"
            output_dict[inst_id]["func"]["golden"] = list(func_set)
            output_dict[inst_id]["func"]["model"] = unique_preserve_order(
                model_func_ranked
            )
            func_acc_at_5, func_f1_at_5 = top_k_scores(func_set, model_func_ranked)
            output_dict[inst_id]["func"]["acc@5"] = func_acc_at_5
            output_dict[inst_id]["func"]["f1@5"] = func_f1_at_5
            func_acc_at_5_list.append(func_acc_at_5)
            func_f1_at_5_list.append(func_f1_at_5)
            func_prec_list.append(
                len(func_set.intersection(model_func_set)) / len(model_func_set)
                if len(model_func_set)
                else 1
            )

    total_cnt = len(issues)
    print(f"File match: {file_match}/{total_cnt}, {file_match / total_cnt * 100:.2f}%")
    print(
        f"Mean File Acc@5: {np.mean(file_acc_at_5_list) * 100:.2f}%, Std File Acc@5: {np.std(file_acc_at_5_list) * 100:.2f}%"
    )
    print(
        f"Mean File F1@5: {np.mean(file_f1_at_5_list) * 100:.2f}%, Std File F1@5: {np.std(file_f1_at_5_list) * 100:.2f}%"
    )
    print(
        f"Mean File Precision: {np.mean(file_prec_list) * 100:.2f}%, Std File Precision: {np.std(file_prec_list) * 100:.2f}%"
    )
    print(
        f"Function Match: {func_match}/{total_cnt}, {func_match / total_cnt * 100:.2f}%"
    )
    print(
        f"Mean Function Acc@5: {np.mean(func_acc_at_5_list) * 100:.2f}%, Std Function Acc@5: {np.std(func_acc_at_5_list) * 100:.2f}%"
    )
    print(
        f"Mean Function F1@5: {np.mean(func_f1_at_5_list) * 100:.2f}%, Std Function F1@5: {np.std(func_f1_at_5_list) * 100:.2f}%"
    )
    print(
        f"Mean Function Precision: {np.mean(func_prec_list) * 100:.2f}%, Std Function Precision: {np.std(func_prec_list) * 100:.2f}%"
    )
    print(
        f"Json not gen: {notgen_cnt}/{total_cnt}, {notgen_cnt / total_cnt * 100:.2f}%"
    )
    output_path = f"{artifact_dir}/assets/orcar_parsed_output.json"
    with open(output_path, "w") as handle:
        json.dump(output_dict, handle, indent=4)
    print(f"Parsed output dumped to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--artifact_dir",
        default="./artifact",
        help=f"The directory of the artifact folder",
    )
    parser.add_argument(
        "-l",
        "--output_json",
        default="./evaluation/output.json",
        help=f"The file path of the output json",
    )
    parser.add_argument(
        "-f",
        "--file_path_key",
        default="file_path",
        help=f"The directory of the output dir(agent's output)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="lite",
        help=f"The dataset to use",
    )
    args = parser.parse_args()
    ds_golden = download_golden_data(
        artifact_dir=args.artifact_dir, dataset=args.dataset
    )
    parse_output_json(ds_golden, args)


if __name__ == "__main__":
    main()
