import json
import os

from datasets import load_dataset, load_from_disk

from get_repo_structure.get_repo_structure import get_project_structure_from_scratch

LOCAL_DATASET_DIR = "./datasets/Loc-Bench_V1_test"
HF_DATASET_NAME = "czlll/Loc-Bench_V1"
OUTPUT_DIR = "./repo_structures_V1"


def load_V1_dataset():
    if os.path.exists(LOCAL_DATASET_DIR):
        print(f"从本地加载数据: {LOCAL_DATASET_DIR}")
        return load_from_disk(LOCAL_DATASET_DIR)

    print(f"本地数据不存在，从 Hugging Face 加载: {HF_DATASET_NAME}")
    return load_dataset(HF_DATASET_NAME, split="test")


print("加载数据")
loc_bench_data = load_V1_dataset()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 逐个处理数据集中的 bug 实例
for bug in loc_bench_data:
    instance_id = bug['instance_id']
    # 构造 JSON 文件路径
    json_file_path = os.path.join(OUTPUT_DIR, f"{instance_id}.json")

    # 检查文件是否已经存在，如果存在则跳过
    if os.path.exists(json_file_path):
        print(f"文件 {json_file_path} 已存在，跳过该实例。")
        continue

    # 如果文件不存在，则生成项目结构并保存
    print(f"处理实例 {instance_id}...")
    d = get_project_structure_from_scratch(
        bug["repo"], bug["base_commit"], instance_id, "playground"
    )

    # 将项目结构保存到 JSON 文件中
    with open(json_file_path, "w") as json_file:
        json.dump(d, json_file, indent=4, ensure_ascii=False)

    print(f"已保存 {json_file_path}")
