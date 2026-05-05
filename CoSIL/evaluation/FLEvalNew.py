import argparse
import json


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def top_k_accuracy(gt, preds, k):
    if not gt:
        return False
    return set(gt).issubset(set(preds[:k]))


def extract_token_count(obj):
    if isinstance(obj, dict):
        total = 0
        if "prompt_tokens" in obj or "completion_tokens" in obj:
            total += obj.get("prompt_tokens", 0) + obj.get("completion_tokens", 0)
        for value in obj.values():
            total += extract_token_count(value)
        return total
    if isinstance(obj, list):
        return sum(extract_token_count(item) for item in obj)
    return 0


def extract_inference_time(obj):
    if isinstance(obj, dict):
        total = obj.get("inference_time", 0)
        for value in obj.values():
            total += extract_inference_time(value)
        return total
    if isinstance(obj, list):
        return sum(extract_inference_time(item) for item in obj)
    return 0


def acc_at_k(gt, preds, k):
    return 1.0 if top_k_accuracy(gt, preds, k) else 0.0


def f1_at_k(gt, preds, k):
    if not gt:
        return 0
    hits = sum(1 for item in preds[:k] if item in gt)
    precision = hits / k
    recall = hits / len(gt)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def average_precision(gt, preds):
    """
    计算单个查询的AP（平均精度）
    """
    if not gt:
        return 0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(preds):
        if p in gt:
            num_hits += 1.0
            score += num_hits / (i + 1)
    return score / len(gt)


def reciprocal_rank(gt, preds):
    """
    计算单个查询的倒数排名（Reciprocal Rank）
    """
    for i, p in enumerate(preds):
        if p in gt:
            return 1.0 / (i + 1)
    return 0.0


def parse_gt_methods(gt_entries):
    """
    解析ground truth中的条目，并统一处理为文件级别和函数级别的定位。
    """
    files, methods = set(), set()

    for entry in gt_entries:
        parts = entry.split('::')

        if len(parts) == 2:  # File::Method 或 File::Class
            file_name, method_or_class = parts
            files.add(file_name)
            methods.add(method_or_class)

        elif len(parts) == 1:  # File
            files.add(parts[0])

    return files, methods


def extract_predicted_methods(found_related_locs):
    """
    从 found_related_locs 中提取预测的类和函数名称。
    """
    predicted_methods = []
    for sublist in found_related_locs:
        for loc in sublist:
            for entry in loc.split('\n'):
                if 'function:' in entry or 'class:' in entry:
                    try:
                        predicted_methods.append(entry.split(': ')[1])
                    except:
                        pass
    return predicted_methods


def construct_pred_func(file_locs, func_locs):
    final_funcs = []
    # for file_loc in file_locs:
    #     final_funcs.append(func_locs.get(file_loc, []))
    for key, item in func_locs.items():
        final_funcs.append(item)
    return final_funcs


def evaluate_accuracy(loc_outputs, gt_data, verbose=True):
    # 文件级TOPN统计
    top1_file_correct = 0
    top3_file_correct = 0
    top5_file_correct = 0

    # 函数级TOPN统计
    top1_func_correct = 0
    top3_func_correct = 0
    top5_func_correct = 0
    file_acc5_sum = 0.0
    func_acc5_sum = 0.0
    file_f1_5_sum = 0.0
    func_f1_5_sum = 0.0

    # 初始化MAP和MRR的累加器
    file_AP_sum = 0.0
    file_RR_sum = 0.0
    func_AP_sum = 0.0
    func_RR_sum = 0.0

    # 只用实际跑出的、有 ground truth 的实例作为分母。
    total_instances = sum(
        1 for loc_output in loc_outputs if loc_output.get("instance_id") in gt_data
    )
    if total_instances == 0:
        return {
            "file_level": {"ACC@5": 0, "F1@5": 0},
            "function_level": {"ACC@5": 0, "F1@5": 0},
            "tokens": {"total": 0, "average": 0},
            "inference_time": {"total": 0, "average": 0},
        }

    empty_count = 0
    total_tokens = 0
    total_inference_time = 0.0

    # 对每个实例进行评估
    for loc_output in loc_outputs:
        instance_id = loc_output['instance_id']
        if instance_id not in gt_data:
            continue

        total_tokens += extract_token_count(loc_output)
        total_inference_time += extract_inference_time(loc_output)
        if verbose:
            print(instance_id)
        predicted_files = loc_output['found_files'][:5]
        if not predicted_files:
            empty_count += 1
            continue
        pred_funcs = construct_pred_func(predicted_files, loc_output.get('found_related_locs', {}))[:5]
        predicted_methods = extract_predicted_methods(pred_funcs)
        if verbose:
            print(f"predicted_files:{predicted_files}, predicted_methods:{predicted_methods}")

        gt_files, gt_methods = parse_gt_methods(gt_data[instance_id])
        if verbose:
            print(f"gt_files:{gt_files}, gt_methods:{gt_methods}")

        # 计算TOPN准确率（文件级）
        if top_k_accuracy(gt_files, predicted_files, 1):
            top1_file_correct += 1
        if top_k_accuracy(gt_files, predicted_files, 3):
            top3_file_correct += 1
        if top_k_accuracy(gt_files, predicted_files, 5):
            top5_file_correct += 1
        file_acc5_sum += acc_at_k(gt_files, predicted_files, 5)
        file_f1_5_sum += f1_at_k(gt_files, predicted_files, 5)

        # 计算TOPN准确率（函数级，包括类和方法）
        if top_k_accuracy(gt_methods, predicted_methods, 1):
            top1_func_correct += 1
        if top_k_accuracy(gt_methods, predicted_methods, 3):
            top3_func_correct += 1
        if top_k_accuracy(gt_methods, predicted_methods, 5):
            top5_func_correct += 1
        func_acc5_sum += acc_at_k(gt_methods, predicted_methods, 5)
        func_f1_5_sum += f1_at_k(gt_methods, predicted_methods, 5)

        # 计算MAP和MRR
        ap_file = average_precision(gt_files, predicted_files)
        rr_file = reciprocal_rank(gt_files, predicted_files)
        ap_func = average_precision(gt_methods, predicted_methods)
        rr_func = reciprocal_rank(gt_methods, predicted_methods)

        file_AP_sum += ap_file
        file_RR_sum += rr_file
        func_AP_sum += ap_func
        func_RR_sum += rr_func

        if verbose:
            print(top1_file_correct, top3_file_correct, top5_file_correct)
            print(top1_func_correct, top3_func_correct, top5_func_correct)

    # 计算TOPN准确率百分比
    top1_file_accuracy = top1_file_correct / total_instances * 100
    top3_file_accuracy = top3_file_correct / total_instances * 100
    top5_file_accuracy = top5_file_correct / total_instances * 100

    top1_func_accuracy = top1_func_correct / total_instances * 100
    top3_func_accuracy = top3_func_correct / total_instances * 100
    top5_func_accuracy = top5_func_correct / total_instances * 100
    file_acc5 = file_acc5_sum / total_instances * 100
    func_acc5 = func_acc5_sum / total_instances * 100
    file_f1_5 = file_f1_5_sum / total_instances * 100
    func_f1_5 = func_f1_5_sum / total_instances * 100

    # 计算MAP和MRR（乘以100转化为百分比）
    map_file = file_AP_sum / total_instances * 100
    mrr_file = file_RR_sum / total_instances * 100
    map_func = func_AP_sum / total_instances * 100
    mrr_func = func_RR_sum / total_instances * 100

    empty_percent = empty_count / total_instances * 100
    avg_tokens = total_tokens / total_instances
    avg_inference_time = total_inference_time / total_instances

    return {
        "file_level": {
            "ACC@5": file_acc5,
            "F1@5": file_f1_5,
        },
        "function_level": {
            "ACC@5": func_acc5,
            "F1@5": func_f1_5,
        },
        "tokens": {
            "total": total_tokens,
            "average": avg_tokens,
        },
        "inference_time": {
            "total": total_inference_time,
            "average": avg_inference_time,
        }
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="lite",
        choices=["lite", "verified", "v1"],
        help="Current supported dataset for evaluation",
    )
    parser.add_argument("--loc_file", type=str, default="loc_outputs.jsonl")
    args = parser.parse_args()
    loc_outputs = load_jsonl(args.loc_file)

    if args.dataset == "lite":
        gt_data = load_json('gt.json')
    elif args.dataset == "verified":
        gt_data = load_json('gt_verified.json')
    else:
        gt_data = load_json('gt_v1.json')
    print(len(loc_outputs))

    # 进行评估
    accuracy_results = evaluate_accuracy(loc_outputs, gt_data)

    # 输出评估结果
    print("File-level accuracy:")
    for k, v in accuracy_results['file_level'].items():
        print(f"{k}: {v:.2f}%")

    print("\nFunction-level accuracy:")
    for k, v in accuracy_results['function_level'].items():
        print(f"{k}: {v:.2f}%")

    print("\nToken usage:")
    print(f"total: {accuracy_results['tokens']['total']}")
    print(f"average: {accuracy_results['tokens']['average']:.2f}")

    print("\nInference time:")
    print(f"total: {accuracy_results['inference_time']['total']:.2f}s")
    print(f"average: {accuracy_results['inference_time']['average']:.2f}s")
