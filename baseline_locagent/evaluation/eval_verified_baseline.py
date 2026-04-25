import argparse
import json
import os
from statistics import mean, pstdev

from datasets import load_dataset

from util.benchmark.parse_patch import get_oracle_filenames
from util.utils import load_jsonl


def normalize_function_predictions(found_entities):
    normalized = []
    for entity in found_entities:
        if entity.endswith('.__init__'):
            entity = entity[: -len('.__init__')]
        if entity not in normalized:
            normalized.append(entity)
    return normalized


def build_gt_from_patch(instance: dict):
    file_targets = sorted(get_oracle_filenames(instance['patch']))
    func_targets = []

    for row in instance.get('gt_file_changes', []) or []:
        file_name = row.get('file')
        changes = row.get('changes', {})
        for entity in changes.get('edited_entities', []) or []:
            if entity.endswith('.__init__'):
                entity = entity[: -len('.__init__')]
            if entity not in func_targets:
                func_targets.append(entity)

        if not file_name:
            continue
        for entity in changes.get('added_entities', []) or []:
            if entity.endswith('.__init__'):
                entity = entity[: -len('.__init__')]
            if entity not in func_targets:
                func_targets.append(entity)

    return file_targets, func_targets


def build_gt_maps(dataset_name: str, split: str, selected_ids: set[str]):
    dataset = load_dataset(dataset_name, split=split)
    file_gt = {}
    func_gt = {}
    for instance in dataset:
        instance_id = instance['instance_id']
        if selected_ids and instance_id not in selected_ids:
            continue
        if 'edit_functions' in instance:
            file_targets = []
            func_targets = []
            for func in instance['edit_functions']:
                file_name = func.split(':')[0]
                func_name = func.split(':')[-1]
                if func_name.endswith('.__init__'):
                    func_name = func_name[: -len('.__init__')]
                full_func_name = f'{file_name}:{func_name}'
                if file_name not in file_targets:
                    file_targets.append(file_name)
                if full_func_name not in func_targets:
                    func_targets.append(full_func_name)
        else:
            file_targets, func_targets = build_gt_from_patch(instance)
        file_gt[instance_id] = file_targets
        func_gt[instance_id] = func_targets
    return file_gt, func_gt


def calc_metrics(preds: list[str], gts: list[str], k: int):
    clipped_preds = preds[:k]
    gt_count = min(len(gts), k)
    hits = sum(1 for item in clipped_preds if item in gts)
    acc = 1.0 if hits == gt_count else 0.0
    precision = hits / k if k else 0.0
    recall = hits / gt_count if gt_count else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def load_usage_map(traj_file: str):
    usage_map = {}
    if not os.path.exists(traj_file):
        return usage_map
    for row in load_jsonl(traj_file):
        trajs = row.get('loc_trajs', {}).get('trajs', [])
        usage = row.get('usage', {})
        usage_map[row['instance_id']] = {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0),
            'time': sum(traj.get('time', 0.0) for traj in trajs),
        }
    return usage_map


def load_prebuild_time(prebuild_time_file: str) -> float:
    if not prebuild_time_file or not os.path.exists(prebuild_time_file):
        return 0.0
    with open(prebuild_time_file, 'r', encoding='utf-8') as file:
        payload = json.load(file)
    return float(payload.get('total_seconds', 0.0))


def summarize(values: list[float]):
    if not values:
        return {'mean': 0.0, 'std': 0.0}
    return {
        'mean': round(mean(values), 4),
        'std': round(pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def summarize_mean(values: list[float]):
    if not values:
        return 0.0
    return round(mean(values), 4)


def has_content(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(has_content(item) for item in value)
    return bool(value)


def has_nonempty_predictions(row: dict) -> bool:
    for key in ('found_files', 'found_modules', 'found_entities', 'raw_output_loc'):
        if has_content(row.get(key)):
            return True
    return False


def evaluate_single_run(loc_rows, file_gt, func_gt, usage_map, prebuild_seconds=0.0):
    per_instance = []
    prebuild_time_per_instance = prebuild_seconds / len(loc_rows) if loc_rows else 0.0
    for row in loc_rows:
        instance_id = row['instance_id']
        file_metrics = calc_metrics(row.get('found_files', []), file_gt.get(instance_id, []), 5)
        func_preds = normalize_function_predictions(row.get('found_entities', []))
        func_metrics = calc_metrics(func_preds, func_gt.get(instance_id, []), 3)
        usage = usage_map.get(
            instance_id,
            {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'time': 0.0},
        )
        success = has_nonempty_predictions(row)
        localize_time = usage['time']
        total_time = localize_time + prebuild_time_per_instance
        per_instance.append(
            {
                'instance_id': instance_id,
                'success': success,
                'files_acc@5': round(file_metrics['acc'], 4),
                'files_f1': round(file_metrics['f1'], 4),
                'func_acc@3': round(func_metrics['acc'], 4),
                'func_f1': round(func_metrics['f1'], 4),
                '#tokens': usage['total_tokens'],
                'localize_time': round(localize_time, 4),
                'prebuild_time': round(prebuild_time_per_instance, 4),
                'time': round(total_time, 4),
            }
        )

    successful_rows = [row for row in per_instance if row['success']]
    aggregate = {
        'count': len(successful_rows),
        'total_count': len(per_instance),
        'failed_count': len(per_instance) - len(successful_rows),
        'files_acc@5': summarize_mean([row['files_acc@5'] for row in successful_rows]),
        'files_f1': summarize_mean([row['files_f1'] for row in successful_rows]),
        'func_acc@3': summarize_mean([row['func_acc@3'] for row in successful_rows]),
        'func_f1': summarize_mean([row['func_f1'] for row in successful_rows]),
        '#tokens': summarize_mean([row['#tokens'] for row in successful_rows]),
        'localize_time': summarize_mean([row['localize_time'] for row in successful_rows]),
        'prebuild_time': summarize_mean([row['prebuild_time'] for row in successful_rows]),
        'time': summarize_mean([row['time'] for row in successful_rows]),
    }
    return {'per_instance': per_instance, 'aggregate': aggregate}


def summarize_run_metric(run_payloads, metric_name):
    values = []
    for payload in run_payloads:
        values.append(payload['aggregate'][metric_name])
    return summarize(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_file', required=True, help='Merged localization results JSONL.')
    parser.add_argument('--traj_file', default='', help='Trajectory JSONL with token/time usage.')
    parser.add_argument('--dataset', default='princeton-nlp/SWE-bench_Verified')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of full-dataset runs.')
    parser.add_argument('--runs_root', default='', help='Root directory containing run_1, run_2, ... subdirectories.')
    parser.add_argument('--prebuild_time_file', default='', help='JSON file with offline graph/BM25 prebuild time.')
    parser.add_argument('--output_file', default='')
    args = parser.parse_args()

    if args.num_runs <= 1:
        loc_rows = load_jsonl(args.loc_file)
        selected_ids = {row['instance_id'] for row in loc_rows}
        file_gt, func_gt = build_gt_maps(args.dataset, args.split, selected_ids)
        usage_map = load_usage_map(args.traj_file) if args.traj_file else {}
        prebuild_seconds = load_prebuild_time(args.prebuild_time_file)
        payload = evaluate_single_run(loc_rows, file_gt, func_gt, usage_map, prebuild_seconds)
    else:
        if not args.runs_root:
            raise ValueError('--runs_root is required when --num_runs > 1')

        run_payloads = []
        selected_ids = set()
        run_rows = []
        for run_idx in range(1, args.num_runs + 1):
            run_loc_file = os.path.join(args.runs_root, f'run_{run_idx}', 'location', os.path.basename(args.loc_file))
            run_traj_file = os.path.join(args.runs_root, f'run_{run_idx}', 'location', os.path.basename(args.traj_file)) if args.traj_file else ''
            loc_rows = load_jsonl(run_loc_file)
            run_rows.append((run_idx, loc_rows, run_traj_file))
            selected_ids.update(row['instance_id'] for row in loc_rows)

        file_gt, func_gt = build_gt_maps(args.dataset, args.split, selected_ids)

        shared_prebuild_seconds = load_prebuild_time(args.prebuild_time_file)
        per_run_shared_prebuild = shared_prebuild_seconds / args.num_runs if args.num_runs else 0.0

        for run_idx, loc_rows, run_traj_file in run_rows:
            usage_map = load_usage_map(run_traj_file) if run_traj_file else {}
            run_prebuild_file = os.path.join(args.runs_root, f'run_{run_idx}', 'prebuild_times.json')
            run_prebuild_seconds = load_prebuild_time(run_prebuild_file)
            if not run_prebuild_seconds:
                run_prebuild_seconds = per_run_shared_prebuild
            run_payload = evaluate_single_run(
                loc_rows, file_gt, func_gt, usage_map, run_prebuild_seconds
            )
            run_payloads.append(
                {
                    'run_id': run_idx,
                    'aggregate': run_payload['aggregate'],
                }
            )

        aggregate = {
            'num_runs': args.num_runs,
            'count': summarize_run_metric(run_payloads, 'count'),
            'total_count': summarize_run_metric(run_payloads, 'total_count'),
            'failed_count': summarize_run_metric(run_payloads, 'failed_count'),
            'files_acc@5': summarize_run_metric(run_payloads, 'files_acc@5'),
            'files_f1': summarize_run_metric(run_payloads, 'files_f1'),
            'func_acc@3': summarize_run_metric(run_payloads, 'func_acc@3'),
            'func_f1': summarize_run_metric(run_payloads, 'func_f1'),
            '#tokens': summarize_run_metric(run_payloads, '#tokens'),
            'localize_time': summarize_run_metric(run_payloads, 'localize_time'),
            'prebuild_time': summarize_run_metric(run_payloads, 'prebuild_time'),
            'time': summarize_run_metric(run_payloads, 'time'),
        }
        payload = {'runs': run_payloads, 'aggregate': aggregate}

    print(json.dumps(payload, indent=2))

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2)


if __name__ == '__main__':
    main()
