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


def summarize(values: list[float]):
    if not values:
        return {'mean': 0.0, 'std': 0.0}
    return {
        'mean': round(mean(values), 4),
        'std': round(pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def has_nonempty_predictions(row: dict) -> bool:
    for key in ('found_files', 'found_modules', 'found_entities', 'raw_output_loc'):
        value = row.get(key, [])
        if isinstance(value, list) and value:
            if any(item for item in value):
                return True
    return False


def evaluate_single_run(loc_rows, file_gt, func_gt, usage_map):
    per_instance = []
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
        per_instance.append(
            {
                'instance_id': instance_id,
                'success': success,
                'files_acc@5': round(file_metrics['acc'], 4),
                'files_f1': round(file_metrics['f1'], 4),
                'func_acc@3': round(func_metrics['acc'], 4),
                'func_f1': round(func_metrics['f1'], 4),
                '#tokens': usage['total_tokens'],
                'time': round(usage['time'], 4),
            }
        )

    successful_rows = [row for row in per_instance if row['success']]
    aggregate = {
        'count': len(successful_rows),
        'total_count': len(per_instance),
        'failed_count': len(per_instance) - len(successful_rows),
        'files_acc@5': summarize([row['files_acc@5'] for row in successful_rows]),
        'files_f1': summarize([row['files_f1'] for row in successful_rows]),
        'func_acc@3': summarize([row['func_acc@3'] for row in successful_rows]),
        'func_f1': summarize([row['func_f1'] for row in successful_rows]),
        '#tokens': summarize([row['#tokens'] for row in successful_rows]),
        'time': summarize([row['time'] for row in successful_rows]),
    }
    return {'per_instance': per_instance, 'aggregate': aggregate}


def summarize_run_metric(run_payloads, metric_name):
    values = []
    for payload in run_payloads:
        values.append(payload['aggregate'][metric_name]['mean'])
    return summarize(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_file', required=True, help='Merged localization results JSONL.')
    parser.add_argument('--traj_file', default='', help='Trajectory JSONL with token/time usage.')
    parser.add_argument('--dataset', default='princeton-nlp/SWE-bench_Verified')
    parser.add_argument('--split', default='test')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of full-dataset runs.')
    parser.add_argument('--runs_root', default='', help='Root directory containing run_1, run_2, ... subdirectories.')
    parser.add_argument('--output_file', default='')
    args = parser.parse_args()

    if args.num_runs <= 1:
        loc_rows = load_jsonl(args.loc_file)
        selected_ids = {row['instance_id'] for row in loc_rows}
        file_gt, func_gt = build_gt_maps(args.dataset, args.split, selected_ids)
        usage_map = load_usage_map(args.traj_file) if args.traj_file else {}
        payload = evaluate_single_run(loc_rows, file_gt, func_gt, usage_map)
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

        for run_idx, loc_rows, run_traj_file in run_rows:
            usage_map = load_usage_map(run_traj_file) if run_traj_file else {}
            run_payload = evaluate_single_run(loc_rows, file_gt, func_gt, usage_map)
            run_payloads.append(
                {
                    'run_id': run_idx,
                    **run_payload,
                }
            )

        aggregate = {
            'num_runs': args.num_runs,
            'count': summarize([payload['aggregate']['count'] for payload in run_payloads]),
            'total_count': summarize([payload['aggregate']['total_count'] for payload in run_payloads]),
            'failed_count': summarize([payload['aggregate']['failed_count'] for payload in run_payloads]),
            'files_acc@5': summarize_run_metric(run_payloads, 'files_acc@5'),
            'files_f1': summarize_run_metric(run_payloads, 'files_f1'),
            'func_acc@3': summarize_run_metric(run_payloads, 'func_acc@3'),
            'func_f1': summarize_run_metric(run_payloads, 'func_f1'),
            '#tokens': summarize_run_metric(run_payloads, '#tokens'),
            'time': summarize_run_metric(run_payloads, 'time'),
        }
        payload = {'per_run': run_payloads, 'aggregate': aggregate}

    print(json.dumps(payload, indent=2))

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2)


if __name__ == '__main__':
    main()
