import os
import json
import logging
import argparse
import re
from time import time
from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from tqdm import tqdm
from transformers import AutoTokenizer
from copy import deepcopy
import torch
import csv
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from statistics import mean, pstdev

METRIC_KEYS = ["time", "Func Acc@5", "Func F1@5", "File Acc@5", "File F1@5"]

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
        
def get_top_docs(results: dict, corpus: dict, task_id: str, topk: int = 100) -> list[str]:
    if task_id not in results: return []
    doc_scores = results[task_id]
    doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    doc_scores_sorted = doc_scores_sorted[:topk]
    doc_code_snippets = [code_id for code_id, score in doc_scores_sorted]
    return doc_code_snippets

def save_beir_results_to_tsv(results, output_file):
    # Create a defaultdict to store the results
    formatted_results = defaultdict(dict)

    # Process the results
    for query_id, doc_scores in results.items():
        for doc_id, score in doc_scores.items():
            formatted_results[query_id][doc_id] = score

    # Write to TSV
    with open(output_file, 'w', newline='') as tsvfile:
        tsvwriter = csv.writer(tsvfile, delimiter='\t')
        tsvwriter.writerow(['Query ID', 'Corpus ID', 'Relevance Score'])

        for query_id, doc_scores in formatted_results.items():
            for doc_id, score in doc_scores.items():
                tsvwriter.writerow([query_id, doc_id, score])

    print(f"Results saved to {output_file}")

def save_beir_results_to_tsv_list(all_results, output_file):
    results_dct = {}
    for result in all_results:
        for k, v in result.items():
            if k in results_dct:
                import pdb;pdb.set_trace()
            results_dct[k] = v 
    save_beir_results_to_tsv(results_dct, output_file) 

def calculate_ranking_metrics(ranked_docs, positives, k=5):
    if not positives:
        return {"Acc@5": 0.0, "F1@5": 0.0}

    top_docs = ranked_docs[:k]
    hits = len(set(top_docs) & positives)

    acc_denom = min(len(positives), k)
    acc = min(hits, acc_denom) / acc_denom

    precision = hits / len(top_docs) if top_docs else 0.0
    recall = hits / len(positives)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"Acc@5": acc, "F1@5": f1}

def function_id_to_file_id(doc_id):
    match = re.match(r"(.+\.py)(/.*)?", doc_id)
    return match.group(1) if match else doc_id

def dedupe_preserving_order(items):
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped

def evaluate_topk(results, qrels, k=5):
    acc_scores = []
    f1_scores = []
    file_acc_scores = []
    file_f1_scores = []

    for query_id, gt_docs in qrels.items():
        if query_id not in results:
            continue

        positives = {doc_id for doc_id, score in gt_docs.items() if score > 0}
        if not positives:
            continue

        sorted_docs = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)
        ranked_funcs = [doc_id for doc_id, _ in sorted_docs]
        func_metrics = calculate_ranking_metrics(ranked_funcs, positives, k)
        acc_scores.append(func_metrics["Acc@5"])
        f1_scores.append(func_metrics["F1@5"])

        positive_files = {function_id_to_file_id(doc_id) for doc_id in positives}
        ranked_files = dedupe_preserving_order(function_id_to_file_id(doc_id) for doc_id in ranked_funcs)
        file_metrics = calculate_ranking_metrics(ranked_files, positive_files, k)
        file_acc_scores.append(file_metrics["Acc@5"])
        file_f1_scores.append(file_metrics["F1@5"])

    if not acc_scores:
        return {
            "Func Acc@5": 0.0,
            "Func F1@5": 0.0,
            "File Acc@5": 0.0,
            "File F1@5": 0.0,
        }

    return {
        "Func Acc@5": sum(acc_scores) / len(acc_scores),
        "Func F1@5": sum(f1_scores) / len(f1_scores),
        "File Acc@5": sum(file_acc_scores) / len(file_acc_scores),
        "File F1@5": sum(file_f1_scores) / len(file_f1_scores),
    }

def format_metrics(title, metrics):
    metric_lines = [title]
    for key in METRIC_KEYS:
        value = metrics[key]
        if key == "time":
            metric_lines.append(f"  {key}: {value:.2f}s")
        else:
            metric_lines.append(f"  {key}: {value:.4f}")
    return "\n".join(metric_lines)

def format_summary(title, summary):
    metric_lines = [title]
    for key in METRIC_KEYS:
        values = summary[key]
        if key == "time":
            metric_lines.append(f"  {key}: mean={values['mean']:.2f}s, std={values['std']:.2f}s")
        else:
            metric_lines.append(f"  {key}: mean={values['mean']:.4f}, std={values['std']:.4f}")
    return "\n".join(metric_lines)

def summarize_runs(run_results):
    return {
        key: {
            "mean": mean(result[key] for result in run_results),
            "std": pstdev(result[key] for result in run_results),
        }
        for key in METRIC_KEYS
    }

def main():
    args.output_file = f"{args.output_dir}/model={args.model}_dataset={args.dataset}_split={args.split}_level={args.level}_eval_mode=default_output.json"
    args.results_file = f"{args.output_dir}/model={args.model}_dataset={args.dataset}_split={args.split}_level={args.level}_eval_mode=default_results.json"
    args.rerank_input_file = f"{args.output_dir}/model={args.model}_dataset={args.dataset}_split={args.split}_level={args.level}_eval_mode=default_rerank_input.tsv"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if os.path.exists(args.results_file):
        os.remove(args.results_file)

    contrast_encoder = models.SentenceBERT()
    contrast_encoder.q_model = SentenceTransformer(args.model, trust_remote_code= True).to(torch.bfloat16)
    contrast_encoder.doc_model = SentenceTransformer(args.model, trust_remote_code= True).to(torch.bfloat16)
    contrast_encoder.q_model.max_seq_length = args.sequence_length
    contrast_encoder.doc_model.max_seq_length = args.sequence_length

    model = DRES(contrast_encoder, batch_size=args.batch_size, corpus_chunk_size=512*9999)
    retriever = EvaluateRetrieval(model, score_function="dot")

    if args.tok:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code = True)
        query_tokens = []
        corpus_tokens = []
    

    if args.num_runs <= 0:
        raise ValueError("--num_runs must be a positive integer")

    if args.dataset.startswith("swe-bench") or args.dataset.startswith("loc-bench"):
        if args.dataset.startswith("swe-bench"):
            if 'lite' in args.dataset.lower():
                swebench = load_dataset("princeton-nlp/SWE-bench_Lite")[args.split]
            elif 'verified' in args.dataset.lower():
                swebench = load_dataset("princeton-nlp/SWE-bench_Verified")[args.split]
            else:
                swebench = load_dataset("princeton-nlp/SWE-bench")[args.split]
        elif args.dataset.startswith("loc-bench"):
            swebench = load_dataset("czlll/Loc-Bench_V1")[args.split]
        if args.split == 'test':
            prefx = f"{args.dataset}"
        else:
            prefx = f"{args.dataset}-{args.split}"
        if args.level == 'file':
            prefx += '_'
        else:
            prefx += f'-{args.level}_'
        
        instance_list = sorted([i for i in os.listdir(args.dataset_dir) if i.startswith(prefx)])
        if len(instance_list) == 0:
            raise FileNotFoundError(
                f"No BEIR instance directories found in {args.dataset_dir!r} with prefix {prefx!r}"
            )
        if args.num_instances is not None:
            if args.num_instances <= 0:
                raise ValueError("--num_instances must be a positive integer")
            instance_list = instance_list[:args.num_instances]
            logging.info("Limiting evaluation to %d instance(s)", len(instance_list))

        run_results = []
        all_runs_eval_results = []
        final_top_docs = [[] for _ in swebench]

        for run_idx in range(1, args.num_runs + 1):
            logging.info("Starting run %d/%d", run_idx, args.num_runs)
            all_eval_results = []
            all_top_docs = [[] for _ in swebench]

            for ins_dir in tqdm(instance_list):
                logging.info("Instance Repo: {}".format(ins_dir))
                # load data and perform retrieval
                corpus, queries, qrels = GenericDataLoader(
                    data_folder=os.path.join(args.dataset_dir, ins_dir)
                ).load(split="test")
                if args.add_prefix and args.query_prefix != '':
                    queries = {k : f'{args.query_prefix}: {v}' for k, v in queries.items()}
                
                logging.info(f"Instance #{ins_dir}: #{len(corpus)} corpus, #{len(queries)} queries")
                
                if args.tok:
                    for v in queries.values():
                        query_tokens.append(tokenizer(v, padding=True, truncation=False, return_tensors="pt")['input_ids'].shape[1])

                    for v in corpus.values():
                        corpus_tokens.append(tokenizer(v['text'], padding=True, truncation=False, return_tensors="pt")['input_ids'].shape[1])

                    continue
                
                rerank_tsv_file = args.rerank_input_file.split('.')
                rerank_tsv_file = f'{rerank_tsv_file[0]}_run{run_idx}_{ins_dir[len(prefx):]}.tsv'

                
                start_time = time()
                if len(queries) == 1:
                    queries.update({"dummy": "dummy"})
                results = retriever.retrieve(corpus, queries)
                save_beir_results_to_tsv(results, rerank_tsv_file)
                if "dummy" in queries:
                    queries.pop("dummy")
                    results.pop("dummy")
                end_time = time()

                # get topk retrieved docs
                if args.dataset.startswith("swe-bench") or args.dataset.startswith("loc-bench"):
                    indices = [i for i,ex in enumerate(swebench) if ex["instance_id"] in queries]
                    for index in indices:
                        instance_id = swebench[index]["instance_id"]
                        all_top_docs[index] = get_top_docs(results, corpus, instance_id)

                # evaluate retrieval results
                if len(qrels) == 0:
                    logging.info("No qrels found for this dataset.")
                    return
                eval_results = {
                    "run": run_idx,
                    "instance": ins_dir,
                    "time": end_time - start_time
                }
                eval_results.update(evaluate_topk(results, qrels, k=5))
                logging.info(format_metrics(f"Run {run_idx} Instance #{ins_dir} Results:", eval_results))
                all_eval_results.append(eval_results)
                
                with open(args.output_file + "_all", "w") as f:
                    json.dump(all_runs_eval_results + [{"run": run_idx, "instances": all_eval_results}], f, indent=2)
            
            if args.tok:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                # Create the plot
                plt.figure(figsize=(10, 6))
                sns.histplot(query_tokens, bins=50, kde=True, stat='probability', label='queries distribution')

                # Set the axis labels
                plt.xlabel('tokens length')
                plt.ylabel('frequency')
                plt.legend()

                # Display the plot
                plt.savefig('queries_tok.png')
                plt.close()
                
                # Create the plot
                plt.figure(figsize=(10, 6))
                sns.histplot(corpus_tokens, bins=50, kde=True, log_scale = (True, False), stat='probability', label='corpus distribution')

                # Set the axis labels
                plt.xlabel('tokens length')
                plt.ylabel('frequency')
                plt.legend()

                # Display the plot
                plt.savefig('corpus_tok.png')
                plt.close()
                
                print('queries mean tokens ', np.mean(query_tokens))
                print('corpus mean tokens ', np.mean(corpus_tokens))
                return

            avg_eval_results = {
                "run": run_idx,
                **{
                    key: sum(e[key] for e in all_eval_results) / len(all_eval_results)
                    for key in METRIC_KEYS
                }
            }
            print(format_metrics(f"Run {run_idx} Average Results:", avg_eval_results))
            run_results.append(avg_eval_results)
            all_runs_eval_results.append({"run": run_idx, "instances": all_eval_results, "average": avg_eval_results})
            final_top_docs = all_top_docs

        if args.dataset.startswith("swe-bench") or args.dataset.startswith("loc-bench"):
            swebench = swebench.add_column("docs", final_top_docs)
            swebench.to_json(args.results_file)

        summary = summarize_runs(run_results)
        print(format_summary(f"Final Summary over {args.num_runs} run(s):", summary))
        with open(args.output_file, "w") as f:
            json.dump({"runs": run_results, "summary": summary}, f, indent=2)
        with open(args.output_file + "_all", "w") as f:
            json.dump(all_runs_eval_results, f, indent=2)

    else:
        raise ValueError(f"`dataset` should start with either 'swe-bench' or 'loc-bench'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="swe-bench-lite",
                        help="Dataset to use for evaluation")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use for evaluation")
    parser.add_argument("--level", type=str, default="function",
                        help="Localization level to use for evaluation")
    parser.add_argument("--model", type=str, default="cornstack/CodeRankEmbed", help="Sentence-BERT model to use")
    parser.add_argument("--tokenizer", type=str, default= "Snowflake/snowflake-arctic-embed-m-long", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for retrieval")
    parser.add_argument("--sequence_length", type=int, default=1024, help="Sequence length for retrieval")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--dataset_dir", type=str, default="datasets",
                        help="Specify the filepath where the dataset is storied")
    parser.add_argument("--num_instances", type=int, default=None,
                        help="Only evaluate the first N BEIR instance directories after sorting")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of full evaluation runs used to compute final mean/std")
    parser.add_argument('--tok', default= False, type = bool)
    parser.add_argument('--add_prefix', default= True, type = bool)
    parser.add_argument('--query_prefix', default= 'Represent this query for searching relevant code', type = str)
    parser.add_argument('--document_prefix', default= '', type = str)
    args = parser.parse_args()

    main()
