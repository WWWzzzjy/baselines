import os
import json
import random
import logging
import pathlib
import argparse
from time import time
from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from tqdm import tqdm
from transformers import AutoTokenizer
import csv
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import torch


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower().replace("_", "-")
    aliases = {
        "swe-bench-verified": "swe-bench-verified",
        "swebench-verified": "swe-bench-verified",
        "swe-bench-lite": "swe-bench-lite",
        "swebench-lite": "swe-bench-lite",
        "loc-bench": "loc-bench",
        "locbench": "loc-bench",
    }
    return aliases.get(normalized, normalized)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def load_jsonl(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(file_path: str, row: dict) -> None:
    with open(file_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def instance_id_from_dir(instance_dir: str, prefix: str) -> str:
    if instance_dir.startswith(prefix):
        return instance_dir[len(prefix):]
    return instance_dir


def build_retriever_output_paths(results_file: str, output_file: str) -> dict:
    results_path = pathlib.Path(results_file)
    output_path = pathlib.Path(output_file)

    if results_path.stem.endswith("_results"):
        results_per_instance_name = results_path.stem + "_per_instance.json"
        results_checkpoint_name = results_path.stem + "_checkpoint.jsonl"
    else:
        results_per_instance_name = results_path.stem + "_per_instance.json"
        results_checkpoint_name = results_path.stem + "_checkpoint.jsonl"

    if output_path.stem.endswith("_summary"):
        eval_summary_name = output_path.name
        eval_per_instance_name = output_path.stem.replace("_summary", "_per_instance") + ".json"
        eval_checkpoint_name = output_path.stem.replace("_summary", "_checkpoint") + ".jsonl"
    else:
        eval_summary_name = output_path.stem + "_summary.json"
        eval_per_instance_name = output_path.stem + "_per_instance.json"
        eval_checkpoint_name = output_path.stem + "_checkpoint.jsonl"

    return {
        "results": str(results_path),
        "results_per_instance": str(results_path.with_name(results_per_instance_name)),
        "results_checkpoint": str(results_path.with_name(results_checkpoint_name)),
        "eval_summary": str(output_path.with_name(eval_summary_name)),
        "eval_per_instance": str(output_path.with_name(eval_per_instance_name)),
        "eval_checkpoint": str(output_path.with_name(eval_checkpoint_name)),
    }


def sync_results_files(dataset, docs_by_instance, eval_rows, results_file, output_file):
    paths = build_retriever_output_paths(results_file, output_file)

    completed_dataset_rows = []
    for row in dataset:
        instance_id = row["instance_id"]
        if instance_id not in docs_by_instance:
            continue
        row = dict(row)
        row["docs"] = docs_by_instance[instance_id]
        completed_dataset_rows.append(row)

    with open(paths["results"], "w", encoding="utf-8") as handle:
        json.dump(completed_dataset_rows, handle, ensure_ascii=False, indent=2)

    with open(paths["results_per_instance"], "w", encoding="utf-8") as handle:
        json.dump(completed_dataset_rows, handle, ensure_ascii=False, indent=2)

    if eval_rows:
        avg_eval_results = {}
        for key, value in eval_rows[0].items():
            if isinstance(value, dict):
                avg_eval_results.update({
                    metric_key: sum(row[key][metric_key] for row in eval_rows) / len(eval_rows)
                    for metric_key in value.keys()
                })
            elif isinstance(value, float):
                avg_eval_results[key] = sum(row[key] for row in eval_rows) / len(eval_rows)
            else:
                raise ValueError(f"Unsupported eval value type for key={key}: {type(value)}")
    else:
        avg_eval_results = {}

    with open(paths["eval_summary"], "w", encoding="utf-8") as handle:
        json.dump(avg_eval_results, handle, ensure_ascii=False, indent=2)

    with open(paths["eval_per_instance"], "w", encoding="utf-8") as handle:
        json.dump(eval_rows, handle, ensure_ascii=False, indent=2)

    logging.info(
        "Checkpointed %d instances to %s, %s, %s, and %s",
        len(completed_dataset_rows),
        paths["results"],
        paths["eval_summary"],
        paths["results_per_instance"],
        paths["eval_per_instance"],
    )

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)
        
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

def main():
    args.dataset = normalize_dataset_name(args.dataset)

    args.model_name_or_path = args.model

    contrast_encoder = models.SentenceBERT()
    contrast_encoder.q_model = SentenceTransformer(args.model, trust_remote_code= True).to(torch.bfloat16)
    contrast_encoder.doc_model = SentenceTransformer(args.model, trust_remote_code= True).to(torch.bfloat16)
    contrast_encoder.q_model.max_seq_length = args.sequence_length
    contrast_encoder.doc_model.max_seq_length = args.sequence_length
    model = DRES(contrast_encoder, batch_size=args.batch_size, corpus_chunk_size=512*9999)
    retriever = EvaluateRetrieval(model, score_function="dot")       

    if args.dataset == "swe-bench-lite":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite")[args.split]
    elif args.dataset == "swe-bench-verified":
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified")[args.split]
    elif args.dataset == "loc-bench":
        dataset = load_dataset("czlll/Loc-Bench_V1")[args.split]
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    output_paths = build_retriever_output_paths(args.results_file, args.output_file)
    results_sidecar = output_paths["results_checkpoint"]
    eval_sidecar = output_paths["eval_checkpoint"]

    completed_rows = load_jsonl(results_sidecar)
    completed_eval_rows = load_jsonl(eval_sidecar)
    docs_by_instance = {row["instance_id"]: row["docs"] for row in completed_rows}
    completed_ids = set(docs_by_instance.keys())

    if completed_ids:
        logging.info("Resuming retriever run with %d completed instances from checkpoints.", len(completed_ids))
        sync_results_files(dataset, docs_by_instance, completed_eval_rows, args.results_file, args.output_file)

    if args.split == 'test':
        prefx = f"{args.dataset}" 
    else: 
        prefx = f"{args.dataset}-{args.split}"
    if args.level == 'file':
        prefx += '_'
    else:
        prefx += f'-{args.level}_'

    instance_list = [i for i in os.listdir(args.dataset_dir) if i.startswith(prefx)]
    if args.dataset == "loc-bench":
        loc_bench_ids = [prefx + i for i in dataset['instance_id']]
        instance_list = [i for i in instance_list if i in loc_bench_ids]
        assert len(instance_list) == len(loc_bench_ids)

    for ins_dir in tqdm(instance_list):
        instance_id = instance_id_from_dir(ins_dir, prefx)
        if instance_id in completed_ids:
            logging.info("Skipping completed instance: %s", ins_dir)
            continue

        logging.info("Instance Repo: {}".format(ins_dir))
        # load data and perform retrieval
        corpus, queries, qrels = GenericDataLoader(
            data_folder=os.path.join(args.dataset_dir, ins_dir)
        ).load(split="test")
        if args.add_prefix:
            if "SweRankEmbed-small".lower() in args.model.lower():
                query_prefix = "Represent this query for searching relevant code"
            elif "SweRankEmbed-large".lower() in args.model.lower():
                query_prefix = "Instruct: Given a github issue, identify the code that needs to be changed to fix the issue.\nQuery"
            else:
                raise ValueError(f"Model {args.model} not supported, make sure to define the query prefix")
            
            queries = {k : f'{query_prefix}: {v}' for k, v in queries.items()}
        logging.info(f"Instance #{ins_dir}: #{len(corpus)} corpus, #{len(queries)} queries")
            
        start_time = time()
        if len(queries) == 1:
            queries.update({"dummy": "dummy"})

        results = retriever.retrieve(corpus, queries)
                
        if "dummy" in queries:
            queries.pop("dummy")
            results.pop("dummy")
        end_time = time()
        logging.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        current_docs = {}
        indices = [i for i,ex in enumerate(dataset) if ex["instance_id"] in queries]
        for index in indices:
            instance_id = dataset[index]["instance_id"]
            current_docs[instance_id] = get_top_docs(results, corpus, instance_id)
            docs_by_instance[instance_id] = current_docs[instance_id]

        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")                
                
        eval_results = {
            "ndcg": ndcg, "mrr": mrr,
            "recall": recall, "precision": precision,
            "time": end_time - start_time
        }
        logging.info(f"Instance #{ins_dir}: {eval_results}")
        completed_eval_rows.append(eval_results)
        for current_instance_id, docs in current_docs.items():
            append_jsonl(results_sidecar, {"instance_id": current_instance_id, "docs": docs})
            completed_ids.add(current_instance_id)
        append_jsonl(eval_sidecar, eval_results)
        sync_results_files(dataset, docs_by_instance, completed_eval_rows, args.results_file, args.output_file)

    if completed_eval_rows:
        print("Average Eval Results: ", load_json(output_paths["eval_summary"]))
    else:
        print("No instances were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets",
                        help="Dataset directory to use for evaluation")
    parser.add_argument("--dataset", type=str, default="humaneval",
                        help="Dataset to use for evaluation")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use for evaluation")
    parser.add_argument("--level", type=str, default="function",
                        help="Localization level to use for evaluation")
    parser.add_argument('--eval_mode', type = str, default = 'default')
    parser.add_argument("--model", type=str, default="Salesforce/SweRankEmbed-Small", help="Embedding model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for retrieval")
    parser.add_argument("--sequence_length", type=int, default=1024, help="Sequence length for retrieval")
    parser.add_argument("--output_file", type=str, default="outputs.json",
                        help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default="results.json",
                        help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument('--add_prefix', action='store_true', help="Add prefix to the queries")
    args = parser.parse_args()

    main()
