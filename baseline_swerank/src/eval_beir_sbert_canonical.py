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
from local_dataset_utils import instance_id_from_dir, list_local_instance_dirs

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

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
        dataset_records = [dict(item) for item in dataset]
    elif args.dataset == "loc-bench":
        instance_list = list_local_instance_dirs(args.dataset_dir, args.dataset, args.split, args.level)
        dataset_records = [
            {"instance_id": instance_id_from_dir(ins_dir, args.dataset, args.split, args.level)}
            for ins_dir in instance_list
        ]
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    all_eval_results = []
    all_top_docs = [[] for _ in dataset_records]
    if args.split == 'test':
        prefx = f"{args.dataset}" 
    else: 
        prefx = f"{args.dataset}-{args.split}"
    if args.level == 'file':
        prefx += '_'
    else:
        prefx += f'-{args.level}_'

    if args.dataset != "loc-bench":
        instance_list = [i for i in os.listdir(args.dataset_dir) if i.startswith(prefx)]
    if args.max_instances is not None:
        instance_list = instance_list[:args.max_instances]
        if args.dataset == "loc-bench":
            dataset_records = dataset_records[:args.max_instances]
            all_top_docs = all_top_docs[:args.max_instances]
        logging.info(f"Limiting to {args.max_instances} instances")

    if not instance_list:
        raise ValueError(f"No BEIR instance directories found in {args.dataset_dir} with prefix {prefx}")

    instance_index = {
        item["instance_id"]: index
        for index, item in enumerate(dataset_records)
    }

    for ins_dir in tqdm(instance_list):
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

        for instance_id in queries:
            if instance_id in instance_index:
                all_top_docs[instance_index[instance_id]] = get_top_docs(results, corpus, instance_id)

        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")                
                
        eval_results = {
            "ndcg": ndcg, "mrr": mrr,
            "recall": recall, "precision": precision,
            "time": end_time - start_time
        }
        logging.info(f"Instance #{ins_dir}: {eval_results}")
        all_eval_results.append(eval_results)

    with open(args.results_file, "w", encoding="utf-8") as f:
        for item, docs in zip(dataset_records, all_top_docs):
            output_item = dict(item)
            output_item["docs"] = docs
            f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    avg_eval_results = {}
    for k,v_dict in all_eval_results[0].items():
        if isinstance(v_dict, dict):
            avg_v_dict = {}
            for vk,vv in v_dict.items():
                avg_vv = sum([e[k][vk] for e in all_eval_results])/len(all_eval_results)
                avg_v_dict[vk] = avg_vv
            avg_eval_results.update(avg_v_dict)
        elif isinstance(v_dict, float):
            avg_v = sum([e[k] for e in all_eval_results])/len(all_eval_results)
            avg_eval_results[k] = avg_v
        else:
            raise ValueError
        
    print("Average Eval Results: ", avg_eval_results)
    with open(args.output_file, "w") as f:
        json.dump(avg_eval_results, f)

    with open(args.output_file + "_all", "w") as f:
        json.dump(all_eval_results, f)

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
    parser.add_argument("--max_instances", type=int, default=None, help="Limit number of instances to evaluate")
    args = parser.parse_args()

    main()
