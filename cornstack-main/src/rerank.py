import os
import json
import argparse
import subprocess
from pathlib import Path

def evaluate_results(eval_dir, dataset_name, qrels_path, results_path):
    """
    Evaluate reranking results using MRR@k metrics
    """
    # Load qrels
    with open(qrels_path) as f:
        qrels = {}
        for i, line in enumerate(f):
            if i == 0 and line.lower().startswith("query-id"):
                continue
            
            qid, docid, score = line.strip().split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(score)
    
    # Load results
    with open(results_path) as f:
        results = json.load(f)
    
    # Calculate MRR@k for different k values
    metrics = [1, 3, 5, 10, 20, 100]
    mrr_at_k = {}
    
    for k in metrics:
        mrr_sum = 0.0
        num_queries = 0
        
        for qid in qrels:
            if qid in results:
                sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
                
                for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
                    if doc_id in qrels[qid] and qrels[qid][doc_id] > 0:
                        mrr_sum += 1.0 / rank
                        break
                num_queries += 1
        
        mrr = mrr_sum / num_queries if num_queries > 0 else 0.0
        mrr_at_k[k] = mrr
    
    # Save evaluation results
    os.makedirs(os.path.join(eval_dir, "eval_results"), exist_ok=True)
    eval_path = os.path.join(eval_dir, "eval_results", f"{dataset_name}_eval.json")
    with open(eval_path, "w") as f:
        json.dump(mrr_at_k, f, indent=4)
    
    return mrr_at_k

def run_convert_and_rerank(args):
    """
    First convert results and then run the reranker on retriever outputs.
    The retriever stores datasets in BEIR format at:
    - CSN: {args.dataset_dir}/code_datasets/csn_{lang}
    - SWE-bench: {args.dataset_dir}/swe-bench-lite-function_{instance_id}
    """
    reranker_path = os.path.join(os.path.dirname(__file__), "..", "llm-reranker")
    convert_script = os.path.join(reranker_path, "scripts", "convert_results.py")
    rerank_script = os.path.join(reranker_path, "scripts", "rerank_llm.py")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    code_datasets_dir = os.path.join(args.dataset_dir, "code_datasets")
    if os.path.exists(code_datasets_dir):
        # datasets = os.listdir(code_datasets_dir)
        datasets = ["csn_ruby"]
    else:
        datasets = []
    
    if os.path.exists(args.dataset_dir):
        datasets.extend([d for d in os.listdir(args.dataset_dir) if d.startswith("swe-bench-lite-function_")])
    for dataset_name in datasets:
        # Determine dataset type and instance ID
        if dataset_name.startswith("csn_"):
            lang = dataset_name.split("_")[1]
            instance_id = None
            data_type = "codedataset"
            prompt_type = "docstring"
            dataset_path = os.path.join(code_datasets_dir, dataset_name)
        elif dataset_name.startswith("swe-bench-lite-function_"):
            instance_id = dataset_name.split("_")[-1]
            data_type = "codedataset"
            prompt_type = "github_issue"
            dataset_path = os.path.join(args.dataset_dir, dataset_name)
        else:
            continue
            
        if not os.path.isdir(dataset_path):
            continue
            
        # First convert results
        print(f"Converting results for {dataset_name}...")
        convert_cmd = [
            "python", convert_script,
            "--dataset", dataset_name,
            "--output_dir", args.output_dir,
            "--data_type", data_type,
            "--data_dir", args.dataset_dir if dataset_name.startswith("swe-bench") else code_datasets_dir,
            "--top_k", str(args.top_k),
            "--rerank_type", "code"
        ]
        
        try:
            subprocess.run(convert_cmd, check=True)
            print(f"Successfully converted {dataset_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {dataset_name}: {e}")
            continue
            
        # Then run reranker
        print(f"Running reranker on {dataset_name}...")
        rerank_cmd = [
            "python", rerank_script,
            "--model", "cornstack/CodeRankLLM",
            "--dataset", dataset_name,
            "--output_dir", args.output_dir,
            "--data_type", data_type,
            "--data_dir", args.dataset_dir if dataset_name.startswith("swe-bench") else code_datasets_dir,
            "--use_logits", "0",
            "--use_alpha", "0",
            "--llm_top_k", str(args.top_k),
            "--window_size", str(args.window_size),
            "--step_size", str(args.step_size),
            "--do_batched", "1",
            "--rerank_type", "code",
            "--code_prompt_type", prompt_type
        ]
        
        try:
            subprocess.run(rerank_cmd, check=True)
            print(f"Successfully reranked {dataset_name}")
            
            # Evaluate results
            print(f"Evaluating results for {dataset_name}...")
            qrels_path = os.path.join(dataset_path, "qrels", "test.tsv")
            results_path = os.path.join(args.output_dir, "code_datasets", dataset_name, f"rerank_{args.top_k}_llm_gen_num.json")
            
            mrr_at_k = evaluate_results(args.eval_dir, dataset_name, qrels_path, results_path)
            print(f"Evaluation results for {dataset_name}:")
            for k, mrr in mrr_at_k.items():
                print(f"MRR@{k}: {mrr:.4f}")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to rerank {dataset_name}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                      help="Directory containing retriever outputs in BEIR format")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to store reranker outputs")
    parser.add_argument("--eval_dir", type=str, required=True,
                      help="Directory to store evaluation results")
    parser.add_argument("--top_k", type=int, default=100,
                      help="Number of candidates to rerank")
    parser.add_argument("--window_size", type=int, default=10,
                      help="Window size for reranking")
    parser.add_argument("--step_size", type=int, default=5,
                      help="Step size for reranking")
    args = parser.parse_args()
    
    run_convert_and_rerank(args)

if __name__ == "__main__":
    main() 