# SweRank

SweRank is a framework for software issue localization that uses a two-stage retrieve-and-rerank pipeline. It first retrieves relevant code snippets for a given issue and then reranks them to find the most likely functions to fix the issue.

Link to Paper: [https://arxiv.org/pdf/2505.07849](https://arxiv.org/pdf/2505.07849)

## Setup

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/gangiswag/SweRank.git
cd SweRank
conda create --name swerank python=3.11.9
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src"
```

## Reproducing Paper Results

Follow these steps to reproduce the evaluation results from our paper.

### 1. Download Evaluation Datasets

The processed versions of `SWE-Bench-Lite` and `LocBench` datasets are available for download [here](https://drive.google.com/file/d/1pjWkpEfFRAts5hkcCGFSoBW1HiQQTsTo/view?usp=share_link). You will need to download and unzip the file to get the `datasets` folder, which is required for the evaluation scripts.

### 2. Run Evaluation

#### SweRankEmbed Evaluation (Retrieval)

To run the `SweRankEmbed-Small` retriever on `SWE-Bench-Lite`:
```bash
bash script/run_retriever.sh Salesforce/SweRankEmbed-Small SweRankEmbed-Small <path_to_datasets_folder> swe-bench-lite
```

To run the retriever on `LocBench`:
```bash
bash script/run_retriever.sh Salesforce/SweRankEmbed-Small SweRankEmbed-Small <path_to_datasets_folder> loc-bench
```

To run the `SweRankEmbed-Large` retriever, replace the model name in the commands above:
```bash
bash script/run_retriever.sh Salesforce/SweRankEmbed-Large SweRankEmbed-Large <path_to_datasets_folder> swe-bench-lite
bash script/run_retriever.sh Salesforce/SweRankEmbed-Large SweRankEmbed-Large <path_to_datasets_folder> loc-bench
```

#### SweRankLLM Evaluation (Reranking)

The reranking scripts use the JSON file with retriever results as input.

To run reranking on `SweRankEmbed-Small` outputs with the `SweRankLLM-Small` model on `SWE-Bench-Lite`:
```bash
bash script/run_rerank.sh SweRankEmbed-Small Salesforce/SweRankLLM-Small SweRankLLM-Small <path_to_datasets_folder> swe-bench-lite
```

To evaluate on `LocBench`:
```bash
bash script/run_rerank.sh SweRankEmbed-Small Salesforce/SweRankLLM-Small SweRankLLM-Small <path_to_datasets_folder> loc-bench
```

To perform reranking with GPT models:
```bash
bash script/run_rerank_gpt.sh ${retriever_name} ${dataset_name}
```
Here, `${retriever_name}` is the name of the retriever output (e.g., `SweRankEmbed-Small`), and `${dataset_name}` is the evaluation dataset (e.g., `swe-bench-lite`).

## Creating the SweLoc Training Dataset

The training data, which we call `SweLoc`, is collected from GitHub issues and pull requests from popular public Python repositories. The collection process is heavily based on [SWE-Bench](https://github.com/SWE-bench/SWE-bench).

### 1. Setup

Set your GitHub access token as an environment variable. You can generate a token from your [GitHub settings](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
```bash
export GITHUB_TOKEN=<your_github_token_here>
```

### 2. Collect Raw Data

The first step is to collect GitHub issues and their corresponding human-written patches.
```bash
cd src/collect
bash collect_data.sh
```

### 3. Mine Negatives for Retriever Data

After collecting issues and the functions modified in patches, this step mines hard negatives from other functions in the repository. This creates a contrastive dataset for training the retriever model.

From the `src/collect` directory, run:
```bash
cd ..
bash negative_mining.sh
```
This script performs quality filtering and saves the retriever training data to `repo_contrastive_mined_filtered.jsonl`.

### 4. Construct Reranker Data

To generate training data for the reranker from the contrastive mined dataset, run the following from the `src` directory:
```bash
bash get_reranker_data.sh
```

## Training Models

### Retriever Training
The file `repo_contrastive_mined_filtered.jsonl` created in the data creation step can be used for contrastive training of a retriever model. The training script for the retriever is not included in this repository.

### Reranker Training

#### Single-Node Training
For training on a single machine with multiple GPUs, run:
```bash
bash script/run_train_reranker_single_node.sh
```
Note: You will need to specify the number of GPUs to be used inside the script.

#### Multi-Node Training
For distributed training across multiple nodes, run:
```bash
bash script/run_train_reranker_multi_node.sh
```
You will have to set your distributed training arguments within the script. We provide a sample script for a SLURM cluster at `script/slurm_run_multinode_train.sh`. Make sure to load any required modules within the example script as well.

## Citation

If you find this work useful in your research, please consider citing our paper:
```
@article{reddy2025swerank,
  title={SweRank: Software Issue Localization with Code Ranking},
  author={Reddy, Revanth Gangi and Suresh, Tarun and Doo, JaeHyeok and Liu, Ye and Nguyen, Xuan Phi and Zhou, Yingbo and Yavuz, Semih and Xiong, Caiming and Ji, Heng and Joty, Shafiq},
  journal={arXiv preprint arXiv:2505.07849},
  year={2025}
}
```