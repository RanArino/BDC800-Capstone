# RAG Framework Experiments

This directory (`experiments/`) is designed for configuring, running, and managing experiments to evaluate and compare different RAG (Retrieval-Augmented Generation) framework configurations.

## Overview

The primary goal of this module is to provide a structured way to execute RAG pipelines defined in `core/frameworks` using datasets from `core/datasets` and configurations specified in `core/configs`. It automates the process of running tests, collecting detailed results, and generating summary metrics.

## Core Components

-   **`main.py`**:
    -   **Purpose**: Serves as the main entry point for executing predefined batches of experiments.
    -   **Functionality**: It defines several functions (e.g., `run_simple_rag_sentences`, `run_scaler_rag_reasoning`) each designed to test specific RAG framework variations (SimpleRAG, ScalerRAG, ScalerV1RAG) with different configurations (chunking strategies, reasoning models, dimensionality reduction) specified in YAML files within `core/configs/`. When executed (`python experiments/main.py`), it calls the `run_experiment` function from `base.py` for each configuration within the selected function(s). You can uncomment the desired function calls in the `if __name__ == "__main__":` block to run specific experiment suites.

-   **`base.py`**:
    -   **Purpose**: Contains the core logic for running a single RAG experiment configuration.
    -   **Functionality**: Defines the `run_experiment` function, which is the workhorse of the experimentation process. Its key responsibilities include:
        1.  **Loading Configuration**: Reads a specified configuration file (e.g., `core/configs/simple_rag/rag01_sentence_chunk.yaml`) and a specific configuration name within that file.
        2.  **Initializing Framework**: Instantiates the appropriate RAG framework (`SimpleRAG`, `ScalerRAG`, or `ScalerV1RAG`) based on the configuration path.
        3.  **Loading Dataset**: Loads the dataset specified in the configuration.
        4.  **Running RAG**: Executes the RAG pipeline (indexing, retrieval, generation) for the loaded dataset (handling both `IntraDocumentQA` and `InterDocumentQA` types). It uses the `core/evaluation` tools internally to calculate metrics during the run.
        5.  **Summarizing Metrics**: Aggregates metrics across all queries using `accumulate_and_summarize_metrics` from `core/evaluation.metrics_summary`.
        6.  **Storing Results**: Saves the outputs of the experiment run.

-   **`__init__.py`**:
    -   **Purpose**: Standard Python package initializer.
    -   **Functionality**: Makes the `run_experiment` function from `base.py` directly importable from the `experiments` package (e.g., `from experiments import run_experiment`).

## Running Experiments and Output Files

To run experiments, execute the `main.py` script:

```bash
python experiments/main.py
```

Make sure to modify the `if __name__ == "__main__":` block in `main.py` to uncomment the specific experiment suite(s) you wish to run.

After successfully running an experiment configuration (e.g., `scaler_rag_03_01` via `run_experiment`), the following output files are generated automatically, timestamped to avoid overwriting previous runs:

1.  **Raw Responses (`responses/`)**:
    *   **Filename**: `responses/<config_name>-<YYYYMMDD-HHMMSS>.json`
    *   **Content**: A JSON file containing a list of detailed `RAGResponse` objects (as defined in `core/frameworks/schema.py`) for each query processed in the experiment. This includes the original query, generated answer, retrieved context, ground truth, etc.
    *   **Controlled by**: `store_responses=True` argument in `run_experiment`.

2.  **Metrics Summary (`metrics/`)**:
    *   **Filename**: `metrics/<config_name>-<YYYYMMDD-HHMMSS>.json`
    *   **Content**: A JSON file containing the aggregated `MetricsSummary` object (as defined in `core/evaluation/schema.py`). This includes overall scores for retrieval metrics (precision, recall), generation metrics (ROUGE, BLEU, faithfulness), answer relevance, and profiling information (latencies).
    *   **Controlled by**: `store_metrics=True` argument in `run_experiment`.

3.  **Detailed Metrics Dataframe (`detailed_dfs/`)**:
    *   **Filename**: `detailed_dfs/<config_name>-<YYYYMMDD-HHMMSS>.csv`
    *   **Content**: A CSV file providing a detailed breakdown of metrics for *each individual query* processed during the experiment. This allows for fine-grained analysis of performance across different questions.
    *   **Controlled by**: `store_detailed_df=True` argument in `run_experiment`.

## Other Files and Directories

-   **`responses/`, `metrics/`, `detailed_dfs/`**: Subdirectories automatically created by `base.py` to store the generated output files described above.
-   **`result_*.csv`, `*_summary.txt`**: These appear to be manually generated or previously saved summary files from specific experiment runs (e.g., `frames_summary.txt`, `multihoprag_summary.txt`). They might contain high-level results or analyses.
-   **`update_llm_eval.py`**: A utility script, potentially for re-evaluating saved responses using different LLM-based metrics or updating results without rerunning the full RAG pipeline.
