# RAG Framework Evaluation Module

This directory provides a comprehensive suite for evaluating the performance of Retrieval-Augmented Generation (RAG) systems. It includes modules for calculating various metrics, summarizing results, and visualizing performance aspects.

## Overview

The evaluation module is designed to assess different facets of a RAG system, focusing on both the retrieval and generation components, as well as the overall quality and factuality of the generated responses.

## Directory Structure

-   **`metrics/`**: Contains the implementation of specific evaluation metrics categorized by their focus:
    -   `retrieval.py`: Metrics evaluating the effectiveness of the retrieval component (e.g., precision, recall, context relevance).
    -   `generation.py`: Metrics assessing the quality of the generated text (e.g., ROUGE, BLEU, cosine similarity).
    -   `self_checker.py`: Metrics related to the system's ability to self-assess or verify the factuality of its outputs against retrieved context.
    -   `__init__.py`: Makes metrics easily importable.
-   **`viz/`**: Contains tools and notebooks for visualizing evaluation results:
    -   `viz_funcs.py`: Python functions to generate various plots and visualizations based on evaluation data.
    -   `viz_results.ipynb`: A Jupyter Notebook demonstrating how to use the visualization functions and potentially showcasing example outputs.
    -   `viz_test.ipynb`: A Jupyter Notebook likely used for testing the visualization functions.
-   **`metrics_summary.py`**: A script responsible for aggregating results from different metrics, potentially calculating overall scores, and presenting a consolidated view of the system's performance.
-   **`schema.py`**: Defines Python data classes and structures used throughout the evaluation module to ensure consistent data handling (e.g., `EvaluationResult`, `MetricScore`).
-   **`__init__.py`**: Initializes the evaluation module, potentially exporting key classes or functions.

## Key Components

### Metrics

The `metrics/` sub-directory provides a modular approach to evaluation. Key metric categories include:

1.  **Retrieval Metrics (`retrieval.py`)**: Focus on how well the system retrieves relevant documents or passages. Common metrics might include Context Precision, Context Recall, and NDCG.
2.  **Generation Metrics (`generation.py`)**: Assess the quality of the final output text. This could involve metrics like ROUGE, BLEU, Faithfulness (consistency with context), and Answer Relevance.
3.  **Self-Checking/Factuality (`self_checker.py`)**: Evaluates the system's internal consistency and adherence to factual information found in the retrieved context.

### Visualization

The `viz/` sub-directory aids in understanding evaluation results through visual means. `viz_funcs.py` likely provides functions to create plots comparing different system configurations, showing score distributions, or illustrating trade-offs between different metrics.

### Summary

`metrics_summary.py` acts as a central point for running evaluations and consolidating results. It likely orchestrates the calculation of various metrics and produces a summary report or data structure containing the overall performance assessment.

## Usage

To evaluate a RAG system using this module:

1.  **Prepare Data**: Ensure your RAG system's outputs (retrieved contexts, generated answers) and the ground truth data (reference answers, relevant documents) are formatted according to the structures potentially defined in `schema.py` or expected by the metric functions.
2.  **Run Metrics**: Utilize functions within the `metrics/` sub-modules or potentially use `metrics_summary.py` to calculate scores across desired metrics.
3.  **Analyze Results**: Examine the calculated scores and use `metrics_summary.py` for an aggregated view.
4.  **Visualize (Optional)**: Use the functions in `viz/viz_funcs.py` or the example notebooks (`viz_results.ipynb`) to generate visualizations for deeper insights.

Refer to specific Python files and potentially the `viz_results.ipynb` notebook for detailed examples and function signatures.

## Target Audience

This module is intended for:

-   **Researchers/Developers**: Building and iterating on RAG systems who need standardized methods to measure improvements.
-   **Evaluators/Professors**: Assessing the performance and capabilities of RAG systems developed within this repository or similar projects.
