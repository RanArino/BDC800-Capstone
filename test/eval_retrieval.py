# test/eval_retrieval.py

"""
Retrieval evaluation test

- Run the script to debug;
    ```
    python -m pdb test/eval_retrieval.py
    ```
- Input examples are stored in:
    - `test/input_data/multihoprag_qa_list.json` (ground truth)
    - `test/input_data/RAGResponse_multihoprag_test.json` (retrieval results)
- Output example is generated in `test/output_data/retrieval_metrics.json`
"""

import sys
import os
import json
from typing import List, Dict, Set, Tuple
from pathlib import Path
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.evaluation.metrics.retrieval import calculate_retrieval_metrics
from core.datasets.schema import InterDocumentQA, Document
from core.frameworks.schema import RAGResponse

def extract_doc_ids(
        responses: List[RAGResponse],
        qa_pairs: List[InterDocumentQA]
    ) -> Tuple[List[List[str]], List[Set[str]]]:
    """
    Extract retrieved and relevant document IDs from RAG responses and QA pairs.
    
    Args:
        responses: List of RAG response objects
        qa_pairs: List of QA pair objects with ground truth
        
    Returns:
        Tuple of (retrieved document IDs list, relevant document IDs list)
    """
    retrieved_docs_list = []
    relevant_docs_list = []
    
    for response, qa in zip(responses, qa_pairs):
        # Extract retrieved documents (assuming they're ordered by relevance)
        retrieved = [
            doc.metadata['document_id'] for doc in response.context
        ]
        retrieved_docs_list.append(retrieved)
        
        # Extract relevant documents from ground truth QA
        relevant = set(qa.document_ids)
        relevant_docs_list.append(relevant)
    
    return retrieved_docs_list, relevant_docs_list

if __name__ == "__main__":
    from core.datasets.schema import InterDocumentQA
    from core.frameworks.schema import RAGResponse
    
    # Load QA list (ground truth)
    qa_file = "test/input_data/multihoprag_qa_list.json"
    with open(qa_file, 'r') as f:
        qa_data = json.load(f)
        qa_list = [InterDocumentQA(**qa) for qa in qa_data]

    # Load response list (retrieval results)
    response_file = "test/input_data/RAGResponse_multihoprag_test.json"
    with open(response_file, 'r') as f:
        response_data = json.load(f)
        response_list = [RAGResponse(**response) for response in response_data]
    
    # Extract document IDs from both sources
    retrieved_docs_list, relevant_docs_list = extract_doc_ids(response_list, qa_list)
    
    # Define k values for evaluation
    k_values = [1, 3, 5, 10]
    
    # Calculate retrieval metrics
    aggregated_metrics, individual_metrics = calculate_retrieval_metrics(
        retrieved_docs_list,
        relevant_docs_list,
        k_values
    )
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame([
        {
            'query_id': qa.id,
            'question': qa.q,
            **{f"{metric}@{k}": metric_scores[str(k)]
               for metric, metric_scores in metrics.items()
               for k in k_values}
        }
        for qa, metrics in zip(qa_list, individual_metrics)
    ])
    
    # Save metrics to JSON and CSV files
    output_dir = Path("test/output_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed metrics as JSON
    metrics_data = {
        'aggregated': aggregated_metrics,
        'individual': individual_metrics,
        'metadata': {
            'num_queries': len(qa_list),
            'k_values': k_values
        }
    }
    json_file = output_dir / "retrieval_metrics.json"
    with open(json_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # Save DataFrame as CSV
    csv_file = output_dir / "retrieval_metrics.csv"
    df.to_csv(csv_file, index=False)
    
    # Print summary statistics
    print("\nAggregated Metrics:")
    print("==================")
    for metric_name, scores in aggregated_metrics.items():
        print(f"\n{metric_name.upper()}:")
        for k, score in scores.items():
            print(f"@{k}: {score:.3f}")
    
    print("\nMetric Statistics:")
    print("=================")
    metric_cols = [col for col in df.columns if '@' in col]
    for col in metric_cols:
        print(f"\n{col}:")
        print(f"Mean: {df[col].mean():.3f}")
        print(f"Std:  {df[col].std():.3f}")
        print(f"Min:  {df[col].min():.3f}")
        print(f"Max:  {df[col].max():.3f}")
