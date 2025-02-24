# core/evaluation/metrics_summary.py

"""
Metrics summary module for RAG evaluation.

This module provides functions to calculate, accumulate, and summarize metrics
for RAG system evaluation. It's designed to work with the BaseRAGFramework.run() method
and calculates metrics per QA pair while efficiently managing memory usage.
"""

from typing import List, Union, Tuple, Set, Optional
import pandas as pd
from collections import defaultdict

from core.datasets.schema import IntraDocumentQA, InterDocumentQA
from core.frameworks.schema import RAGResponse
from core.evaluation.metrics import calculate_retrieval_metrics, calculate_generation_metrics
from core.evaluation.schema import (
    RougeType,
    RougeMetricType, 
    MetricsSummary, 
    MetricsSummaryStats,
    StatValue,
    RougeMetrics,
    GenerationEval,
    RetrievalEval
)


def calculate_metrics_for_qa(
    qa: Union[IntraDocumentQA, InterDocumentQA],
    response: RAGResponse,
    k_values: List[int] = [1, 3, 5, 10],
    rouge_types: List[RougeType] = ['rouge1', 'rouge2', 'rougeL'],
    rouge_metric_types: List[RougeMetricType] = ['precision', 'recall', 'fmeasure']
) -> MetricsSummary:
    """
    Computes retrieval and generation metrics for a QA pair and its RAG response.

    Args:
        qa: Question-answer pair (IntraDocumentQA or InterDocumentQA).
        response: RAG response with generated answer and context.
        k_values: k values for retrieval metrics.
        rouge_types: ROUGE metrics to calculate.
        rouge_metric_types: ROUGE metrics to return.

    Returns:
        MetricsSummary object with calculated metrics.
    """
    # Extract necessary information
    query = qa.q
    ground_truth_answer = qa.a
    # ground_truth_evidence = qa.e
    generated_text = response.llm_answer
    
    # Initialize metrics dictionary with basic information
    metrics = {
        'qa_id': qa.id,
        'query': query[:30],
        'ground_truth': ground_truth_answer[:30],
        'llm_answer': generated_text[:30]
    }

    # Calculate retrieval metrics if qa is an InterDocumentQA
    if isinstance(qa, InterDocumentQA):
        # Extract document IDs
        retrieved_docs_list, relevant_docs_list = _extract_doc_ids(
            responses=[response],
            qa_pairs=[qa]
        )
        # NOTE: this experiment passes a single QA pair, so both returns are the same.
        retrieval_metrics, _ = calculate_retrieval_metrics(
            retrieved_docs_list=retrieved_docs_list,
            relevant_docs_list=relevant_docs_list,
            k_values=k_values
        )
        metrics['retrieval'] = retrieval_metrics
    
    # Calculate generation metrics
    generation_metrics = calculate_generation_metrics(
        qa_id=qa.id,
        question=query,
        generated_text=generated_text,
        reference_text=ground_truth_answer,
        rouge_types=rouge_types,
        rouge_metric_types=rouge_metric_types
    )
    metrics['generation'] = generation_metrics
    
    # Return as a MetricsSummary object
    return MetricsSummary(**metrics)

def _extract_doc_ids(
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

def _create_stat_value(values):
    """
    Create a StatValue object from a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        StatValue object with statistical measures or None if values is empty
    """
    if not values:
        return None
    
    values_series = pd.Series(values)
    return StatValue(
        mean=float(values_series.mean()),
        std=float(values_series.std()),
        median=float(values_series.median()),
        q1=float(values_series.quantile(0.25)),
        q3=float(values_series.quantile(0.75)),
        min=float(values_series.min()),
        max=float(values_series.max())
    )