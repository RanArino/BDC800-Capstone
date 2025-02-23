# core/evaluation/metrics/retrieval.py

"""
Metrics for evaluating retrieval performance in RAG systems.
- MAP@K: Mean Average Precision@K
- MRR@K: Mean Reciprocal Rank@K
- Hit@K: Hit Rate@K
"""
from typing import List, Dict, Set, Tuple
from statistics import mean

from core.evaluation.schema import RankCutOff

# ===== Main functions =====
def calculate_retrieval_metrics(
    retrieved_docs_list: List[List[str]],
    relevant_docs_list: List[Set[str]],
    k_values: List[int]
) -> Tuple[Dict[str, Dict[RankCutOff, float]], List[Dict[str, Dict[RankCutOff, float]]]]:
    """
    Calculate all retrieval metrics for multiple k values.
    
    Args:
        retrieved_docs_list: List of retrieved document IDs for each query
        relevant_docs_list: List of relevant document IDs for each query
        k_values: List of k values to calculate metrics for
    
    Returns:
        Tuple of (aggregated metrics, list of individual metrics per query)
    """
    pass

# ===== Metrics calculation =====
# MAP@K
def calculate_map_at_k(
        retrieved_docs_list: List[List[str]], 
        relevant_docs_list: List[Set[str]], 
        k_values: List[int]
    ) -> Tuple[Dict[RankCutOff, float], List[Dict[RankCutOff, float]]]:
    """Calculate Mean Average Precision at K across all queries."""
    if not retrieved_docs_list or not relevant_docs_list:
        empty_result = {str(k): 0.0 for k in k_values}
        return empty_result, [empty_result] * len(retrieved_docs_list)
    
    individual_scores = []
    for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
        query_scores = {}
        for k in k_values:
            ap = calculate_average_precision(retrieved, relevant, k)
            query_scores[str(k)] = ap
        individual_scores.append(query_scores)
    
    # Aggregate scores
    map_scores = {}
    for k in k_values:
        k_str = str(k)
        map_scores[k_str] = mean(score[k_str] for score in individual_scores)
    
    return map_scores, individual_scores

# MRR@K
def calculate_mrr_at_k(
        retrieved_docs_list: List[List[str]], 
        relevant_docs_list: List[Set[str]], 
        k_values: List[int]
    ) -> Tuple[Dict[RankCutOff, float], List[Dict[RankCutOff, float]]]:
    """Calculate Mean Reciprocal Rank at K across all queries."""
    if not retrieved_docs_list or not relevant_docs_list:
        empty_result = {str(k): 0.0 for k in k_values}
        return empty_result, [empty_result] * len(retrieved_docs_list)
    
    individual_scores = []
    for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
        query_scores = {}
        for k in k_values:
            # Calculate reciprocal rank for this query at k
            rr = 0.0
            for rank, doc in enumerate(retrieved[:k], 1):
                if doc in relevant:
                    rr = 1.0 / rank
                    break
            query_scores[str(k)] = rr
        individual_scores.append(query_scores)
    
    # Aggregate scores
    mrr_scores = {}
    for k in k_values:
        k_str = str(k)
        mrr_scores[k_str] = mean(score[k_str] for score in individual_scores)
    
    return mrr_scores, individual_scores

# Hit@K
def calculate_hit_at_k(
        retrieved_docs_list: List[List[str]], 
        relevant_docs_list: List[Set[str]], 
        k_values: List[int]
    ) -> Tuple[Dict[RankCutOff, float], List[Dict[RankCutOff, float]]]:
    pass


# ===== Helper functions =====
def calculate_average_precision(
        retrieved_docs: List[str], 
        relevant_docs: Set[str], 
        k: int
    ) -> float:
    """Calculate average precision for a single query up to position k."""
    if not retrieved_docs or not relevant_docs:
        return 0.0
    
    precision_scores = []
    relevant_found = 0
    
    for i, doc in enumerate(retrieved_docs[:k], 1):
        if doc in relevant_docs:
            relevant_found += 1
            precision_scores.append(relevant_found / i)
    
    return mean(precision_scores) if precision_scores else 0.0
