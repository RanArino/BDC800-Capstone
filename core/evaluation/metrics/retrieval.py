# core/evaluation/metrics/retrieval.py

"""
Metrics for evaluating retrieval performance in RAG systems.
- MAP@K: Mean Average Precision@K
- MRR@K: Mean Reciprocal Rank@K
- Hit@K: Hit Rate@K
"""
from typing import List, Dict, Set, Tuple
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
    pass

# MRR@K
def calculate_mrr_at_k(
        retrieved_docs_list: List[List[str]], 
        relevant_docs_list: List[Set[str]], 
        k_values: List[int]
    ) -> Tuple[Dict[RankCutOff, float], List[Dict[RankCutOff, float]]]:
    pass

# Hit@K
def calculate_hit_at_k(
        retrieved_docs_list: List[List[str]], 
        relevant_docs_list: List[Set[str]], 
        k_values: List[int]
    ) -> Tuple[Dict[RankCutOff, float], List[Dict[RankCutOff, float]]]:
    pass

