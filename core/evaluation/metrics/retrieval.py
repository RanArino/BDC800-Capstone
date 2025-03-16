# core/evaluation/metrics/retrieval.py

"""
Metrics for evaluating retrieval performance in RAG systems.
- MAP@K: Mean Average Precision@K
- MRR@K: Mean Reciprocal Rank@K
- Hit@K: Hit Rate@K
"""
from typing import List, Dict, Set, Tuple, Union, Optional
from statistics import mean
from langchain_core.documents import Document

from .self_checker import check_retrieval_chunks
from core.evaluation.schema import RankCutOff, SelfCheckerAnswer
from core.logger.logger import get_logger

logger = get_logger(__name__)

# ===== Main functions =====
def calculate_retrieval_metrics(
    qa_id: str,
    question: str,
    ground_truth_answer: str,
    retrieval_chunks: List[Document],
    relevant_doc_ids: Set[str],
    k_values: List[int] = [1, 3, 5, 10],
    llm_eval_enabled: bool = False,
    llm_model: Optional[str] = None
) -> Tuple[Dict[str, Dict[RankCutOff, float]], List[Dict[str, Dict[RankCutOff, float]]]]:
    """
    Calculate all retrieval metrics for multiple k values.
    
    Args:
        qa_id: Unique identifier for the QA pair
        question: The original question
        ground_truth_answer: The ground truth answer
        retrieval_chunks: List of retrieved Langchain Documents
        relevant_doc_ids: Set of relevant document IDs
        k_values: List of k values to calculate metrics for
        llm_eval_enabled: Whether to evaluate retrieval quality using LLM
        llm_model: The model to use for LLM-based evaluation
    
    Returns:
        Tuple of (aggregated metrics, list of individual metrics per query)
    """
    try:
        # Extract document IDs from retrieval chunks
        retrieved_doc_ids = [doc.metadata.get('document_id', '') for doc in retrieval_chunks]
        
        # Create lists for compatibility with existing functions
        retrieved_docs_list = [retrieved_doc_ids]
        relevant_docs_list = [relevant_doc_ids]
        
        map_scores, map_individual = calculate_map_at_k(retrieved_docs_list, relevant_docs_list, k_values)
        mrr_scores, mrr_individual = calculate_mrr_at_k(retrieved_docs_list, relevant_docs_list, k_values)
        hit_scores, hit_individual = calculate_hit_at_k(retrieved_docs_list, relevant_docs_list, k_values)
        
        aggregated_metrics = {
            'map': map_scores,
            'mrr': mrr_scores,
            'hit': hit_scores
        }
        
        # Add LLM-based retrieval evaluation if enabled
        if llm_eval_enabled:
            logger.info(f"Evaluating retrieval quality with LLM for qa_id: {qa_id}")
            try:
                llm_eval_results = check_retrieval_chunks(
                    qa_id=qa_id,
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    retrieval_chunks=retrieval_chunks,
                    top_k=k_values,
                    model=llm_model
                )
                
                # Convert results to the expected format
                if isinstance(llm_eval_results, dict):
                    # Convert int keys to string keys to match other metrics
                    llm_eval_formatted = {str(k): v for k, v in llm_eval_results.items()}
                else:
                    # If a single result is returned, apply it to all k values
                    llm_eval_formatted = {str(k): llm_eval_results for k in k_values}
                
                aggregated_metrics['llm_eval'] = llm_eval_formatted
                logger.debug(f"LLM evaluation results: {llm_eval_formatted}")
            except Exception as e:
                logger.error(f"Error during LLM retrieval evaluation: {e}")
                aggregated_metrics['llm_eval'] = {str(k): "Undetermined" for k in k_values}
        
        individual_metrics = [
            {'map': map_i, 'mrr': mrr_i, 'hit': hit_i}
            for map_i, mrr_i, hit_i in zip(map_individual, mrr_individual, hit_individual)
        ]
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        empty_scores = {str(k): 0.0 for k in k_values}
        aggregated_metrics = {
            'map': empty_scores,
            'mrr': empty_scores,
            'hit': empty_scores
        }
        if llm_eval_enabled:
            aggregated_metrics['llm_eval'] = {str(k): "Undetermined" for k in k_values}
        
        individual_metrics = [{'map': empty_scores, 'mrr': empty_scores, 'hit': empty_scores}]
    
    return aggregated_metrics, individual_metrics

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
    """Calculate Hit Rate at K across all queries."""
    if not retrieved_docs_list or not relevant_docs_list:
        empty_result = {str(k): 0.0 for k in k_values}
        return empty_result, [empty_result] * len(retrieved_docs_list)
    
    individual_scores = []
    for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
        query_scores = {}
        for k in k_values:
            # Calculate hit for this query at k
            hit = float(any(doc in relevant for doc in retrieved[:k]))
            query_scores[str(k)] = hit
        individual_scores.append(query_scores)
    
    # Aggregate scores
    hit_scores = {}
    for k in k_values:
        k_str = str(k)
        hit_scores[k_str] = mean(score[k_str] for score in individual_scores)
    
    return hit_scores, individual_scores


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
