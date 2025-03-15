# core/evaluation/metrics_summary.py

"""
Metrics summary module for RAG evaluation.

This module provides functions to calculate, accumulate, and summarize metrics
for RAG system evaluation. It's designed to work with the BaseRAGFramework.run() method
and calculates metrics per QA pair while efficiently managing memory usage.
"""

from typing import List, Union, Tuple, Set, Optional, Dict, TYPE_CHECKING
import pandas as pd
from collections import defaultdict

from core.datasets.schema import IntraDocumentQA, InterDocumentQA
# Import RAGResponse only for type checking to avoid circular imports
if TYPE_CHECKING:
    from core.frameworks.schema import RAGResponse
from core.evaluation.metrics import (
    calculate_retrieval_metrics, 
    calculate_generation_metrics, 
    check_retrieval_chunks
)
from core.evaluation.schema import (
    ProfilerTimingKey,
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
    response: "RAGResponse",
    k_values: List[int] = [1, 3, 5, 10],
    rouge_types: List[RougeType] = ['rouge1', 'rouge2', 'rougeL'],
    rouge_metric_types: List[RougeMetricType] = ['precision', 'recall', 'fmeasure'],
    llm_eval_enabled: bool = True,
    llm_model: str = 'phi4:14b'
) -> MetricsSummary:
    """
    Computes retrieval and generation metrics for a QA pair and its RAG response.

    Args:
        qa: Question-answer pair (IntraDocumentQA or InterDocumentQA).
        response: RAG response with generated answer and context.
        k_values: k values for retrieval metrics.
        rouge_types: ROUGE metrics to calculate.
        rouge_metric_types: ROUGE metrics to return.
        llm_eval_enabled: Whether to evaluate retrieval quality using LLM.
        llm_model: The model to use for LLM-based evaluation.

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
        'query': query,
        'ground_truth': ground_truth_answer,
        'llm_answer': generated_text
    }

    # Calculate retrieval metrics if context is available
    if response.context:
        # For InterDocumentQA, we have document IDs for traditional metrics
        if isinstance(qa, InterDocumentQA):
            # Get the set of relevant document IDs from the QA pair
            relevant_doc_ids = set(qa.document_ids)
            
            # Calculate retrieval metrics
            retrieval_metrics, _ = calculate_retrieval_metrics(
                qa_id=qa.id,
                question=query,
                ground_truth_answer=ground_truth_answer,
                retrieval_chunks=response.context,
                relevant_doc_ids=relevant_doc_ids,
                k_values=k_values,
                llm_eval_enabled=llm_eval_enabled,
                llm_model=llm_model
            )
            metrics['retrieval'] = retrieval_metrics
        
        # For IntraDocumentQA, we only do LLM-based evaluation
        elif isinstance(qa, IntraDocumentQA) and llm_eval_enabled:
            # Use check_retrieval_chunks directly for efficiency
            llm_eval_results = check_retrieval_chunks(
                qa_id=qa.id,
                question=query,
                ground_truth_answer=ground_truth_answer,
                retrieval_chunks=response.context,
                top_k=k_values,
                model=llm_model
            )
            
            # Create a retrieval metrics dictionary with only llm_eval
            # Convert int keys to string keys if needed
            if isinstance(llm_eval_results, dict):
                llm_eval_formatted = {str(k): v for k, v in llm_eval_results.items()}
            else:
                # If a single result is returned, apply it to all k values
                llm_eval_formatted = {str(k): llm_eval_results for k in k_values}
            
            # Create empty metrics for MAP, MRR, Hit
            empty_scores = {str(k): 0.0 for k in k_values}
            
            retrieval_metrics = {
                'map': empty_scores,
                'mrr': empty_scores,
                'hit': empty_scores,
                'llm_eval': llm_eval_formatted
            }
            
            metrics['retrieval'] = retrieval_metrics
  
    # Calculate generation metrics
    if generated_text:
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

def accumulate_and_summarize_metrics(
    metrics_list: List[MetricsSummary],
    profiler_metrics: Dict[ProfilerTimingKey, Dict[str, float]],
) -> Tuple[MetricsSummaryStats, pd.DataFrame]:
    """
    Accumulate and summarize metrics from multiple QA pairs.
    
    Args:
        metrics_list: List of MetricsSummary objects
        profiler: Profiler instance to extract performance metrics
        return_detailed: If True, returns detailed metrics DataFrame
        
    Returns:
        Tuple of (MetricsSummaryStats object, optional DataFrame with detailed metrics)
    """
    # Initialize accumulators for different metric types
    generation_metrics = defaultdict(lambda: defaultdict(list))
    retrieval_metrics = defaultdict(lambda: defaultdict(list))
    self_checker_results = []
    llm_retrieval_results = defaultdict(list)
    
    # Create separate lists for single-value metrics
    bleu_values = []
    cosine_sim_values = []
    
    # Process each metrics entry
    for metrics in metrics_list:
        # Process generation metrics
        if metrics.generation:
            # Handle ROUGE metrics
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                rouge_metrics = getattr(metrics.generation, rouge_type, None)
                if rouge_metrics:
                    # Handle precision, recall, fmeasure
                    if rouge_metrics.precision is not None:
                        generation_metrics[rouge_type]['precision'].append(rouge_metrics.precision)
                    if rouge_metrics.recall is not None:
                        generation_metrics[rouge_type]['recall'].append(rouge_metrics.recall)
                    if rouge_metrics.fmeasure is not None:
                        generation_metrics[rouge_type]['fmeasure'].append(rouge_metrics.fmeasure)
            
            # Handle other generation metrics
            if metrics.generation.bleu is not None:
                bleu_values.append(metrics.generation.bleu)
            if metrics.generation.cosine_sim is not None:
                cosine_sim_values.append(metrics.generation.cosine_sim)
            
            # Handle self_checker
            if metrics.generation.self_checker:
                self_checker_results.append(metrics.generation.self_checker)
        
        # Process retrieval metrics if available
        if metrics.retrieval:
            # Handle MAP metrics
            if hasattr(metrics.retrieval, 'map') and metrics.retrieval.map:
                for k, value in metrics.retrieval.map.items():
                    retrieval_metrics['map'][k].append(value)
            
            # Handle MRR metrics
            if hasattr(metrics.retrieval, 'mrr') and metrics.retrieval.mrr:
                for k, value in metrics.retrieval.mrr.items():
                    retrieval_metrics['mrr'][k].append(value)
            
            # Handle Hit metrics
            if hasattr(metrics.retrieval, 'hit') and metrics.retrieval.hit:
                for k, value in metrics.retrieval.hit.items():
                    retrieval_metrics['hit'][k].append(value)
                    
            # Handle LLM-based retrieval evaluation
            if hasattr(metrics.retrieval, 'llm_eval') and metrics.retrieval.llm_eval:
                for k, result in metrics.retrieval.llm_eval.items():
                    llm_retrieval_results[str(k)].append(result)
    
    # Build generation stats
    generation_stats = None
    if generation_metrics or bleu_values or cosine_sim_values:
        # Build ROUGE metrics stats
        rouge1_stats = None
        rouge2_stats = None
        rougeL_stats = None
        
        if 'rouge1' in generation_metrics:
            precision_stats = _create_stat_value(generation_metrics['rouge1'].get('precision'))
            recall_stats = _create_stat_value(generation_metrics['rouge1'].get('recall'))
            fmeasure_stats = _create_stat_value(generation_metrics['rouge1'].get('fmeasure'))
            rouge1_stats = RougeMetrics(
                precision=precision_stats,
                recall=recall_stats,
                fmeasure=fmeasure_stats
            )
        
        if 'rouge2' in generation_metrics:
            precision_stats = _create_stat_value(generation_metrics['rouge2'].get('precision'))
            recall_stats = _create_stat_value(generation_metrics['rouge2'].get('recall'))
            fmeasure_stats = _create_stat_value(generation_metrics['rouge2'].get('fmeasure'))
            rouge2_stats = RougeMetrics(
                precision=precision_stats,
                recall=recall_stats,
                fmeasure=fmeasure_stats
            )
        
        if 'rougeL' in generation_metrics:
            precision_stats = _create_stat_value(generation_metrics['rougeL'].get('precision'))
            recall_stats = _create_stat_value(generation_metrics['rougeL'].get('recall'))
            fmeasure_stats = _create_stat_value(generation_metrics['rougeL'].get('fmeasure'))
            rougeL_stats = RougeMetrics(
                precision=precision_stats,
                recall=recall_stats,
                fmeasure=fmeasure_stats
            )
        
        # Build other metrics stats
        bleu_stats = None
        if bleu_values:
            bleu_stats = _create_stat_value(bleu_values)
        
        cosine_sim_stats = None
        if cosine_sim_values:
            cosine_sim_stats = _create_stat_value(cosine_sim_values)
        
        # Calculate self_checker accuracy
        self_checker_accuracy = None
        if self_checker_results:
            yes_count = sum(1 for result in self_checker_results if result == "Yes")
            self_checker_accuracy = yes_count / len(self_checker_results)
        
        generation_stats = GenerationEval(
            rouge1=rouge1_stats,
            rouge2=rouge2_stats,
            rougeL=rougeL_stats,
            bleu=bleu_stats,
            cosine_sim=cosine_sim_stats,
            self_checker_accuracy=self_checker_accuracy
        )
    
    # Build retrieval stats
    retrieval_stats = None
    if retrieval_metrics or llm_retrieval_results:
        map_stats = {}
        mrr_stats = {}
        hit_stats = {}
        llm_eval_stats = {}
        
        if 'map' in retrieval_metrics:
            for k, values in retrieval_metrics['map'].items():
                map_stats[k] = _create_stat_value(values)
        
        if 'mrr' in retrieval_metrics:
            for k, values in retrieval_metrics['mrr'].items():
                mrr_stats[k] = _create_stat_value(values)
        
        if 'hit' in retrieval_metrics:
            for k, values in retrieval_metrics['hit'].items():
                hit_stats[k] = _create_stat_value(values)
        
        # Calculate LLM-based retrieval evaluation stats
        if llm_retrieval_results:
            for k, results in llm_retrieval_results.items():
                yes_count = sum(1 for result in results if result == "Yes")
                llm_eval_stats[k] = yes_count / len(results) if results else 0.0
        
        retrieval_stats = RetrievalEval(
            map=map_stats,
            mrr=mrr_stats,
            hit=hit_stats,
            llm_eval=llm_eval_stats if llm_eval_stats else None
        )
    
    # Extract profiler metrics if provided
    processing_time = {}
    memory_usage = {}
    for key, data in profiler_metrics.items():
        # Add processing time metrics
        processing_time[key] = data.get("duration", 0)
        
        # Add memory usage metrics
        memory_usage[key] = data["memory"].get("total_mb", 0)
    
    # Create the MetricsSummaryStats object
    summary_stats = MetricsSummaryStats(
        total_queries=len(metrics_list),
        generation=generation_stats,
        retrieval=retrieval_stats,
        processing_time=processing_time,
        memory_usage=memory_usage
    )
    
    # Convert MetricsSummary objects to dictionaries with flattened structure
    flat_metrics = []
    for metrics in metrics_list:
        flat_dict = {
            'qa_id': metrics.qa_id,
            'query': metrics.query,
            'ground_truth': metrics.ground_truth,
            'llm_answer': metrics.llm_answer
        }
        
        # Add generation metrics
        if metrics.generation:
            # Add ROUGE metrics
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                rouge_metrics = getattr(metrics.generation, rouge_type, None)
                if rouge_metrics:
                    if rouge_metrics.precision is not None:
                        flat_dict[f"{rouge_type}_p"] = rouge_metrics.precision
                    if rouge_metrics.recall is not None:
                        flat_dict[f"{rouge_type}_r"] = rouge_metrics.recall
                    if rouge_metrics.fmeasure is not None:
                        flat_dict[f"{rouge_type}_f"] = rouge_metrics.fmeasure
            
            # Add other generation metrics
            if metrics.generation.bleu is not None:
                flat_dict['bleu'] = metrics.generation.bleu
            if metrics.generation.cosine_sim is not None:
                flat_dict['cosine_sim'] = metrics.generation.cosine_sim
            if metrics.generation.self_checker:
                flat_dict['self_checker'] = metrics.generation.self_checker
        
        # Add retrieval metrics
        if metrics.retrieval:
            # Add MAP metrics
            if metrics.retrieval.map:
                for k, value in metrics.retrieval.map.items():
                    flat_dict[f"map@{k}"] = value
            
            # Add MRR metrics
            if metrics.retrieval.mrr:
                for k, value in metrics.retrieval.mrr.items():
                    flat_dict[f"mrr@{k}"] = value
            
            # Add Hit metrics
            if metrics.retrieval.hit:
                for k, value in metrics.retrieval.hit.items():
                    flat_dict[f"hit@{k}"] = value
                    
            # Add LLM-based retrieval evaluation
            if hasattr(metrics.retrieval, 'llm_eval') and metrics.retrieval.llm_eval:
                for k, value in metrics.retrieval.llm_eval.items():
                    flat_dict[f"llm_eval@{k}"] = value
        
        flat_metrics.append(flat_dict)
    
    detailed_df = pd.DataFrame(flat_metrics)
    
    return summary_stats, detailed_df

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