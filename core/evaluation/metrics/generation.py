# core/evaluation/metrics/generation.py

"""
Metrics for evaluating text generation quality in RAG systems.
"""
from typing import Dict, List
import numpy as np
from rouge_score import rouge_scorer
import sacrebleu
from sentence_transformers import SentenceTransformer

from core.evaluation.schema import RougeType, RougeMetricType, RougeMetrics, GenerationEval
from core.logger.logger import get_logger

logger = get_logger(__name__)

# ROUGE scores
def calculate_rouge_scores(
        generated_text: str, 
        reference_text: str, 
        rouge_types: List[RougeType] = ['rouge1', 'rouge2', 'rougeL'],
        metric_types: List[RougeMetricType] = ['recall']
    ) -> Dict[RougeType, RougeMetrics]:
    """
    Calculate ROUGE scores for generated text against reference.
    Returns only the specified metric types.
    
    Args:
        generated_text: Generated text to evaluate
        reference_text: Ground truth text
        rouge_types: Types of ROUGE metrics to calculate
        metric_types: Types of metrics to return ('precision', 'recall', 'fmeasure')
        
    Returns:
        Dict containing ROUGE scores where keys are rouge types (e.g. rouge1) and values are RougeMetrics objects
    """
    try:
        if not generated_text or not reference_text:
            return {rouge_type: RougeMetrics(**{metric: 0.0 for metric in metric_types}) 
                   for rouge_type in rouge_types}

        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        scores = scorer.score(reference_text, generated_text)
        
        return {
            rouge_type: RougeMetrics(**{
                metric: getattr(scores[rouge_type], metric)
                for metric in metric_types
            })
            for rouge_type in rouge_types
        }
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores: {e}")
        return {rouge_type: RougeMetrics(**{metric: 0.0 for metric in metric_types}) 
                for rouge_type in rouge_types}

# BLEU scores
def calculate_bleu_score(
        generated_text: str, 
        reference_text: str
    ) -> float:
    """
    Calculate BLEU score for generated text against reference.
    
    Args:
        generated_text: Generated text to evaluate
        reference_text: Ground truth text
        
    Returns:
        float: BLEU score
    """
    try:
        if not generated_text or not reference_text:
            return 0.0
            
        # SacreBLEU expects a list of references
        bleu = sacrebleu.metrics.BLEU()
        score = bleu.corpus_score([generated_text], [[reference_text]])
        return score.score / 100.0  # Normalize to [0,1] range
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        return 0.0
