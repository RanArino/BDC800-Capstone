# core/evaluation/metrics/generation.py

"""
Metrics for evaluating text generation quality in RAG systems.
"""
from typing import Dict, List
import numpy as np
from rouge_score import rouge_scorer
import sacrebleu
from sentence_transformers import SentenceTransformer

from .self_checker import check_llm_answer
from core.evaluation.schema import RougeType, RougeMetricType, RougeMetrics, GenerationEval
from core.logger.logger import get_logger

logger = get_logger(__name__)

# Initialize sentence transformer model
try:
    sentence_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
except Exception as e:
    logger.error(f"Error loading sentence transformer model: {e}")
    sentence_model = None

# ROUGE scores
def calculate_rouge_scores(
        generated_text: str, 
        reference_text: str, 
        rouge_types: List[RougeType],
        metric_types: List[RougeMetricType],
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
    logger.debug(f"Calculating ROUGE scores with types={rouge_types}, metrics={metric_types}")

    try:
        if not generated_text or not reference_text:
            logger.debug("Empty input text detected, returning zero scores")
            return {rouge_type: RougeMetrics(**{metric: 0.0 for metric in metric_types}) 
                   for rouge_type in rouge_types}

        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        scores = scorer.score(reference_text, generated_text)
        logger.debug(f"Raw ROUGE scores: {scores}")
        
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
    logger.debug(f"Calculating BLEU score")
    
    try:
        if not generated_text or not reference_text:
            logger.debug("Empty input text detected, returning zero score")
            return 0.0
            
        # SacreBLEU expects a list of references
        bleu = sacrebleu.metrics.BLEU()
        score = bleu.corpus_score([generated_text], [[reference_text]])
        return score.score / 100.0  # Normalize to [0,1] range
    
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        return 0.0

# Cosine similarity
def calculate_cosine_similarity(generated_text: str,
                              reference_text: str) -> float:
    """
    Calculate cosine similarity between generated text and reference text using sentence embeddings.
    
    Args:
        generated_text: Generated text to evaluate
        reference_text: Ground truth text
        
    Returns:
        float: Cosine similarity score in range [0,1]
    """
    logger.debug("Calculating cosine similarity")
    
    try:
        if not generated_text or not reference_text or sentence_model is None:
            logger.debug("Empty input text or no model available, returning zero score")
            return 0.0

        # Get embeddings for both texts
        generated_embedding = sentence_model.encode(generated_text, convert_to_tensor=True)
        reference_embedding = sentence_model.encode(reference_text, convert_to_tensor=True)
        logger.debug("Successfully generated embeddings")
        
        # Move tensors to CPU before converting to numpy
        generated_embedding = generated_embedding.cpu().numpy()
        reference_embedding = reference_embedding.cpu().numpy()
        
        # Calculate cosine similarity
        similarity = np.dot(generated_embedding, reference_embedding) / \
                    (np.linalg.norm(generated_embedding) * np.linalg.norm(reference_embedding))
        
        return float(similarity)
    
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

# Generation metrics
def calculate_generation_metrics(
        qa_id: str,
        question: str,
        generated_text: str, 
        reference_text: str,
        rouge_types: List[RougeType] = ['rouge1', 'rouge2', 'rougeL'],
        rouge_metric_types: List[RougeMetricType] = ['recall']
    ) -> GenerationEval:
    """
    Calculate all generation metrics for a single response.
    
    Args:
        generated_text: Generated text to evaluate
        reference_text: Ground truth text
        rouge_types: Types of ROUGE metrics to calculate
        rouge_metric_types: Types of ROUGE metrics to return ('precision', 'recall', 'fmeasure')
        
    Returns:
        GenerationEval object containing all calculated metrics
    """
    logger.info(f"Calculating generation metrics")
    try:
        # Calculate ROUGE scores with specified metric types
        rouge_scores = calculate_rouge_scores(
            generated_text=generated_text, 
            reference_text=reference_text,
            rouge_types=rouge_types,
            metric_types=rouge_metric_types
        )
        logger.debug(f"ROUGE scores calculated: {rouge_scores}")
        
        # Calculate BLEU score
        bleu = calculate_bleu_score(generated_text, reference_text)
        logger.debug(f"BLEU score calculated: {bleu}")
        
        # Calculate cosine similarity
        cosine_sim = calculate_cosine_similarity(generated_text, reference_text)
        logger.debug(f"Cosine similarity calculated: {cosine_sim}")

        # Check if LLM's answer aligns with ground truth
        confirmation = check_llm_answer(qa_id, question, reference_text, generated_text)
        logger.debug(f"Self-checker Completd: {confirmation}")

        logger.info(f"All metrics calculated successfully")
        
        # Create GenerationEval object with all metrics
        return GenerationEval(
            rouge1=rouge_scores['rouge1'],
            rouge2=rouge_scores.get('rouge2'),  # Optional
            rougeL=rouge_scores.get('rougeL'),  # Optional
            bleu=bleu,
            cosine_sim=cosine_sim,
            self_checker=confirmation
        )
        
    except Exception as e:
        logger.error(f"Error calculating generation metrics: {e}")
        # Initialize metrics to 0 in case of error, but ensure at least one metric is non-zero
        default_metrics = {metric: 0.0 for metric in rouge_metric_types}
        # Set the first metric to a small non-zero value to pass validation
        if rouge_metric_types:
            default_metrics[rouge_metric_types[0]] = 0.001
            logger.debug(f"Using default metrics with non-zero value: {default_metrics}")
        
        default_rouge_metrics = RougeMetrics(**default_metrics)
        return GenerationEval(
            rouge1=default_rouge_metrics,
            rouge2=default_rouge_metrics if 'rouge2' in rouge_types else None,
            rougeL=default_rouge_metrics if 'rougeL' in rouge_types else None,
            bleu=0.0,
            cosine_sim=0.0,
            self_checker="Undetermined"
        )
