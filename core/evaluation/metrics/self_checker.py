# core/evaluation/metrics/self_checker.py

"""
This module provides functionality to check if the reasoning for an answer is logically valid
and consistent with the given question using Gemini AI.
"""

import os
from typing import Literal, Union, Optional, List
import google.generativeai as genai
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document

from core.evaluation.schema import SelfCheckerAnswer, SelfCheckerModel
from core.logger.logger import get_logger

logger = get_logger(__name__)

# Configuration options - can be overridden with environment variables
DEFAULT_SELF_CHECKER_MODEL = os.environ.get("DEFAULT_SELF_CHECKER_MODEL", "phi4:14b")

# Flag to track if Gemini is available
GEMINI_AVAILABLE = False

# Systen prompt
SYSTEM_PROMPT = """You are an AI evaluator. Your task is to determine if the 'Generated Answer' is factually correct and consistent with the 'Ground Truth Answer' for the given 'Question'. Respond ONLY with "Yes" or "No". Do not provide explanations or any other text."""

# Try to configure Gemini if API key is available
try:
    if "GEMINI_API_KEY" in os.environ:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        GEMINI_AVAILABLE = True
    else:
        logger.warning("GEMINI_API_KEY not found in environment variables. Gemini models will not be available.")
except Exception as e:
    logger.warning(f"Failed to configure Gemini: {e}. Gemini models will not be available.")

# Base generation configuration
base_generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
}

# Gemini-specific configuration
gemini_generation_config = {
    **base_generation_config,
    "max_output_tokens": 64,
    "response_mime_type": "text/plain",
}

# Ollama-specific configuration
ollama_generation_config = {
    **base_generation_config,
    "num_predict": 64,  # equivalent to max_output_tokens
}

# Lazy loading model cache
_MODEL_INSTANCES = {}

def get_model(model_name: Optional[SelfCheckerModel] = None) -> Union[OllamaLLM, genai.GenerativeModel, None]:
    """
    Get or create a model instance with lazy loading and caching.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        The model instance or None if the model is not available
    """
    if model_name not in _MODEL_INSTANCES:
        logger.info(f"Initializing {model_name} model (first use)")
        
        # Create the appropriate model based on model_name
        if model_name == "phi4:14b":
            _MODEL_INSTANCES[model_name] = OllamaLLM(
                model="phi4:14b",
                temperature=ollama_generation_config["temperature"],
                top_p=ollama_generation_config["top_p"],
                top_k=ollama_generation_config["top_k"],
                num_predict=ollama_generation_config["num_predict"],
                system_instruction=SYSTEM_PROMPT,
            )
        elif model_name == "deepseek-r1:14b":
            _MODEL_INSTANCES[model_name] = OllamaLLM(
                model="deepseek-r1:14b",
                temperature=ollama_generation_config["temperature"],
                top_p=ollama_generation_config["top_p"],
                top_k=ollama_generation_config["top_k"],
                num_predict=ollama_generation_config["num_predict"],
                system_instruction=SYSTEM_PROMPT,
            )
        elif model_name == "gemini-2.0-flash":
            if not GEMINI_AVAILABLE:
                logger.warning(f"Cannot initialize {model_name}: Gemini is not available")
                return None
            _MODEL_INSTANCES[model_name] = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=gemini_generation_config,
                system_instruction=SYSTEM_PROMPT,
            )
        else:
            raise ValueError(f"Invalid model: {model_name}")
            
    return _MODEL_INSTANCES.get(model_name)

def check_llm_answer(
        qa_id: str,
        question: str, 
        ground_truth_answer: str,
        llm_answer: str,
        model: SelfCheckerModel = None
    ) -> SelfCheckerAnswer:
    """
    Check if the LLM's reasoning/answer aligns with the ground truth answer.
    
    Args:
        qa_id (str): The unique identifier for the QA pair
        question (str): The original question
        ground_truth_answer (str): The ground truth answer
        llm_answer (str): The LLM generated answer/reasoning
        model (str, optional): The model to use for self-checking. If None, uses the default model from environment.
        
    Returns:
        SelfCheckerAnswer: Contains evaluation result ('Yes'/'No'/'Undetermined')
                          indicating if LLM's answer aligns with ground truth
    """
    # Use the default model if none specified
    model_to_use = model if model is not None else DEFAULT_SELF_CHECKER_MODEL
    
    try:
        model_instance = get_model(model_to_use)
        if model_instance is None:
            logger.warning(f"Self-checker (qa_id: {qa_id}) - Model {model_to_use} not available, skipping check")
            return "Undetermined"
    except ValueError as e:
        logger.error(f"Invalid model: {e}")
        return "Undetermined"
    
    prompt = f"""
Question:
{question}

Ground Truth Answer: 
{ground_truth_answer}

Generated Answer: 
{llm_answer}
"""
    
    try:
        # Initialize chat session if using Gemini
        chat_session = None
        if isinstance(model_instance, genai.GenerativeModel):
            chat_session = model_instance.start_chat(history=[])
        
        def get_response(prompt_text: str) -> str:
            """Get response from model and normalize it"""
            if isinstance(model_instance, genai.GenerativeModel):
                response = chat_session.send_message(prompt_text)
                return response.text.strip().lower()
            else:  # OllamaLLM
                response = model_instance.invoke(prompt_text)
                return response.strip().lower()
        
        def check_response(response_text: str) -> Optional[SelfCheckerAnswer]:
            """Check if response contains yes/no and return appropriate result"""
            if "yes" in response_text:
                logger.info(f"Self-checker (qa_id: {qa_id}) - Yes")
                return "Yes"
            elif "no" in response_text:
                logger.info(f"Self-checker (qa_id: {qa_id}) - No")
                return "No"
            return None
        
        # First attempt
        first_response = get_response(prompt)
        result = check_response(first_response)
        if result:
            return result
        
        # Second attempt with explicit prompt
        retry_prompt = "Please answer ONLY with 'Yes' or 'No'. Is the LLM's answer aligned with the ground truth answer?"
        second_response = get_response(retry_prompt)
        result = check_response(second_response)
        if result:
            return result
        
        # If both attempts fail to get Yes/No
        logger.info(f"Self-checker (qa_id: {qa_id}) - Undetermined")
        return "Undetermined"
        
    except Exception as e:
        logger.error(f"Error during self-checking: {e}")
        return "Undetermined"

def check_retrieval_chunks(
        qa_id: str,
        question: str,
        ground_truth_answer: str,
        retrieval_chunks: List[Document],
        top_k: Union[int, List[int]] = None,
        model: SelfCheckerModel = None
    ) -> Union[SelfCheckerAnswer, dict[int, SelfCheckerAnswer]]:
    """
    Check if the retrieved chunks provide enough information to answer the question.
    
    Args:
        qa_id (str): The unique identifier for the QA pair
        question (str): The original question
        ground_truth_answer (str): The ground truth answer
        retrieval_chunks (List[Document]): The retrieved Langchain Documents
        top_k (Union[int, List[int]], optional): Number(s) of top chunks to evaluate.
                                               Can be a single integer or a list of integers.
                                               If None, uses all chunks.
        model (str, optional): The model to use for self-checking. If None, uses the default model.
        
    Returns:
        Union[SelfCheckerAnswer, dict[int, SelfCheckerAnswer]]: 
            If top_k is an integer or None: Contains evaluation result ('Yes'/'No'/'Undetermined')
            If top_k is a list: Dictionary mapping each top_k value to its evaluation result
    """
    # Use the default model if none specified
    model_to_use = model if model is not None else DEFAULT_SELF_CHECKER_MODEL
    
    try:
        model_instance = get_model(model_to_use)
    except ValueError as e:
        logger.error(f"Invalid model: {e}")
        if isinstance(top_k, list):
            return {k: 0.0 for k in top_k}  # Return 0.0 (No) for all k values
        return 0.0  # Return 0.0 (No)
    
    # Handle the case where top_k is a list
    if isinstance(top_k, list):
        results = {}
        # Sort top_k to evaluate smaller values first
        sorted_k_values = sorted(top_k)
        
        for k in sorted_k_values:
            # Skip if k is more than the number of the received chunks
            if k > len(retrieval_chunks):
                continue

            # If we already have a "Yes" result for a smaller k,
            # automatically assign "Yes" for all larger k values
            smaller_k_with_yes = next((
                smaller_k for smaller_k in results 
                if smaller_k < k and results[smaller_k] == 1.0
            ), None)
            
            if smaller_k_with_yes is not None:
                logger.debug(f"Auto-assigning 'Yes' for top_{k} chunks for qa_id: {qa_id} (based on top_{smaller_k_with_yes} result)")
                results[k] = 1.0
                continue

            logger.info(f"Evaluating retrieval with top_{k} chunks for qa_id: {qa_id}")
            result = _evaluate_chunks(
                qa_id=qa_id,
                question=question,
                ground_truth_answer=ground_truth_answer,
                retrieval_chunks=retrieval_chunks[:k],
                model_instance=model_instance
            )
            results[k] = result
        return results
    
    # Handle the case where top_k is an integer or None
    chunks_to_evaluate = retrieval_chunks[:top_k] if top_k is not None else retrieval_chunks
    return _evaluate_chunks(
        qa_id=qa_id,
        question=question,
        ground_truth_answer=ground_truth_answer,
        retrieval_chunks=chunks_to_evaluate,
        model_instance=model_instance
    )

def _evaluate_chunks(
        qa_id: str,
        question: str,
        ground_truth_answer: str,
        retrieval_chunks: List[Document],
        model_instance: Union[OllamaLLM, genai.GenerativeModel]
    ) -> float:
    """
    Helper function to evaluate a specific set of chunks.
    
    Args:
        qa_id (str): The unique identifier for the QA pair
        question (str): The original question
        ground_truth_answer (str): The ground truth answer
        retrieval_chunks (List[Document]): The chunks to evaluate
        model_instance: The model instance to use for evaluation
        
    Returns:
        float: 1.0 for "Yes", 0.0 for "No" or "Undetermined"
    """
    # Format chunks for the prompt
    formatted_chunks = ""
    for i, chunk in enumerate(retrieval_chunks):
        formatted_chunks += f"Chunk {i+1}:\n{chunk.page_content}\n\n"
    
    prompt = f"""
Question: {question}
Ground Truth Answer: {ground_truth_answer}

Retrieved Information:
{formatted_chunks}

Evaluate if the retrieved chunks provide enough information to answer the question accurately.

Please respond with ONLY 'Yes' if the retrieved chunks provide sufficient information to answer the question accurately, or 'No' if critical information is missing.
"""
    
    try:
        # Initialize chat session if using Gemini
        chat_session = None
        if isinstance(model_instance, genai.GenerativeModel):
            chat_session = model_instance.start_chat(history=[])
        
        def get_response(prompt_text: str) -> str:
            """Get response from model and normalize it"""
            if isinstance(model_instance, genai.GenerativeModel):
                response = chat_session.send_message(prompt_text)
                return response.text.strip().lower()
            else:  # OllamaLLM
                response = model_instance.invoke(prompt_text)
                return response.strip().lower()
        
        def check_response(response_text: str) -> Optional[float]:
            """Check if response contains yes/no and return appropriate result"""
            if "yes" in response_text:
                logger.info(f"Retrieval-checker (qa_id: {qa_id}, chunks: {len(retrieval_chunks)}) - Yes")
                return 1.0
            elif "no" in response_text:
                logger.info(f"Retrieval-checker (qa_id: {qa_id}, chunks: {len(retrieval_chunks)}) - No")
                return 0.0
            return None
        
        # First attempt
        first_response = get_response(prompt)
        result = check_response(first_response)
        if result is not None:
            return result
        
        # Second attempt with explicit prompt
        retry_prompt = "Please answer ONLY with 'Yes' or 'No'. Do the retrieved chunks provide enough information to answer the question accurately?"
        second_response = get_response(retry_prompt)
        result = check_response(second_response)
        if result is not None:
            return result
        
        # If both attempts fail to get Yes/No
        logger.info(f"Retrieval-checker (qa_id: {qa_id}, chunks: {len(retrieval_chunks)}) - Undetermined")
        return 0.0  # Return 0.0 for "Undetermined"
        
    except Exception as e:
        logger.error(f"Error during retrieval checking: {e}")
        return 0.0  # Return 0.0 for "Undetermined"
