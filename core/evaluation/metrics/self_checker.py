# core/evaluation/metrics/self_checker.py

"""
This module provides functionality to check if the reasoning for an answer is logically valid
and consistent with the given question using Gemini AI.
"""

import os
from typing import Literal, Union, Optional
import google.generativeai as genai
from langchain_ollama.llms import OllamaLLM

from core.evaluation.schema import SelfCheckerAnswer, SefCheckerModel
from core.logger.logger import get_logger

logger = get_logger(__name__)

# Configuration options - can be overridden with environment variables
DEFAULT_SELF_CHECKER_MODEL = os.environ.get("DEFAULT_SELF_CHECKER_MODEL", "phi4:14b")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

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

def get_model(model_name: SefCheckerModel) -> Union[OllamaLLM, genai.GenerativeModel]:
    """
    Get or create a model instance with lazy loading and caching.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        The model instance
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
                system_instruction="Given the following question, answer, and reasoning, determine if the reasoning for the answer is logically valid and consistent with question and the answer.\\",
            )
        elif model_name == "deepseek-r1:14b":
            _MODEL_INSTANCES[model_name] = OllamaLLM(
                model="deepseek-r1:14b",
                temperature=ollama_generation_config["temperature"],
                top_p=ollama_generation_config["top_p"],
                top_k=ollama_generation_config["top_k"],
                num_predict=ollama_generation_config["num_predict"],
                system_instruction="Given the following question, answer, and reasoning, determine if the reasoning for the answer is logically valid and consistent with question and the answer.\\",
            )
        elif model_name == "gemini-2.0-flash":
            _MODEL_INSTANCES[model_name] = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=gemini_generation_config,
                system_instruction="Given the following question, answer, and reasoning, determine if the reasoning for the answer is logically valid and consistent with question and the answer.\\",
            )
        else:
            raise ValueError(f"Invalid model: {model_name}")
            
    return _MODEL_INSTANCES[model_name]

def check_llm_answer(
        qa_id: str,
        question: str, 
        ground_truth_answer: str,
        llm_answer: str,
        model: SefCheckerModel = None
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
    except ValueError as e:
        logger.error(f"Invalid model: {e}")
        return "Undetermined"
    
    prompt = f"""
Question: {question}
Answer: {ground_truth_answer}
Reasoning: {llm_answer}

Evaluate if the LLM's answer captures the key information and meaning from the ground truth answer.
Consider:
1. Factual accuracy compared to ground truth
2. Key concepts coverage
3. No contradictions with ground truth

Please respond with ONLY 'Yes' if the LLM's answer is sufficiently aligned with ground truth, or 'No' if there are significant discrepancies.
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
