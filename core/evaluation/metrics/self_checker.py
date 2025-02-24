# core/evaluation/metrics/self_checker.py

"""
This module provides functionality to check if the reasoning for an answer is logically valid
and consistent with the given question using Gemini AI.
"""

import os
from typing import Literal
import google.generativeai as genai

from core.evaluation.schema import SelfCheckerAnswer
from core.logger.logger import get_logger

logger = get_logger(__name__)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
  system_instruction="Given the following question, answer, and reasoning, determine if the reasoning for the answer is logically valid and consistent with question and the answer.\\",
)

def check_llm_answer(
        qa_id: str,
        question: str, 
        ground_truth_answer: str,
        llm_answer: str
    ) -> SelfCheckerAnswer:
    """
    Check if the LLM's reasoning/answer aligns with the ground truth answer.
    
    Args:
        qa_id (str): The unique identifier for the QA pair
        question (str): The original question
        ground_truth_answer (str): The ground truth answer
        llm_answer (str): The LLM generated answer/reasoning
        
    Returns:
        SelfCheckerAnswer: Contains evaluation result ('Yes'/'No'/'Undetermined')
                          indicating if LLM's answer aligns with ground truth
    """
    try:
        chat_session = model.start_chat(history=[])
        
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
        
        # First attempt
        response = chat_session.send_message(prompt)
        first_response = response.text.strip().lower()
        
        if "yes" in first_response:
            logger.info(f"Self-checker (qa_id: {qa_id}) - Yes")
            return "Yes"
        elif "no" in first_response:
            logger.info(f"Self-checker (qa_id: {qa_id}) - No")
            return "No"
        
        # Second attempt - explicitly ask for Yes/No
        retry_prompt = "Please answer ONLY with 'Yes' or 'No'. Is the LLM's answer aligned with the ground truth answer?"
        response = chat_session.send_message(retry_prompt)
        second_response = response.text.strip().lower()
        
        if "yes" in second_response:
            logger.info(f"Self-checker (qa_id: {qa_id}) - Yes")
            return "Yes"
        elif "no" in second_response:
            logger.info(f"Self-checker (qa_id: {qa_id}) - No")
            return "No"
        
        # If both attempts fail to get Yes/No
        logger.info(f"Self-checker (qa_id: {qa_id}) - Undetermined")
        return "Undetermined"
    finally:
        # Cleanup chat session
        del chat_session

