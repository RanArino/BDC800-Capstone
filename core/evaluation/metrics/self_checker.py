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

def check_reasoning(
        qa_id: str,
        question: str, 
        answer: str, 
        reasoning: str
    ) -> SelfCheckerAnswer:
    """
    Check if the reasoning for an answer is logically valid and consistent.
    
    Args:
        question (str): The original question
        answer (str): The provided answer
        reasoning (str): The reasoning behind the answer
        
    Returns:
        str: 'Yes' if reasoning is valid and consistent, 'No' otherwise.
             Returns 'Undetermined' if LLM fails to provide clear answer.
    """
    try:
        chat_session = model.start_chat(history=[])
        
        prompt = f"""
Question: {question}
Answer: {answer}
Reasoning: {reasoning}

Is the reasoning logically valid and consistent with both the question and answer?
Please respond with ONLY 'Yes' or 'No'.
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
        retry_prompt = "Please answer ONLY with 'Yes' or 'No'. Is the reasoning logically valid and consistent?"
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

