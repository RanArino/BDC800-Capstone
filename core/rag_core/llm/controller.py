# core/rag_core/llm/controller.py

import os
# Set tokenizers parallelism to avoid warnings when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
from typing import Dict
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama.llms import OllamaLLM
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from core.utils.path import get_project_root
from .schema import LLMConfig, EmbeddingConfig

SYSTEM_PROMPT  = """You are an AI assistant. Your task is to answer questions based *solely* on the provided context. Be factual and concise. Do not add information or explanations not explicitly found in the text."""

class LLMController:
    """Controller class for managing both LLM and embedding models"""
    
    def __init__(
        self,
        llm_id: str,
        embedding_id: str,
    ):
        # Load and parse configs
        config_path = get_project_root() / "core/configs/models.yaml"
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
            
        # Parse into typed dictionaries
        llm_config_data = raw_config["llms"].get(llm_id)
        if llm_config_data is None:
            raise ValueError(f"LLM model '{llm_id}' not found in configuration. Available models: {list(raw_config['llms'].keys())}")
        self.llm_config = LLMConfig(**llm_config_data)
        
        embedding_config_data = raw_config["embeddings"].get(embedding_id)
        if embedding_config_data is None:
            raise ValueError(f"Embedding model '{embedding_id}' not found in configuration. Available models: {list(raw_config['embeddings'].keys())}")
        self.embedding_config = EmbeddingConfig(**embedding_config_data)

        # Initialize models
        self.llm = self._init_llm_models()
        self.embedding = self._init_embedding_models(embedding_id)

    def _init_llm_models(self) -> OllamaLLM | genai.GenerativeModel:
        """Get the model config for a given model name
        
        Args:
            model_name: Name of the model to get config for
            
        Returns:
            Model config
        """
        if self.llm_config.model_name.startswith("gemini"):
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
            return genai.GenerativeModel(
                model_name="gemini-1.5-flash-8b",
                safety_settings=safety_settings,
                system_instruction=SYSTEM_PROMPT
                )

        else:
            return OllamaLLM(model=self.llm_config.model_name, system_instruction=SYSTEM_PROMPT)
    
    def _init_embedding_models(self, model_id: str) -> HuggingFaceEmbeddings | GoogleGenerativeAIEmbeddings:
        """Get the model config for a given model name
        
        Args:
            model_id: Embedding model id to get config for (check core/configs/models.yaml)
            
        Returns:
            Model config
        """
        if model_id.startswith("huggingface"):
            return HuggingFaceEmbeddings(
                model_name=self.embedding_config.model_name,
            )
        elif model_id.startswith("google"):
            return GoogleGenerativeAIEmbeddings(
                model=self.embedding_config.model_name,
            )
        else:
            raise ValueError(f"Invalid embedding model: {model_id}")

    @property
    def get_llm(self):
        """Get the LLM model for use with other LangChain components
        
        Returns:
            Langchain LLM model
        """
        return self.llm
    
    @property
    def get_embedding(self):
        """Get the embedding model for use with other LangChain components
        
        Returns:
            Langchain embedding model
        """
        return self.embedding
    
    @property
    def get_embedding_dim(self):
        """Get the dimension of the embedding model
        
        Returns:
            Dimension of the embedding model
        """
        return self.embedding_config.dimension

    def generate_text(self, prompt: str) -> str:
        """Generate text using the LLM
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text response with <think></think> tags removed
        """
        if isinstance(self.llm, genai.GenerativeModel):
            try:
                chat_session = self.llm.start_chat(history=[])
                return chat_session.send_message(prompt).text
            except Exception as e:
                # If BLOCKLIST error occurs, return empty string or appropriate fallback
                if "BLOCKLIST" in str(e):
                    return ""  # or return appropriate fallback message
                raise  # re-raise other exceptions
        else:
            # Get raw response from LLM
            raw_response = self.llm.invoke(prompt)
            
            # Remove <think></think> tags and their content
            cleaned_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
            
            # Remove any empty lines that might result from the removal
            cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)
            
            return cleaned_response.strip()
