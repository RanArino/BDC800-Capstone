# core/utils/llm.py

import yaml
from pathlib import Path
from typing import Literal, List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama.llms import OllamaLLM

from .path import get_project_root

class LLMController:
    """Controller class for managing both LLM and embedding models"""
    
    def __init__(
        self,
        llm_model: str,
        embedding_model: str,
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        # Load config
        config_path = get_project_root() / "core/configs/models.yaml"
        with open(config_path, "r") as f:
            self.config: dict = yaml.safe_load(f)

        # Initialize LLM
        self.llm = self._get_llm_models(llm_model)
        self.embedding = self._get_embedding_models(embedding_model)

    def _get_llm_models(self, model_name: str) -> OllamaLLM:
        """Get the model config for a given model name
        
        Args:
            model_name: Name of the model to get config for
            
        Returns:
            Model config
        """
        model_config = self.config["llms"].get(model_name)
        return OllamaLLM(model=model_config)
    
    def _get_embedding_models(self, model_name: str) -> HuggingFaceEmbeddings | GoogleGenerativeAIEmbeddings:
        """Get the model config for a given model name
        
        Args:
            model_name: Name of the model to get config for
            
        Returns:
            Model config
        """
        if model_name.startswith("huggingface"):
            return HuggingFaceEmbeddings(
                model_name=self.config["embeddings"].get(model_name)
            )
        elif model_name.startswith("google"):
            return GoogleGenerativeAIEmbeddings(
                model=self.config["embeddings"].get(model_name)
            )
        else:
            raise ValueError(f"Invalid embedding model: {model_name}")

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
    

    def generate_text(self, prompt: str) -> str:
        """Generate text using the LLM
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text response
        """
        return self.llm.invoke(prompt)
