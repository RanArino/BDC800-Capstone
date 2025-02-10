# core/rag_core/llm/schema.py

from pydantic import BaseModel, Field
from typing import Literal

AVAILABLE_LLMS = Literal["llama3.1:8b", "phi4:14b", "deepseek-r1:8b", "deepseek-r1:14b"]
AVAILABLE_EMBEDDINGS = Literal["sentence-transformers/multi-qa-mpnet-base-cos-v1", "text-embedding-005"]
AVAILABLE_PROVIDERS = Literal["ollama", "huggingface", "google"]


class LLMConfig(BaseModel):
    """Configuration for the LLM component."""
    model_name: AVAILABLE_LLMS = Field(..., description="Name of the LLM model to use")
    provider: AVAILABLE_PROVIDERS = Field(..., description="Provider of the LLM model")
    

class EmbeddingConfig(BaseModel):
    """Configuration for the LLM component."""
    model_name: AVAILABLE_EMBEDDINGS = Field(..., description="Name of the LLM model to use")
    input_length: int = Field(..., description="Maximum input length for the LLM model")
    dimension: int = Field(..., description="Dimension of the LLM model")

