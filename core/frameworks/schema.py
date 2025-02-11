# core/frameworks/schema.py

from typing import Optional, Literal
from pydantic import BaseModel, Field, PositiveInt

# Available Models
AVAILABLE_LLM_ID = Literal["llama3.1", "phi4", "deepseek-r1-8b", "deepseek-r1-14b"]
AVAILABLE_EMBEDDING_ID = Literal["huggingface-multi-qa-mpnet", "google-gecko"]
AVAILABLE_FAISS_SEARCH = Literal["flatl2", "ivf", "hnsw"]

class DatasetConfig(BaseModel):
    """Configuration for the dataset component."""
    name: str = Field(..., description="Name of the dataset to use")
    number_of_docs: Optional[PositiveInt] = Field(None, description="Number of documents to use from the dataset")

class ChunkerConfig(BaseModel):
    """Configuration for the text chunking component."""
    mode: Literal["fixed"] = Field(..., description="Chunking mode (currently only fixed is supported)")
    size: PositiveInt = Field(..., description="Size of each chunk in tokens or characters")
    overlap: float = Field(
        ..., 
        description="Overlap between chunks as a fraction (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

class ModelConfig(BaseModel):
    """Configuration for the model components."""
    llm_id: AVAILABLE_LLM_ID = Field(
        ..., 
        description="ID of the Language Model to use"
    )
    embedding_id: AVAILABLE_EMBEDDING_ID = Field(
        ..., 
        description="ID of the Embedding Model to use"
    )

class RetrievalConfig(BaseModel):
    """Configuration for the retrieval component."""
    faiss_search: AVAILABLE_FAISS_SEARCH = Field(..., description="FAISS index type for vector search")
    top_k: PositiveInt = Field(..., description="Number of top documents to retrieve")

class RAGConfig(BaseModel):
    """Main configuration for the RAG system."""
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    chunker: ChunkerConfig = Field(..., description="Text chunking configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    retrieval: RetrievalConfig = Field(..., description="Retrieval configuration") 