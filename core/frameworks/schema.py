# core/frameworks/schema.py

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, PositiveInt

from langchain_core.documents import Document

# Available Models
AVAILABLE_LLM_ID = Literal["llama3.1", "phi4", "deepseek-r1-8b", "deepseek-r1-14b"]
AVAILABLE_EMBEDDING_ID = Literal["huggingface-multi-qa-mpnet", "google-gecko"]
AVAILABLE_FAISS_SEARCH = Literal["flatl2", "ivf", "hnsw"]

class DatasetConfig(BaseModel):
    """Configuration for the dataset component."""
    name: str = Field(..., description="Name of the dataset to use")
    number_of_docs: Optional[PositiveInt] = Field(None, description="Number of documents to use from the dataset")
    number_of_qas: Optional[PositiveInt] = Field(None, description="Number of question-answer pairs to use from the dataset")
    selection_mode: Optional[Literal["sequential", "random"]] = Field(None, description="Selection mode for the dataset")

class SummarizerConfig(BaseModel):
    """Configuration for the summarizer component."""
    llm_id: AVAILABLE_LLM_ID = Field(..., description="ID of the language model to use")
    output_tokens: PositiveInt = Field(..., description="Expected number of tokens to output")
    embedding_id: AVAILABLE_EMBEDDING_ID = Field(..., description="ID of the embedding model to use")

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
    embedding_id: AVAILABLE_EMBEDDING_ID = Field(..., description="ID of the Embedding Model to use")
    dim_reduction: Optional[Literal["pca", "umap"]] = Field(None, description="Dimension reduction method to use")
    clustering: Optional[Literal["kmeans", "gmm"]] = Field(None, description="Clustering method to use")
    
class RetrievalGenerationConfig(BaseModel):
    """Configuration for the retrieval component."""
    faiss_search: AVAILABLE_FAISS_SEARCH = Field(..., description="FAISS index type for vector search")
    top_k: PositiveInt = Field(..., description="Number of top documents to retrieve")
    llm_id: Optional[AVAILABLE_LLM_ID] = Field(..., description="ID of the Language Model to use")

class RAGConfig(BaseModel):
    """Main configuration for the RAG system."""
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    summarizer: Optional[SummarizerConfig] = Field(None, description="Summarizer configuration")
    chunker: ChunkerConfig = Field(..., description="Text chunking configuration")
    retrieval_generation: RetrievalGenerationConfig = Field(..., description="Retrieval and generation configuration")

class RAGResponse(BaseModel):
    """Schema for the RAG pipeline response."""
    query: str = Field(..., description="The original query string")
    llm_answer: Optional[str] = Field(None, description="The generated answer from the LLM")
    context: List[Document] = Field(..., description="The retrieved documents used as context")

    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True  # To allow Document objects in context 