# core/frameworks/schema.py

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, PositiveInt

from langchain_core.documents import Document

# Available Models & Algorithms
AVAILABLE_LLM_ID = Literal["llama3.1", "phi4", "deepseek-r1-8b", "deepseek-r1-14b"]
AVAILABLE_EMBEDDING_ID = Literal["huggingface-multi-qa-mpnet", "google-gecko"]
AVAILABLE_FAISS_SEARCH = Literal["flatl2", "ivf", "hnsw"]
AVAILABLE_DIM_REDUCTION = Literal["pca", "umap"]
AVAILABLE_CLUSTERING = Literal["k-means", "gmm"]

# Available Layers
AVAILABLE_LAYERS = Literal["doc_cc", "doc", "chunk_cc", "chunk"]

# Other Declarations
PARENT_NODE_ID = str


class DatasetConfig(BaseModel):
    """Configuration for the dataset component."""
    name: str = Field(..., description="Name of the dataset to use")
    number_of_docs: Optional[PositiveInt] = Field(None, description="Number of documents to use from the dataset")
    number_of_qas: Optional[PositiveInt] = Field(None, description="Number of question-answer pairs to use from the dataset")
    selection_mode: Optional[Literal["sequential", "random"]] = Field(None, description="Selection mode for the dataset")

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
    dim_reduction: Optional[AVAILABLE_DIM_REDUCTION] = Field(None, description="Method for dimensionality reduction (e.g., PCA, UMAP)")
    clustering: Optional[AVAILABLE_CLUSTERING] = Field(None, description="Clustering method to use (e.g., k-means, GMM)")


class RetrievalGenerationConfig(BaseModel):
    """Configuration for the retrieval component."""
    faiss_search: AVAILABLE_FAISS_SEARCH = Field(..., description="FAISS index type for vector search")
    top_k: PositiveInt = Field(..., description="Number of top documents to retrieve")
    llm_id: Optional[AVAILABLE_LLM_ID] = Field(..., description="ID of the Language Model to use")

class RAGConfig(BaseModel):
    """Main configuration for the RAG system."""
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
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
        
class HierarchicalFilterOption(BaseModel):
    """Schema for the layered filtering option."""
    layer: AVAILABLE_LAYERS = Field(..., description="The layer to filter by")
    parent_node_id: Optional[PARENT_NODE_ID] = Field(None, description="The parent node ID to filter by")
