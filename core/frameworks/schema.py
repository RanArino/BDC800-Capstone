# core/frameworks/schema.py

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, PositiveInt

from langchain_core.documents import Document

# Available Models & Algorithms
AVAILABLE_LLM_ID = Literal["llama3.2:1b", "llama3.2:3b", "llama3.1", "phi4", "deepseek-r1-8b", "deepseek-r1-14b"]
AVAILABLE_EMBEDDING_ID = Literal["huggingface-multi-qa-mpnet", "google-gecko"]
AVAILABLE_FAISS_SEARCH = Literal["flatl2", "ivf", "hnsw"]
AVAILABLE_DIM_REDUCTION = Literal["pca", "umap"]
AVAILABLE_CLUSTERING = Literal["kmeans", "gmm"]

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

class SummarizerConfig(BaseModel):
    """Configuration for the summarizer component."""
    llm_id: AVAILABLE_LLM_ID = Field(..., description="ID of the language model to use")
    output_tokens: PositiveInt = Field(..., description="Expected number of tokens to output")
    embedding_id: AVAILABLE_EMBEDDING_ID = Field(..., description="ID of the embedding model to use")

class DimReductionConfig(BaseModel):
    """Configuration for dimensionality reduction."""
    method: AVAILABLE_DIM_REDUCTION = Field(..., description="Method for dimensionality reduction")
    n_components: PositiveInt = Field(..., description="Number of components for dimensionality reduction")

class ClusteringConfig(BaseModel):
    """Configuration for clustering."""
    method: AVAILABLE_CLUSTERING = Field(..., description="Clustering method to use")
    n_clusters: Optional[PositiveInt] = Field(None, description="Number of clusters (if None, will be estimated)")
    items_per_cluster: PositiveInt = Field(..., description="Number of items per cluster (used if n_clusters is None)")

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
    dim_reduction: Optional[DimReductionConfig] = Field(None, description="Configuration for dimensionality reduction")
    clustering: Optional[ClusteringConfig] = Field(None, description="Configuration for clustering")

class RetrievalGenerationConfig(BaseModel):
    """Configuration for the retrieval component."""
    faiss_search: AVAILABLE_FAISS_SEARCH = Field(..., description="FAISS index type for vector search")
    top_k: PositiveInt = Field(..., description="Number of top document chunks to retrieve")
    top_k_doc_cc: Optional[PositiveInt] = Field(None, description="Number of top document clusters to retrieve")
    top_k_doc: Optional[PositiveInt] = Field(None, description="Number of top documents to retrieve")
    top_k_chunk_cc: Optional[PositiveInt] = Field(None, description="Number of top chunk clusters to retrieve")
    llm_id: Optional[AVAILABLE_LLM_ID] = Field(None, description="ID of the Language Model to use")

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
        
class HierarchicalFilterOption(BaseModel):
    """Schema for the layered filtering option."""
    layer: AVAILABLE_LAYERS = Field(..., description="The layer to filter by")
    parent_node_id: Optional[PARENT_NODE_ID] = Field(None, description="The parent node ID to filter by")
