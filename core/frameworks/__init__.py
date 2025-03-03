# core/frameworks/__init__.py

from .schema import RAGConfig, DatasetConfig, ChunkerConfig, RetrievalGenerationConfig, RAGResponse
from .simple import SimpleRAG
from .scaler import ScalerRAG

all = [
    "SimpleRAG",
    "ScalerRAG",
    "RAGConfig",
    "DatasetConfig",
    "ChunkerConfig",
    "RetrievalGenerationConfig",
    "RAGResponse"
]