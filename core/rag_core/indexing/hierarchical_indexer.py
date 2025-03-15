# core/rag_core/indexing/hierarchical_indexer.py
"""
Hierarchical Indexer Module

This module provides functionality for indexing documents and chunks in a hierarchical structure,
maintaining parent-child relationships between documents and their chunks.
"""

import os
import numpy as np
import faiss
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any

from langchain_core.documents import Document as LangChainDocument

class HierarchicalIndexer:
    """
    Indexer for hierarchical document-chunk relationships.
    
    This class handles the indexing of documents and their chunks, maintaining
    the parent-child relationships between them for efficient retrieval.
    """
    def __init__(self, dimension: int, metric: str = "l2"):
        """
        Initialize the hierarchical indexer.
        
        Args:
            dimension: Dimension of the vectors
            metric: Distance metric to use ("l2" or "ip" for inner product)
        """
        self.dimension = dimension
        self.metric = metric
        
        # Create separate indices for documents and chunks
        if metric == "l2":
            self.doc_index = faiss.IndexFlatL2(dimension)
            self.chunk_index = faiss.IndexFlatL2(dimension)
        else:  # Inner product
            self.doc_index = faiss.IndexFlatIP(dimension)
            self.chunk_index = faiss.IndexFlatIP(dimension)
            
        # Document metadata
        self.doc_ids = []  # List of document IDs in order they were added
        self.doc_store = {}  # Map from doc_id to document content
        
        # Chunk metadata
        self.chunk_ids = []  # List of chunk IDs in order they were added
        self.chunk_parent_ids = []  # List of parent document IDs for each chunk
        self.chunk_store = {}  # Map from chunk_id to LangChainDocument
        
        # Mapping from parent ID to list of child chunk indices
        self.parent_to_children = {}  # parent_id -> list of indices in chunk_ids
        
        # Optimization: Store chunk vectors by parent for faster retrieval
        self.parent_to_chunk_vectors = {}  # parent_id -> numpy array of chunk vectors
        self.parent_to_chunk_ids = {}  # parent_id -> list of chunk IDs
        
    def add_documents(self, vectors: np.ndarray, ids: List[str], documents: Optional[List[LangChainDocument]] = None):
        """
        Add document vectors to the store.
        
        Args:
            vectors: Document vectors of shape (n, dimension)
            ids: Document IDs corresponding to the vectors
            documents: Optional list of LangChainDocument objects
        """
        if len(vectors) == 0:
            return
            
        # Convert to numpy array if needed
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors).astype('float32')
        
        # Add to document index
        self.doc_index.add(vectors)
        
        # Store document IDs
        start_idx = len(self.doc_ids)
        self.doc_ids.extend(ids)
        
        # Store documents if provided
        if documents is not None:
            for doc_id, doc in zip(ids, documents):
                self.doc_store[doc_id] = doc
        
        # Initialize parent-child mappings
        for doc_id in ids:
            if doc_id not in self.parent_to_children:
                self.parent_to_children[doc_id] = []
                self.parent_to_chunk_vectors[doc_id] = None
                self.parent_to_chunk_ids[doc_id] = []
                
    def add_chunks(
        self, 
        vectors: np.ndarray, 
        ids: List[str], 
        parent_ids: List[str], 
        documents: List[LangChainDocument]
    ):
        """
        Add chunk vectors to the store with their parent document IDs.
        
        Args:
            vectors: Chunk vectors of shape (n, dimension)
            ids: Chunk IDs corresponding to the vectors
            parent_ids: Parent document IDs for each chunk
            documents: LangChainDocument objects for each chunk
        """
        if len(vectors) == 0:
            return
            
        # Convert to numpy array if needed
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors).astype('float32')
        
        # Add to chunk index
        self.chunk_index.add(vectors)
        
        # Store chunk metadata
        start_idx = len(self.chunk_ids)
        self.chunk_ids.extend(ids)
        self.chunk_parent_ids.extend(parent_ids)
        
        # Store chunk documents
        for chunk_id, doc in zip(ids, documents):
            self.chunk_store[chunk_id] = doc
        
        # Update parent-child mappings
        # Group chunks by parent for more efficient processing
        parent_to_chunks = {}
        for i, parent_id in enumerate(parent_ids):
            if parent_id not in parent_to_chunks:
                parent_to_chunks[parent_id] = []
            parent_to_chunks[parent_id].append((start_idx + i, ids[i], vectors[i]))
        
        # Update mappings for each parent
        for parent_id, chunks in parent_to_chunks.items():
            # Initialize parent mappings if they don't exist
            if parent_id not in self.parent_to_children:
                self.parent_to_children[parent_id] = []
            if parent_id not in self.parent_to_chunk_vectors:
                self.parent_to_chunk_vectors[parent_id] = None
            if parent_id not in self.parent_to_chunk_ids:
                self.parent_to_chunk_ids[parent_id] = []
            
            # Add chunk indices
            chunk_indices = [chunk[0] for chunk in chunks]
            self.parent_to_children[parent_id].extend(chunk_indices)
            
            # Add chunk IDs
            chunk_ids = [chunk[1] for chunk in chunks]
            self.parent_to_chunk_ids[parent_id].extend(chunk_ids)
            
            # Add chunk vectors
            chunk_vectors = np.array([chunk[2] for chunk in chunks])
            if self.parent_to_chunk_vectors[parent_id] is None:
                self.parent_to_chunk_vectors[parent_id] = chunk_vectors
            else:
                self.parent_to_chunk_vectors[parent_id] = np.vstack([
                    self.parent_to_chunk_vectors[parent_id], 
                    chunk_vectors
                ])
                
    def save(self, path: str):
        """
        Save the indices and metadata to disk.
        
        Args:
            path: Directory path to save the indices and metadata
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS indices
        faiss.write_index(self.doc_index, os.path.join(path, "doc_index.faiss"))
        faiss.write_index(self.chunk_index, os.path.join(path, "chunk_index.faiss"))
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "metric": self.metric,
            "doc_ids": self.doc_ids,
            "chunk_ids": self.chunk_ids,
            "chunk_parent_ids": self.chunk_parent_ids,
            "parent_to_children": self.parent_to_children,
            "parent_to_chunk_ids": self.parent_to_chunk_ids,
            "doc_store": self.doc_store,
            "chunk_store": self.chunk_store
        }
        
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
            
        # Save parent_to_chunk_vectors separately (can be large)
        for parent_id, vectors in self.parent_to_chunk_vectors.items():
            if vectors is not None:
                np.save(os.path.join(path, f"chunk_vectors_{parent_id}.npy"), vectors)
        
    def load(self, path: str):
        """
        Load indices and metadata from disk.
        
        Args:
            path: Directory path to load the indices and metadata from
        """
        # Load FAISS indices
        self.doc_index = faiss.read_index(os.path.join(path, "doc_index.faiss"))
        self.chunk_index = faiss.read_index(os.path.join(path, "chunk_index.faiss"))
        
        # Load metadata
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            
        self.dimension = metadata["dimension"]
        self.metric = metadata["metric"]
        self.doc_ids = metadata["doc_ids"]
        self.chunk_ids = metadata["chunk_ids"]
        self.chunk_parent_ids = metadata["chunk_parent_ids"]
        self.parent_to_children = metadata["parent_to_children"]
        self.parent_to_chunk_ids = metadata["parent_to_chunk_ids"]
        
        # Handle backward compatibility
        self.doc_store = metadata.get("doc_store", {})
        self.chunk_store = metadata.get("chunk_store", {})
        
        # Load parent_to_chunk_vectors
        self.parent_to_chunk_vectors = {}
        for parent_id in self.parent_to_children.keys():
            vector_path = os.path.join(path, f"chunk_vectors_{parent_id}.npy")
            if os.path.exists(vector_path):
                self.parent_to_chunk_vectors[parent_id] = np.load(vector_path)
            else:
                self.parent_to_chunk_vectors[parent_id] = None
        
    def get_chunk_document(self, chunk_id: str) -> Optional[LangChainDocument]:
        """
        Get the LangChainDocument for a chunk.
        
        Args:
            chunk_id: ID of the chunk
            
        Returns:
            LangChainDocument for the chunk, or None if not found
        """
        return self.chunk_store.get(chunk_id)
        
    def get_document(self, doc_id: str) -> Optional[LangChainDocument]:
        """
        Get the LangChainDocument for a document.
        
        Args:
            doc_id: ID of the document
            
        Returns:
            LangChainDocument for the document, or None if not found
        """
        return self.doc_store.get(doc_id)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed data.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "num_documents": len(self.doc_ids),
            "num_chunks": len(self.chunk_ids),
            "doc_index_size": self.doc_index.ntotal,
            "chunk_index_size": self.chunk_index.ntotal,
            "dimension": self.dimension,
            "metric": self.metric
        } 