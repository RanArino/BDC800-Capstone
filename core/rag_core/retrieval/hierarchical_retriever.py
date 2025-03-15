# core/rag_core/retrieval/hierarchical_retriever.py
"""
Hierarchical Retriever Module

This module provides functionality for retrieving documents and chunks in a hierarchical manner,
using document-level similarity to prune the search space for chunks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set

class HierarchicalRetriever:
    """
    Retriever for hierarchical document-chunk relationships.
    
    This class handles the retrieval of documents and their chunks in a hierarchical manner,
    using document-level similarity to prune the search space for chunks.
    """
    def __init__(self, indexer):
        """
        Initialize the hierarchical retriever.
        
        Args:
            indexer: HierarchicalIndexer instance containing the indexed documents and chunks
        """
        self.indexer = indexer
        
        # Cache for recent queries
        self.query_cache = {}  # (query_hash, top_k_docs, top_k_chunks) -> results
        self.cache_size = 100  # Maximum number of cached queries
        
    def search_documents(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Search for the most similar documents.
        
        Args:
            query: Query vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, document_ids)
        """
        # Ensure query is properly shaped
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
            
        # Convert to float32 if needed
        if query.dtype != np.float32:
            query = query.astype('float32')
            
        # Search document index
        distances, indices = self.indexer.doc_index.search(query, min(k, self.indexer.doc_index.ntotal))
        
        # Get document IDs
        doc_ids = [self.indexer.doc_ids[int(idx)] for idx in indices[0] if idx >= 0 and idx < len(self.indexer.doc_ids)]
        
        return distances[0][:len(doc_ids)], doc_ids
        
    def search_chunks_by_parent(self, query: np.ndarray, parent_id: str, k: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Search for the most similar chunks within a specific parent document.
        
        Args:
            query: Query vector
            parent_id: Parent document ID to search within
            k: Number of results to return
            
        Returns:
            Tuple of (distances, chunk_ids)
        """
        # Check if parent exists and has children
        if parent_id not in self.indexer.parent_to_children or not self.indexer.parent_to_children[parent_id]:
            return np.array([]), []
            
        # Get chunk vectors for this parent
        chunk_vectors = self.indexer.parent_to_chunk_vectors[parent_id]
        chunk_ids = self.indexer.parent_to_chunk_ids[parent_id]
        
        # If there are no chunks, return empty results
        if chunk_vectors is None or len(chunk_vectors) == 0:
            return np.array([]), []
            
        # Ensure query is properly shaped
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
            
        # Convert to float32 if needed
        if query.dtype != np.float32:
            query = query.astype('float32')
        
        # Calculate distances directly using numpy (faster than FAISS for small sets)
        if self.indexer.metric == "l2":
            # L2 distance
            distances = np.sqrt(np.sum((chunk_vectors - query) ** 2, axis=1))
        else:
            # Inner product (convert to distance by negating)
            distances = -np.sum(chunk_vectors * query, axis=1)
        
        # Get top k indices
        if len(distances) <= k:
            top_indices = np.argsort(distances)
        else:
            top_indices = np.argsort(distances)[:k]
            
        # Return results
        result_distances = distances[top_indices]
        result_ids = [chunk_ids[i] for i in top_indices]
        
        return result_distances, result_ids
            
    def hierarchical_search(self, query: np.ndarray, top_k_docs: int = 3, top_k_chunks: int = 10) -> Tuple[List[str], List[str], List[float]]:
        """
        Perform hierarchical search by first finding relevant documents, then searching their chunks.
        
        Args:
            query: Query vector
            top_k_docs: Number of top documents to consider
            top_k_chunks: Number of top chunks to return overall
            
        Returns:
            Tuple of (top_doc_ids, top_chunk_ids, top_chunk_distances)
        """
        # Check cache first
        # Ensure query is a numpy array
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
            
        query_hash = hash(query.tobytes())
        cache_key = (query_hash, top_k_docs, top_k_chunks)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Ensure query is properly shaped
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
            
        # Convert to float32 if needed
        if query.dtype != np.float32:
            query = query.astype('float32')
        
        # Step 1: Find relevant documents
        doc_distances, doc_ids = self.search_documents(query, top_k_docs)
        
        # Step 2: Search chunks within those documents
        all_chunk_results = []
        
        # Process each parent document
        for doc_id in doc_ids:
            chunk_distances, chunk_ids = self.search_chunks_by_parent(query, doc_id, top_k_chunks)
            
            # Add to results
            for i, (chunk_id, distance) in enumerate(zip(chunk_ids, chunk_distances)):
                all_chunk_results.append((chunk_id, doc_id, float(distance)))
                
        # Sort by distance and take top k
        all_chunk_results.sort(key=lambda x: x[2])
        all_chunk_results = all_chunk_results[:top_k_chunks]
        
        # Extract results
        top_chunk_ids = [x[0] for x in all_chunk_results]
        top_doc_ids = [x[1] for x in all_chunk_results]
        top_distances = [x[2] for x in all_chunk_results]
        
        # Cache results
        result = (top_doc_ids, top_chunk_ids, top_distances)
        self.query_cache[cache_key] = result
        
        # Limit cache size
        if len(self.query_cache) > self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        return result
        
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        if doc_id in self.indexer.parent_to_children:
            doc = self.indexer.get_document(doc_id)
            if doc:
                return {
                    "id": doc_id,
                    "num_chunks": len(self.indexer.parent_to_children[doc_id]),
                    "document": doc
                }
            return {
                "id": doc_id,
                "num_chunks": len(self.indexer.parent_to_children[doc_id])
            }
        return None
        
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chunk metadata by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk metadata or None if not found
        """
        if chunk_id in self.indexer.chunk_ids:
            idx = self.indexer.chunk_ids.index(chunk_id)
            parent_id = self.indexer.chunk_parent_ids[idx]
            doc = self.indexer.get_chunk_document(chunk_id)
            if doc:
                return {
                    "id": chunk_id,
                    "parent_id": parent_id,
                    "document": doc
                }
            return {
                "id": chunk_id,
                "parent_id": parent_id
            }
        return None
        
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear() 