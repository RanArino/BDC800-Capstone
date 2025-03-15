# core/frameworks/scaler_v2.py

"""
ScalerV2RAG Framework

This module implements an optimized hierarchical search approach that properly pre-filters vectors
based on document similarity before searching chunks, addressing the performance issues in ScalerV1.

Key improvements:
1. Uses raw FAISS for efficient vector operations
2. Maintains explicit document-chunk relationships
3. Prunes search space by first finding relevant documents, then only searching their chunks
4. Optimized for the specific use case of document-chunk hierarchical search
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Iterable, Any, Set, Generator

from langchain_core.documents import Document as LangChainDocument
from core.datasets.schema import Document as SchemaDocument
from core.rag_core.indexing import HierarchicalIndexer
from core.rag_core.retrieval import HierarchicalRetriever
from core.frameworks.base import BaseRAGFramework
from core.rag_core import run_doc_summary

# Define available layers in the hierarchy
AVAILABLE_LAYERS = str
PARENT_NODE_ID = str


class ScalerV2RAG(BaseRAGFramework):
    """
    ScalerV2 RAG Framework that uses hierarchical search to efficiently retrieve relevant chunks.
    
    This implementation inherits from BaseRAGFramework and overrides the index and retrieve methods
    to use the hierarchical indexing and retrieval approach.
    """
    def __init__(
        self, 
        config_name: str, 
        config_path: str = "core/configs/scaler_v2_rag/test.yaml", 
        is_save_vectorstore: bool = False
    ):
        """
        Initialize the ScalerV2 RAG framework.
        
        Args:
            config_name: Name of the configuration to use
            config_path: Path to the configuration file
            is_save_vectorstore: Whether to save the vector store after indexing
        """
        # Initialize the base class
        super().__init__(config_name, config_path, is_save_vectorstore)
        
        # Initialize indexer and retriever
        self.indexer = None
        self.retriever = None
        
    def index(
        self,
        docs: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]], 
    ):
        """
        Index documents for retrieval.
        
        Args:
            docs: Documents to index
        """
        # Update vectorstore path after setting document ID
        self.vectorstore_path = self._define_vectorstore_path(docs)

        # Try to load existing indexes first
        try:
            loaded = self._load_index(self.vectorstore_path)
            if loaded:
                self.logger.info("Successfully loaded existing index")
                return
        except Exception as e:
            self.logger.warning(f"Failed to load existing index: {str(e)}. Will create new index.")
        
        # Ensure documents are in a generator
        gen_docs = self._ensure_document_generator(docs)
        
        # Initialize indexer if not already done
        if self.indexer is None:
            embedding_dim = self.llm.get_embedding_dim
            if callable(embedding_dim):
                embedding_dim = embedding_dim()
            self.indexer = HierarchicalIndexer(dimension=embedding_dim)
            self.retriever = HierarchicalRetriever(indexer=self.indexer)
        
        try:
            self.logger.debug("Starting document indexing")
            # Process each document with progress tracking
            num_docs = 0
            doc_sum_embed = {}
            doc_summary = {}  # doc id is key, summary is value
            
            for doc in gen_docs:
                # Run document summary
                doc_summary[doc.id], doc_sum_embed[doc.id] = self._doc_summary(doc)
                self.logger.debug(f"Completed document {doc.id} summary")
                
                # Process chunks
                chunks, chunks_embed = self._doc_chunking(doc)
                
                # Add to indexer
                with self.profiler.track("indexing.add_documents"):
                    # Create document LangChainDocument
                    doc_langchain = LangChainDocument(
                        page_content=doc_summary[doc.id],
                        metadata={"id": doc.id, "layer": "doc"}
                    )
                    
                    self.indexer.add_documents(
                        vectors=np.array([doc_sum_embed[doc.id]]), 
                        ids=[doc.id],
                        documents=[doc_langchain]
                    )
                
                with self.profiler.track("indexing.add_chunks"):
                    chunk_ids = [f"{doc.id}_chunk_{i}" for i in range(len(chunks))]
                    parent_ids = [doc.id] * len(chunks)
                    
                    # Update chunk metadata
                    for i, chunk in enumerate(chunks):
                        chunk.metadata.update({
                            "id": chunk_ids[i],
                            "document_id": doc.id,
                            "chunk_index": i,
                            "layer": "chunk"
                        })
                    
                    self.indexer.add_chunks(
                        vectors=np.array(chunks_embed),
                        ids=chunk_ids,
                        parent_ids=parent_ids,
                        documents=chunks
                    )
                
                self.logger.debug(f"Completed chunking for document {doc.id}")
                num_docs += 1
            
            # Save vector store if requested
            if self.is_save_vectorstore:
                self._save_vectorstore()
                
            self.logger.info("Indexing complete")
                
        except Exception as e:
            self.logger.error(f"Error during indexing: {str(e)}")
            raise
            
        return self
    
    def _doc_summary(self, doc: SchemaDocument) -> Tuple[str, List[float]]:
        """
        Summarize document and get embedding
        """
        with self.profiler.track("index.doc_summary"):
            summary = run_doc_summary(doc.content)

        # Get embeddings
        embedding = self.llm.embedding.embed_documents([summary])        
        return summary, embedding[0]
        
    def _doc_chunking(self, doc: SchemaDocument) -> Tuple[List[LangChainDocument], List[List[float]]]:
        """
        Chunk document
        """
        with self.profiler.track("index.chunking"):
            chunks = self.chunker.run([doc], mode=self.chunker_config.mode)
        # Get embeddings
        embeddings = self.llm.embedding.embed_documents([chunk.page_content for chunk in chunks])
        return chunks, embeddings
        
    def retrieve(
        self, 
        query: str, 
        top_k_per_layer: Optional[Dict[str, int]] = None
    ) -> List[LangChainDocument]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: The query to retrieve documents for.
            top_k_per_layer: Dictionary with top_k values for each layer.
            
        Returns:
            A list of documents that are relevant to the query.
        """
        try:
            self.logger.debug(f"Starting retrieval for query: {query}")
            
            if self.indexer is None or self.retriever is None:
                raise ValueError("Indexer and retriever not initialized. Please index documents first.")
                
            # Use config's top_k if not specified
            if top_k_per_layer is None:
                config = self.retrieval_generation_config
                top_k_per_layer = {
                    "doc": getattr(config, "top_k_doc", None) or 5,
                    "chunk": getattr(config, "top_k", None) or 10
                }
                
            # Get query embedding
            query_embedding = self.llm.embedding.embed_query(query)
                
            # Ensure query_embedding is a numpy array
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Perform hierarchical search
            with self.profiler.track("retrieval.search"):
                doc_ids, chunk_ids, chunk_distances = self.retriever.hierarchical_search(
                    query=query_embedding,
                    top_k_docs=top_k_per_layer["doc"],
                    top_k_chunks=top_k_per_layer["chunk"]
                )
            
            # Get LangChainDocument objects directly from the indexer
            results: List[LangChainDocument] = []
            
            for chunk_id in chunk_ids:
                doc = self.indexer.get_chunk_document(chunk_id)
                if doc is not None:
                    results.append(doc)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise
        
    def _save_vectorstore(self):
        """Save the vector store to disk."""
        if not self.is_save_vectorstore:
            self.logger.info("Skipping index saving as is_save_vectorstore is False")
            return
            
        if self.indexer is not None:
            try:
                self.logger.debug("Starting to save index")
                
                # Create directory if it doesn't exist
                os.makedirs(self.vectorstore_path, exist_ok=True)
                
                # Save the indexer to disk
                self.indexer.save(self.vectorstore_path)
                
                self.logger.info(f"Vector store saved to {self.vectorstore_path}")
            except Exception as e:
                self.logger.error(f"Error saving index: {str(e)}")
                raise
            
    def _load_index(self, index_path: str) -> bool:
        """
        Load the index from the given path.
        
        Returns:
            bool: Whether the index was successfully loaded
        """
        try:
            self.logger.debug(f"Loading vector store from: {index_path}")
            
            # Check if the path exists
            if not os.path.exists(index_path):
                self.logger.info(f"No existing index found at {index_path}")
                return False
                
            # Check if required files exist
            doc_index_path = os.path.join(index_path, "doc_index.faiss")
            chunk_index_path = os.path.join(index_path, "chunk_index.faiss")
            metadata_path = os.path.join(index_path, "metadata.pkl")
            
            if not (os.path.exists(doc_index_path) and 
                    os.path.exists(chunk_index_path) and 
                    os.path.exists(metadata_path)):
                self.logger.info(f"Missing required index files in {index_path}")
                return False
                
            # Initialize indexer if not already done
            if self.indexer is None:
                embedding_dim = self.llm.get_embedding_dim
                if callable(embedding_dim):
                    embedding_dim = embedding_dim()
                self.indexer = HierarchicalIndexer(dimension=embedding_dim)
                self.retriever = HierarchicalRetriever(indexer=self.indexer)
            
            # Load the indexer from disk
            self.indexer.load(index_path)
            
            self.logger.info(f"Vector store loaded successfully from {index_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            return False
