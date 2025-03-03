# core/frameworks/scaler.py

"""
Implementation of the SCALER (Semantic Clustered Abstractive Layers for Efficient Retrieval) framework.
"""

import os
import logging
import numpy as np
from typing import List, Optional, Union, Generator, Iterable, Dict, Any, Tuple, Literal

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument

from core.datasets import Document as SchemaDocument

from core.frameworks.base import BaseRAGFramework
from core.frameworks.schema import RAGConfig, RAGResponse, AVAILABLE_LAYERS, PARENT_NODE_ID, HierarchicalFilterOption

from core.rag_core import run_doc_summary, run_dim_reduction, run_clustering, reduce_query_embedding

logger = logging.getLogger(__name__)

class ScalerRAG(BaseRAGFramework):
    """
    SCALER (Semantic Clustered Abstractive Layers for Efficient Retrieval) framework.
    
    This framework implements a hierarchical approach to RAG with:
    1. Document-level summaries
    2. Semantic clustering of chunks
    3. Dimensional reduction for efficient retrieval
    """
    
    def __init__(self, config_name: str, config_path: str = "core/configs/scaler_rag/test.yaml", is_save_vectorstore: bool = False):
        """Initialize the SCALER framework.
        
        Args:
            config: RAG configuration
        """
        super().__init__(config_name, config_path, is_save_vectorstore)
        
        # Initialize layered vector stores
        self.layered_vector_stores: Dict[AVAILABLE_LAYERS, Union[FAISS, Dict[PARENT_NODE_ID, FAISS]]] = {
            "doc": {},
            "chunk": {},
            "doc_cc": None,
            "chunk_cc": {}
        }
        
        # Store dimensional reduction models
        self.dim_reduction_models = {
            "doc": None,
            "chunk": {},
        }

    def index(
        self, 
        docs: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]], 
    ):
        """Index documents using SCALER's hierarchical approach.
        
        This method:
      
        
        Args:
            documents: List of documents to index
        """
        # TODO: Load index if exists

        # ensure docs is a generator
        gen_docs = self._ensure_document_generator(docs)
        
        # Index document
        try:
            self.logger.debug("Starting document indexing")
            # Summarize document and get embedding
            num_docs = 0
            doc_summary = []
            doc_summary_embed = []
            for doc in gen_docs:
                # LLM summary
                summary, sum_embed = self._doc_summary(doc)
                doc_summary.append(summary)
                doc_summary_embed.append(sum_embed)

                # Chunking
                chunks, chunks_embed = self._doc_chunking(doc)
                # Create layered vector store
                self._layered_vector_store(
                    layer="chunk",
                    embeddings=chunks_embed,
                    chunks=chunks,
                    parent_node_id=doc.id
                )
                num_docs += 1
            
            # if a single document is indexed, no need to create a document summary vector store
            if num_docs > 1:
                # Create document summary vector store
                self._layered_vector_store(
                    layer="doc",
                    embeddings=doc_summary_embed,
                    doc_summary=doc_summary,
                    parent_node_id=None
                )
                
            # Save all indexes
            self._save_all_indexes()
            
            self.logger.info("Indexing complete")
            
        except Exception as e:
            self.logger.error(f"Error during indexing: {str(e)}")
            raise

    def _doc_summary(self, doc: SchemaDocument) -> Tuple[str, List[float]]:
        """
        Summarize document and get embedding
        """
        with self.profiler.track("index.doc_summary"):
            summary = run_doc_summary(doc.content)

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

    def _layered_vector_store(
            self, 
            layer: Literal["doc", "chunk"], 
            embeddings: List[List[float]],
            doc_summary: Optional[List[str]] = None,
            chunks: Optional[List[LangChainDocument]] = None,
            parent_node_id: Optional[PARENT_NODE_ID] = None
        ):
        """Create layered vector stores for efficient retrieval."""
        # Conduct Clustering
        cluster_method = self.config.chunker.clustering
        
        # Conduct dimensional reduction
        dim_method = self.config.chunker.dim_reduction
        if dim_method:
            # TODO: Consider adding number of components in ChunkerConfig
            embeddings, dim_model = run_dim_reduction(embeddings, dim_method, n_components=50)
            
            # Store the dimensional reduction model for later use with queries
            if layer == "doc":
                self.dim_reduction_models["doc"] = dim_model
            elif layer == "chunk" and parent_node_id:
                self.dim_reduction_models["chunk"][parent_node_id] = dim_model
            
            # Use reduced embeddings for clustering
            _, centroids, clusters_to_indices = run_clustering(
                embeddings,
                method=cluster_method,
                n_clusters=getattr(self.config.chunker.clustering, 'n_clusters', None),
            )

        # Initialize dictionaries if they don't exist
        if layer == "doc":
            # Store cluster centroids for document layer
            self.layered_vector_stores["doc_cc"] = FAISS.from_embeddings(
                text_embeddings=[(f"doc_cc-{label}", vector) for label, vector in centroids.items()],
                embedding=self.llm.get_embedding
            )
        elif layer == "chunk" and parent_node_id:
            # Store cluster centroids for chunk layer
            self.layered_vector_stores["chunk_cc"][parent_node_id] = FAISS.from_embeddings(
                text_embeddings=[(f"chunk_cc-{label}", vector) for label, vector in centroids.items()],
                embedding=self.llm.get_embedding
            )
        else:
            raise ValueError(f"Invalid layer: {layer}")

        # Store the embeddings and chunks or summaries per cluster
        for cluster, embed_indices in clusters_to_indices.items():
            texts_and_embeddings = []
            metadatas = []
            parent_node_id = parent_node_id + f"-{str(cluster)}" if parent_node_id else str(cluster)
            
            # Extract the embeddings and texts which are associated with the current cluster
            for idx in embed_indices:
                if layer == "doc":
                    texts_and_embeddings.append((doc_summary[idx], embeddings[idx]))
                    metadatas.append({"layer": layer, "cluster": cluster})
                elif layer == "chunk":
                    texts_and_embeddings.append((chunks[idx].page_content, embeddings[idx]))
                    metadatas.append({"layer": layer, "cluster": cluster, **chunks[idx].metadata})

            # Store in the appropriate layer
            self.layered_vector_stores[layer][parent_node_id] = FAISS.from_embeddings(
                text_embeddings=texts_and_embeddings,
                metadatas=metadatas,
                embedding=self.llm.get_embedding
            )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[LangChainDocument]:
        """Retrieve relevant documents using SCALER's hierarchical approach."""
        pass