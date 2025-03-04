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
from core.utils import save_layer_models, load_layer_models

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
        
        # Store clustering models
        self.clustering_models = {
            "doc": None,
            "chunk": {},
        }

    def index(
        self, 
        docs: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]], 
    ):
        """Index documents using SCALER's hierarchical approach.
        
        This method:
        1. Attempts to load existing indexes if available
        2. If some indexes exist, loads them and creates only the missing ones
        3. If no indexes exist or loading fails, creates all new indexes:
            - Creates document-level summaries and embeddings
            - Chunks documents and creates embeddings
            - Creates layered vector stores with clustering
            
        Args:
            documents: List of documents to index
        """
        # Try to load existing indexes first
        try:
            loaded_layers = self._load_all_indexes()
            if all(loaded_layers.values()):
                self.logger.info("Successfully loaded all existing indexes")
                return
            elif any(loaded_layers.values()):
                self.logger.info(f"Partially loaded indexes. Will create missing layers: {[layer for layer, loaded in loaded_layers.items() if not loaded]}")
        except Exception as e:
            self.logger.warning(f"Failed to load existing indexes: {str(e)}. Will create new indexes.")
            loaded_layers = {layer: False for layer in self.layered_vector_stores.keys()}

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
                # Skip document-level processing if both doc and doc_cc are already loaded
                if not (loaded_layers["doc"] and loaded_layers["doc_cc"]):
                    # Skip LLM summary only if docs is a single SchemaDocument
                    if not isinstance(docs, SchemaDocument):
                        summary, sum_embed = self._doc_summary(doc)
                        doc_summary.append(summary)
                        doc_summary_embed.append(sum_embed)

                # Skip chunk processing if both chunk and chunk_cc are already loaded for this document
                chunk_key = str(doc.id)
                # Check if this document has already been processed
                if chunk_key in self.layered_vector_stores["chunk"] or any(k.startswith(chunk_key + "-") for k in self.layered_vector_stores["chunk"].keys()):
                    self.logger.debug(f"Skipping chunk processing for document {chunk_key} as it's already indexed")
                else:
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
            
            # Create or update document summary vector store if we have multiple documents
            if num_docs > 1 and doc_summary and doc_summary_embed:
                # Always recreate the doc and doc_cc layers when adding new documents
                # This ensures proper clustering with all documents
                self._layered_vector_store(
                    layer="doc",
                    embeddings=doc_summary_embed,
                    doc_summary=doc_summary,
                    parent_node_id=None
                )
            
            # Save all indexes if enabled
            if self.is_save_vectorstore:
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
        # Check if embeddings is empty
        if not embeddings or len(embeddings) == 0:
            self.logger.warning(f"Empty embeddings provided to _layered_vector_store for layer {layer}. Skipping vector store creation.")
            return
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Conduct dimensional reduction if configured
        if hasattr(self.config.chunker, "dim_reduction"):
            # Run dimensional reduction
            dim_model = run_dim_reduction(
                embeddings=embeddings_array, 
                method=self.config.chunker.dim_reduction.method, 
                n_components=getattr(self.config.chunker.dim_reduction, "n_components", None)
            )
            
            # Store the dimensional reduction model for later use with queries
            if layer == "doc":
                self.dim_reduction_models[layer] = dim_model
            elif layer == "chunk" and parent_node_id:
                self.dim_reduction_models[layer][parent_node_id] = dim_model
                
            # Apply dimension reduction to embeddings
            embeddings_array = dim_model.transform(embeddings_array)
        
        # Conduct Clustering if configured
        if hasattr(self.config.chunker, "clustering"):
            # Run clustering
            clustering_model = run_clustering(
                embeddings_array,
                method=self.config.chunker.clustering.method,
                n_clusters=getattr(self.config.chunker.clustering, "n_clusters", None),
                items_per_cluster=getattr(self.config.chunker.clustering, "items_per_cluster", None)
            )
            
            # Skip if no model returned
            if clustering_model is None:
                self.logger.warning(f"No clustering model returned for layer {layer}. Skipping vector store creation.")
                return
                
            # Store the clustering model for later use
            if layer == "doc":
                self.clustering_models["doc"] = clustering_model
            elif layer == "chunk" and parent_node_id:
                self.clustering_models["chunk"][parent_node_id] = clustering_model
                
            # Get labels and centroids from the model
            if hasattr(clustering_model, 'labels_'):
                # KMeans
                labels = clustering_model.labels_.tolist()
                centroids = {i: clustering_model.cluster_centers_[i] for i in range(len(clustering_model.cluster_centers_))}
            else:
                # GMM
                labels = clustering_model.predict(embeddings_array).tolist()
                centroids = {i: clustering_model.means_[i] for i in range(len(clustering_model.means_))}
                
            # Create mapping from cluster to indices
            clusters_to_indices = {}
            for i, label in enumerate(labels):
                if label not in clusters_to_indices:
                    clusters_to_indices[label] = []
                clusters_to_indices[label].append(i)
            
            # Check if centroids is empty
            if not centroids:
                self.logger.warning(f"No centroids extracted from clustering model for layer {layer}. Skipping vector store creation.")
                return
            
            # Store cluster centroids
            if layer == "doc":
                # Store cluster centroids for document layer
                centroid_embeddings = [(f"doc_cc-{label}", vector) for label, vector in centroids.items()]
                if centroid_embeddings:  # Only create if we have embeddings
                    self.layered_vector_stores["doc_cc"] = FAISS.from_embeddings(
                        text_embeddings=centroid_embeddings,
                        embedding=self.llm.get_embedding
                    )
            elif layer == "chunk" and parent_node_id:
                # Store cluster centroids for chunk layer
                centroid_embeddings = [(f"{parent_node_id}-{label}", vector) for label, vector in centroids.items()]
                if centroid_embeddings:  # Only create if we have embeddings
                    self.layered_vector_stores["chunk_cc"][parent_node_id] = FAISS.from_embeddings(
                        text_embeddings=centroid_embeddings,
                        embedding=self.llm.get_embedding
                    )
            else:
                raise ValueError(f"Invalid layer: {layer}")

            # Store the embeddings and chunks or summaries per cluster
            for cluster, embed_indices in clusters_to_indices.items():
                if not embed_indices:  # Skip if no indices for this cluster
                    continue
                
                texts_and_embeddings = []
                metadatas = []
                cluster_parent_node_id = parent_node_id + f"-{str(cluster)}" if parent_node_id else str(cluster)
                
                # Extract the embeddings and texts which are associated with the current cluster
                for idx in embed_indices:
                    if layer == "doc":
                        texts_and_embeddings.append((doc_summary[idx], embeddings[idx]))
                        metadatas.append({"layer": layer, "cluster": cluster})
                    elif layer == "chunk":
                        texts_and_embeddings.append((chunks[idx].page_content, embeddings[idx]))
                        metadatas.append({"layer": layer, "cluster": cluster, **chunks[idx].metadata})

                # Store in the appropriate layer if we have embeddings
                if texts_and_embeddings:
                    self.layered_vector_stores[layer][cluster_parent_node_id] = FAISS.from_embeddings(
                        text_embeddings=texts_and_embeddings,
                        metadatas=metadatas,
                        embedding=self.llm.get_embedding
                    )
        else:
            # If no clustering is configured, store all embeddings in a single vector store
            texts_and_embeddings = []
            metadatas = []
            
            for idx, embedding in enumerate(embeddings):
                if layer == "doc":
                    texts_and_embeddings.append((doc_summary[idx], embedding))
                    metadatas.append({"layer": layer})
                elif layer == "chunk":
                    texts_and_embeddings.append((chunks[idx].page_content, embedding))
                    metadatas.append({"layer": layer, **chunks[idx].metadata})
            
            # Store in the appropriate layer if we have embeddings
            if texts_and_embeddings:
                store_key = parent_node_id if parent_node_id else layer
                self.layered_vector_stores[layer][store_key] = FAISS.from_embeddings(
                    text_embeddings=texts_and_embeddings,
                    metadatas=metadatas,
                    embedding=self.llm.get_embedding
                )

                # Store dimension reduction model even when clustering is disabled
                if hasattr(self.config.chunker, "dim_reduction"):
                    if layer == "doc":
                        self.dim_reduction_models[layer] = dim_model
                    elif layer == "chunk" and parent_node_id:
                        self.dim_reduction_models[layer][parent_node_id] = dim_model
    
    def _save_all_indexes(self):
        """Save all indexes to disk."""
        if not self.is_save_vectorstore:
            self.logger.info("Skipping index saving as is_save_vectorstore is False")
            return
            
        try:
            self.logger.debug("Starting to save all indexes")
            
            # Create base directory if it doesn't exist
            os.makedirs(self.vectorstore_path, exist_ok=True)
            
            # Save each layer
            for layer, vector_store in self.layered_vector_stores.items():
                if vector_store is None:
                    continue
                    
                # Create layer directory
                layer_path = os.path.join(self.vectorstore_path, layer)
                os.makedirs(layer_path, exist_ok=True)
                
                if isinstance(vector_store, FAISS):
                    # Handle single FAISS instance (doc_cc layer)
                    save_path = os.path.join(layer_path, f"{layer}.faiss")
                    vector_store.save_local(save_path)
                    self.logger.debug(f"Saved {layer} index to {save_path}")
                elif isinstance(vector_store, dict):
                    # Handle dictionary of FAISS instances (doc, chunk, and chunk_cc layers)
                    for node_id, vs in vector_store.items():
                        if vs is not None:
                            save_path = os.path.join(layer_path, f"{node_id}.faiss")
                            vs.save_local(save_path)
                            self.logger.debug(f"Saved {layer} index for node {node_id} to {save_path}")
                            
                            # Save associated ML models if they exist
                            if layer in ["doc", "chunk"]:
                                save_layer_models(layer, node_id, layer_path, self.dim_reduction_models, self.clustering_models)
            
            self.logger.info("Successfully saved all indexes")
            
        except Exception as e:
            self.logger.error(f"Error saving indexes: {str(e)}")
            raise

    def _load_all_indexes(self) -> Dict[str, bool]:
        """
        Load all indexes from disk.
                
        Returns:
            Dict[str, bool]: Dictionary indicating which layers were successfully loaded
        """
        try:
            self.logger.debug("Starting to load all indexes")
            
            # Check if the base directory exists
            if not os.path.exists(self.vectorstore_path):
                self.logger.info(f"No existing indexes found at {self.vectorstore_path}")
                return {layer: False for layer in self.layered_vector_stores.keys()}
                
            # Initialize empty vector stores
            self.layered_vector_stores = {
                "doc": {},
                "chunk": {},
                "doc_cc": None,
                "chunk_cc": {}
            }
            
            # Initialize empty model stores
            self.dim_reduction_models = {
                "doc": None,
                "chunk": {}
            }
            
            # Initialize empty clustering models
            self.clustering_models = {
                "doc": None,
                "chunk": {}
            }
            
            # Track which layers were loaded
            loaded_layers = {layer: False for layer in self.layered_vector_stores.keys()}
            
            # Load each layer
            for layer in self.layered_vector_stores.keys():
                layer_path = os.path.join(self.vectorstore_path, layer)
                
                if not os.path.exists(layer_path):
                    self.logger.debug(f"No index directory found for layer {layer}")
                    continue
                    
                # Handle single FAISS instance layer (doc_cc)
                if layer == "doc_cc":
                    index_path = os.path.join(layer_path, f"{layer}.faiss")
                    if os.path.exists(index_path):
                        self.layered_vector_stores[layer] = FAISS.load_local(
                            index_path,
                            self.llm.get_embedding,
                            allow_dangerous_deserialization=True
                        )
                        self.logger.debug(f"Loaded {layer} index from {index_path}")
                        loaded_layers[layer] = True
                
                # Handle dictionary FAISS instance layers (doc, chunk, and chunk_cc)
                else:
                    # Get all .faiss files in the directory
                    faiss_files = [f for f in os.listdir(layer_path) if f.endswith('.faiss')]
                    if faiss_files:  # Only mark as loaded if we found some files
                        loaded_layers[layer] = True
                        for faiss_file in faiss_files:
                            node_id = faiss_file[:-6]  # Remove .faiss extension
                            index_path = os.path.join(layer_path, faiss_file)
                            self.layered_vector_stores[layer][node_id] = FAISS.load_local(
                                index_path,
                                self.llm.get_embedding,
                                allow_dangerous_deserialization=True
                            )
                            self.logger.debug(f"Loaded {layer} index for node {node_id} from {index_path}")
            
                            # Try to load associated ML models if they exist
                            if layer in ["doc", "chunk"]:
                                dim_model, cluster_model = load_layer_models(layer, node_id, layer_path)
                                
                                # Store dimension reduction model if loaded
                                if dim_model is not None:
                                    if layer == "doc":
                                        self.dim_reduction_models[layer] = dim_model
                                    elif layer == "chunk":
                                        # For chunk layer, extract the parent document ID if needed
                                        node_key = node_id.split('-')[0] if '-' in node_id else node_id
                                        self.dim_reduction_models[layer][node_key] = dim_model
                                
                                # Store clustering model if loaded
                                if cluster_model is not None:
                                    if layer == "doc":
                                        self.clustering_models[layer] = cluster_model
                                    elif layer == "chunk":
                                        # For chunk layer, extract the parent document ID if needed
                                        node_key = node_id.split('-')[0] if '-' in node_id else node_id
                                        self.clustering_models[layer][node_key] = cluster_model
            
            loaded_count = sum(loaded_layers.values())
            if loaded_count == 0:
                self.logger.info("No indexes were loaded")
            else:
                self.logger.info(f"Successfully loaded {loaded_count} layers: {[layer for layer, loaded in loaded_layers.items() if loaded]}")
            
                # Log information about loaded ML models
                if any(model is not None for model in [self.dim_reduction_models["doc"]] + list(self.dim_reduction_models["chunk"].values())):
                    self.logger.info("Successfully loaded dimension reduction models")
                if any(model is not None for model in [self.clustering_models["doc"]] + list(self.clustering_models["chunk"].values())):
                    self.logger.info("Successfully loaded clustering models")
            
            return loaded_layers
            
        except Exception as e:
            self.logger.error(f"Error loading indexes: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[LangChainDocument]:
        """Retrieve relevant documents using SCALER's hierarchical approach."""
        pass