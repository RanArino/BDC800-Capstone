# core/frameworks/scaler_v1_2.py

from typing import Dict, Union, Iterable, Generator, Tuple, List, Optional, TYPE_CHECKING
import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
import threading
import gc

from core.datasets import Document as SchemaDocument
from core.rag_core import run_doc_summary, run_dim_reduction, run_clustering, reduce_query_embedding
from core.utils import save_ml_models, load_ml_models

from core.frameworks.base import BaseRAGFramework
from core.frameworks.schema import AVAILABLE_LAYERS, PARENT_NODE_ID

if TYPE_CHECKING:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

class ScalerV2RAG(BaseRAGFramework):
    """
    SCALER(Semantic Clustered Abstractive Layers for Efficient Retrieval) framework with FAISS IVF (Inverted File) Index.
    
    This framework implements two-layered vector stores.
    - Document-cluster level vector store: each vector point represents a document cluster centroid(embedding vector).
    - Chunk-level vector store: each vector point represents a chunk embedding vector.
    - Note that query expansion is applied to get the relevant chunks from the chunk-level vector store.

    """
    def __init__(self, config_name: str, config_path: str = "core/configs/scaler_v1_2_rag/test.yaml", is_save_vectorstore: bool = False):
        """Initialize the SCALER framework.
        
        Args:
            config_name: RAG configuration name
            config_path: RAG configuration path
            is_save_vectorstore: Whether to save the vector store
        """
        
        # Initialize cleanup flags
        self._is_shutting_down = False
        self._cleanup_lock = threading.Lock()
        
        super().__init__(config_name, config_path, is_save_vectorstore)
        
        # Initialize layered vector stores
        self.layered_vector_stores: Dict[AVAILABLE_LAYERS, Union[FAISS, Dict[PARENT_NODE_ID, FAISS]]] = {
            "doc_cc": None,  # each vector point represents a document cluster centroid(embedding vector).
            "doc": None,     # each vector point represents a document embedding vector.
            "chunk": {}      # each vector point represents a chunk embedding vector across each clusterd document.
        }
        
        # Store dimensional reduction models
        self.dim_reduction_models = {
            "doc_cc": None,
        }
        
        # Store clustering models
        self.clustering_models: Dict[AVAILABLE_LAYERS, Optional[Union["KMeans", "GaussianMixture"]]] = {
            "doc_cc": None,
        }

    def __del__(self):
        """Ensure proper cleanup when object is destroyed."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources and release memory."""
        with self._cleanup_lock:
            if self._is_shutting_down:
                return
            self._is_shutting_down = True
            
            try:
                # Clean up vector stores
                if self.layered_vector_stores["doc_cc"] is not None:
                    self.layered_vector_stores["doc_cc"].index.reset()
                if self.layered_vector_stores["doc"] is not None:
                    self.layered_vector_stores["doc"].index.reset()
                for chunk_store in self.layered_vector_stores["chunk"].values():
                    if chunk_store is not None:
                        chunk_store.index.reset()
                
                # Clear references
                self.layered_vector_stores = {
                    "doc_cc": None,
                    "doc": None,
                    "chunk": {}
                }
                
                # Clean up clustering model
                if self.clustering_models["doc_cc"] is not None:
                    del self.clustering_models["doc_cc"]
                    self.clustering_models["doc_cc"] = None
                
                # Force garbage collection
                gc.collect()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")
            finally:
                self._is_shutting_down = False

    def index(
        self, 
        docs: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]], 
    ):
        """
        Index documents using SCALER's hierarchical approach.
        
        This method:
        1. Attempts to load existing indexes if available
        2. If some indexes exist, loads them and creates only the missing ones
        3. If no indexes exist or loading fails, creates all new indexes:
            - Creates document-cluster level summaries and embeddings
            - Chunks documents and creates embeddings
            - Creates layered vector stores with clustering
            
        Args:
            documents: List of documents to index
        """
        # Update vectorstore path after setting document ID
        self.vectorstore_path = self._define_vectorstore_path(docs)
        
        # NOTE: set loaded_layers temporally 
        loaded_layers = {layer: False for layer in self.layered_vector_stores.keys()}

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
            doc_ids = []            # list of document ids
            doc_summary = []        # list of document summaries
            doc_summary_embed = []  # list of document summary embeddings
            chunks_dict = {}        # dictionary of chunks for each document
            
            for doc in gen_docs:
                try:
                    # Skip document-level processing if both doc and doc_cc are already loaded
                    if loaded_layers["doc"]:
                        self.logger.debug(f"Skipping document {doc.id} summary as it's already indexed")
                        continue

                    # Skip document-cluster level processing if it's already loaded
                    if loaded_layers["doc_cc"]:
                        self.logger.debug(f"Skipping document {doc.id} cluster as it's already indexed")
                        continue

                    # Skip document-cluster level processing if it's already loaded
                    if isinstance(docs, SchemaDocument):
                        continue

                    # Generate document summary and embedding
                    summary, sum_embed = self._doc_summary(doc)
                    doc_ids.append(doc.id)
                    doc_summary.append(summary)
                    doc_summary_embed.append(sum_embed)
                    # let summary and sum_embed to be garbage collected
                    del summary, sum_embed
                    self.logger.debug(f"Completed document {doc.id} summary")

                    # Create chunks and chunk embeddings
                    chunks_dict[doc.id] = self._doc_chunking(doc)
                    self.logger.debug(f"Completed chunking for document {doc.id}")
        
                    num_docs += 1
                                            
                except Exception as e:
                    self.logger.error(f"Error processing document {doc.id}: {str(e)}")
                    continue
            
            # Create vector stores
            try:
                # (1) Document-layer
                self._doc_vector_store(doc_ids, doc_summary_embed, doc_summary)

                # (2) Document-cluster level
                labels = self._doc_cc_vector_store(doc_summary_embed)

                # (3) Chunk-level
                self._chunk_vector_store(doc_ids, chunks_dict, labels)
            
            except Exception as e:
                self.logger.error(f"Error creating vector stores: {str(e)}")
                raise

            # Save all indexes if enabled
            if self.is_save_vectorstore:
                self._save_all_indexes()
                self._save_ml_models()
            
            self.logger.info("Indexing complete")
            
        except Exception as e:
            self.logger.error(f"Error during indexing: {str(e)}")
            raise
        finally:
            # Clean up resources
            gc.collect()

    def retrieve(
            self, 
            query: str,
            top_k_per_layer: Optional[Dict[AVAILABLE_LAYERS, int]] = None,
        ) -> List[LangChainDocument]:
        """Retrieve documents from the vector stores.
        
        This method:
        1. Apply query to document(doc) level vector store to get the top k_doc relevant documents.
        2. Combine each document's summary to an original query (query expansion), create k_doc number of expanded queries.
        3. Apply each expanded query to document-cluster level vector store to get the top k_doc_cc relevant document-cluster.
        4. Apply each expanded query to chunk-level vector store to get the top k_chunk relevant chunks.

        Returns:
            List of retrieved chunks(k_chunk)
        """
        if top_k_per_layer is None:
            config = self.retrieval_generation_config
            top_k_per_layer = {
                "doc_cc": getattr(config, "top_k_doc_cc", None) or 3,
                "doc": getattr(config, "top_k_doc", None) or 5,
                "chunk": getattr(config, "top_k", None) or 10
            }
        # Define sets
        doc_ccs_ids: set[PARENT_NODE_ID] = set()
        added_chunk_ids: set[str] = set() 
        retrieved_chunks: List[Tuple[LangChainDocument, float]] = []

        # Apply query to document(doc) level vector store to get the top k_doc relevant documents.
        docs = self.layered_vector_stores["doc"].similarity_search(query, k=top_k_per_layer["doc"])

        # Combine each document's summary to an original query (query expansion), create k_doc number of expanded queries.
        expanded_queries = [query + " " + doc.page_content for doc in docs]

        # Apply each expanded query to document-cluster level vector store to get the top k_doc_cc relevant document-cluster.
        for e_query in expanded_queries:
            if self.dim_reduction_models["doc_cc"]:
                query_embedding = self.llm.embedding.embed_query(e_query)
                search_embedding = reduce_query_embedding(query_embedding, self.dim_reduction_models["doc_cc"])
                doc_ccs = self.layered_vector_stores["doc_cc"].similarity_search_by_vector(search_embedding, k=top_k_per_layer["doc_cc"])
            else:
                doc_ccs = self.layered_vector_stores["doc_cc"].similarity_search(e_query, k=top_k_per_layer["doc_cc"])
            
            doc_ccs_ids.update([doc_cc.metadata["vector_store_key"] for doc_cc in doc_ccs])

        # Apply each expanded query to chunk-level vector store to get the top k_chunk relevant chunks.
        for doc_cc_id in doc_ccs_ids:
            for e_query in expanded_queries:
                search_results = self.layered_vector_stores["chunk"][doc_cc_id].similarity_search_with_score(e_query, k=top_k_per_layer["chunk"])
                for result in search_results:
                    if result[0].id not in added_chunk_ids:
                        added_chunk_ids.add(result[0].id)
                        retrieved_chunks.append(result)
                    else:
                        continue

        # Sort retrieved chunks by score
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x[1]) 
        
        # Return only the top k_chunk documents (without scores)
        return [doc for doc, _ in sorted_chunks[:top_k_per_layer["chunk"]]]
        
    def _doc_vector_store(
            self, 
            doc_ids: List[PARENT_NODE_ID],
            doc_summary_embed: List[List[float]],
            doc_summary: List[str]
        ):
        """Create a vector store for the document layer."""
        # Create test embeddings and metadata
        texts_and_embeddings = []
        metadata = []
        for i in range(len(doc_ids)):
            texts_and_embeddings.append((doc_summary[i], doc_summary_embed[i]))
            metadata.append({"vector_store_key": doc_ids[i]})
        # Create vector store
        self.layered_vector_stores['doc'] = FAISS.from_embeddings(
                text_embeddings=texts_and_embeddings,
                metadatas=metadata,
                embedding=self.llm.get_embedding
            )
        
    def _doc_cc_vector_store(
            self, 
            doc_summary_embed: List[List[float]],
        ) -> List[int]:
        """Create a vector store for the document cluster level."""
        embeddings_array = np.array(doc_summary_embed, dtype=np.float32)
        # Conduct dimentional reduction
        if hasattr(self.config.chunker, "dim_reduction") and self.config.chunker.dim_reduction:
            # Run dimensional reduction
            self.dim_reduction_models["doc_cc"] = run_dim_reduction(
                embeddings=embeddings_array, 
                method=self.config.chunker.dim_reduction.method, 
                n_components=getattr(self.config.chunker.dim_reduction, "n_components", None)
            )
            embeddings_array = self.dim_reduction_models["doc_cc"].transform(embeddings_array)

        # Run clustering
        self.clustering_models["doc_cc"] = run_clustering(
            embeddings_array=embeddings_array,
            method=self.config.chunker.clustering.method,
            n_clusters=getattr(self.config.chunker.clustering, "n_clusters", None),
            items_per_cluster=getattr(self.config.chunker.clustering, "items_per_cluster", None)
        )
        
        # Get labels and centroids from the model
        if hasattr(self.clustering_models["doc_cc"], 'labels_'):
            # KMeans
            labels = self.clustering_models["doc_cc"].labels_.tolist()
            centroids = {i: self.clustering_models["doc_cc"].cluster_centers_[i] for i in range(len(self.clustering_models["doc_cc"].cluster_centers_))}
        else:
            # GMM
            labels = self.clustering_models["doc_cc"].predict(embeddings_array).tolist()
            centroids = {i: self.clustering_models["doc_cc"].means_[i] for i in range(len(self.clustering_models["doc_cc"].means_))}

        # Define centroid embeddings and metadata
        centroid_embeddings = []
        metadata = []
        for label, vector in centroids.items():
            centroid_embeddings.append((f"doc_cc-{label}", vector))
            metadata.append({"vector_store_key": f"doc_cc-{label}"})
        # Create vector store
        self.layered_vector_stores["doc_cc"] = FAISS.from_embeddings(
            text_embeddings=centroid_embeddings,
            metadatas=metadata,
            embedding=self.llm.get_embedding
        )

        return labels

    def _chunk_vector_store(
            self, 
            doc_ids: List[PARENT_NODE_ID],
            chunks_dict: Dict[PARENT_NODE_ID, List[LangChainDocument]],
            labels: List[int],
        ):
        """Create a vector store for the chunk level."""
        
        # Create mapping from cluster to indices
        clusters_to_doc_ids = {} 
        for i, label in enumerate(labels):
            if label not in clusters_to_doc_ids:
                clusters_to_doc_ids[label] = []
            clusters_to_doc_ids[label].append(doc_ids[i])  # Store document ID

        # Create vector store for each cluster
        for label, doc_ids in clusters_to_doc_ids.items():
            texts = []
            metadata = []
            for doc_id in doc_ids:
                # create texts
                texts.extend([chunk.page_content for chunk in chunks_dict[doc_id]])
                metadata.extend([{**chunk.metadata, "vector_store_key": f"doc_cc-{label}", "layer": "chunk"} for chunk in chunks_dict[doc_id]])
            # Create vector store
            self.layered_vector_stores["chunk"][f"doc_cc-{label}"] = FAISS.from_texts(
                texts=texts,
                metadatas=metadata,
                embedding=self.llm.get_embedding
            )

    def _doc_summary(self, doc: SchemaDocument) -> Tuple[str, List[float]]:
        """
        Summarize document and get embedding
        """
        # Check if summary already exists in the summaries file (256 tokens)
        max_tokens = self.config.summarizer.output_tokens
        if max_tokens == 256:
            summaries_file = f"experiments/{self.dataset_config.name}_summary_256T.txt"
        else:
            summaries_file = f"experiments/{self.dataset_config.name}_summary.txt"
        
        if os.path.exists(summaries_file):
            with open(summaries_file, 'r') as f:
                for line in f:
                    doc_id, summary = line.split(' ', 1)
                    if doc_id == doc.id:
                        self.logger.debug(f"Summary for document {doc.id} found in summaries file")
                        return summary.replace('\n', ' '), self.llm.embedding.embed_documents([summary.replace('\n', ' ')])[0]

        # If not found, generate summary and store it
        summary = None
        embedding = None
        try:
            with self.profiler.track("index.doc_summary"):
                summary = run_doc_summary(doc.content, max_tokens=max_tokens)
            
            # Remove newlines from the summary
            summary = summary.replace('\n', ' ')
                
            embedding = self.llm.embedding.embed_documents([summary])
            with open(summaries_file, 'a') as f:
                f.write(f"{doc.id} {summary}\n")
        except Exception as e:
            self.logger.error(f"Error during document summarization: {str(e)}")
            # Provide a fallback if summarization fails
            if not summary:
                summary = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                embedding = self.llm.embedding.embed_documents([summary])
        finally:
            # Force garbage collection to clean up any resources that might have been used
            # by the LLM during summarization
            gc.collect()
        return summary, embedding[0]
        
    def _doc_chunking(self, doc: SchemaDocument) -> List[LangChainDocument]:
        """
        Chunk document
        """
        with self.profiler.track("index.chunking"):
            chunks = self.chunker.run([doc], mode=self.chunker_config.mode)
        
        return chunks
    
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
                            
            self.logger.info("Successfully saved all indexes")
            
        except Exception as e:
            self.logger.error(f"Error saving indexes: {str(e)}")
            raise

    def _save_ml_models(self):
        """
        Save dimensional reductions and clustering models
        """
        # clustering model
        if self.clustering_models['doc_cc']:
            model_dir = os.path.join(self.vectorstore_path, 'doc_cc')
            save_ml_models(
                model_dir, 
                'doc_cc', 
                clustering_model=self.clustering_models['doc_cc'],
                dim_reduction_model=self.dim_reduction_models['doc_cc'],
            )
            
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
