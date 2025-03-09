# core/frameworks/scaler_v1.py

from typing import Dict, Union, Iterable, Generator, Tuple, List, Optional
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument

from core.datasets import Document as SchemaDocument
from core.rag_core import run_doc_summary
from core.utils import FAISSIVFCustom

from core.frameworks.base import BaseRAGFramework
from core.frameworks.schema import AVAILABLE_LAYERS, PARENT_NODE_ID

class ScalerV1RAG(BaseRAGFramework):
    """
    SCALER(Semantic Clustered Abstractive Layers for Efficient Retrieval) framework with FAISS IVF (Inverted File) Index.
    
    This framework implements two-layered vector stores.
    - Document-level vector store: each vector point represents a document summary embedding vector.
    - Chunk-level vector store: each vector point represents a chunk embedding vector.

    If a single document is assigned on index(), it will be indexed and retrieved by the FAISS brute force search or IVF (if number of chunks is greater than 10K).
    """
    def __init__(
            self, 
            config_name: str, 
            config_path: str = "core/configs/scaler_v1_rag/test.yaml", 
            is_save_vectorstore: bool = False
        ):
        super().__init__(config_name, config_path, is_save_vectorstore)
        
        # Intialize two layered FAISS IVFs
        self.layered_vector_stores: Dict[AVAILABLE_LAYERS, Union[FAISS, Dict[PARENT_NODE_ID, FAISS]]] = {
            # a single IVF for document cluster level, each vector pointrepresents a document summary embedding vector.
            "doc": None,
            # multiple IVFs for chunk cluster level per each document, each vector point represents a chunk embedding vector.
            "chunk": {}
        }

    def index(
        self,
        docs: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]], 
    ):
        """
        Index the documents into the two-layered vector stores.

        Same process as ScalerRAG, but do not take embedding before indexing.
        """
        # Update vectorstore path after setting document ID
        self.vectorstore_path = self._define_vectorstore_path(docs)

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

        # Ensure docs is a generator
        gen_docs = self._ensure_document_generator(docs)

        # Index document
        try:
            self.logger.debug("Starting document indexing")
            # Summarize document and get embedding
            num_docs = 0
            doc_sum_embed = {}
            doc_summary = {}  # doc id is key, summary is value
            for doc in gen_docs:
                # Skip document-level processing if doc is already loaded
                if loaded_layers["doc"]:
                    self.logger.info(f"Skipping processing for document {doc.id} as it's already indexed")
                else:
                    # Run document summary
                    doc_summary[doc.id], doc_sum_embed[doc.id] = self._doc_summary(doc)
                    self.logger.debug(f"Completed document {doc.id} summary")
                
                # Skip chunk processing if chunk already loaded for this document
                chunk_key = str(doc.id)
                if chunk_key in self.layered_vector_stores["chunk"] or any(k.startswith(chunk_key + "-") for k in self.layered_vector_stores["chunk"].keys()):
                    self.logger.info(f"Skipping chunk processing for document {chunk_key} as it's already indexed")
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
                    self.logger.debug(f"Completed chunking for document {chunk_key}")
                num_docs += 1

            # Create or update document summary vector store if we have multiple documents
            if num_docs > 1 and doc_summary:
                # Always recreate the doc and doc_cc layers when adding new documents
                # This ensures proper clustering with all documents
                self._layered_vector_store(
                    layer="doc",
                    embeddings=[doc_sum_embed[doc_id] for doc_id in doc_summary.keys()],
                    doc_summary=doc_summary,
                    parent_node_id=None
                )
                self.logger.debug(f"Completed document summary vector store")

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

    def _layered_vector_store(
        self, 
        layer: AVAILABLE_LAYERS, 
        embeddings: List[List[float]],
        doc_summary: Optional[Dict[PARENT_NODE_ID, str]] = None,
        chunks: Optional[List[LangChainDocument]] = None,
        parent_node_id: Optional[PARENT_NODE_ID] = None
    ):
        """Create two layered IVFs for efficient retrieval."""

        # If no clustering is configured, store all embeddings in a single vector store
        texts_and_embeddings = []
        metadatas = []

        # For document layer, match embeddings with doc_summary dictionary entries
        if layer == "doc":
            for doc_id, embedding in zip(doc_summary.keys(), embeddings):
                texts_and_embeddings.append((doc_summary[doc_id], embedding))
                metadatas.append({"layer": layer, "node_id": doc_id})

            # Add vector store to the layer
            self.layered_vector_stores[layer] = FAISSIVFCustom.from_embeddings(
                text_embeddings=texts_and_embeddings,
                metadatas=metadatas,
                embedding=self.llm.get_embedding
            )

        # For chunk layer, use the existing approach
        elif layer == "chunk":
            for idx, embedding in enumerate(embeddings):
                texts_and_embeddings.append((chunks[idx].page_content, embedding))
                metadatas.append({"layer": layer, **chunks[idx].metadata})
            
            # Add vector store to the layer
            self.layered_vector_stores[layer][parent_node_id] = FAISSIVFCustom.from_embeddings(
                text_embeddings=texts_and_embeddings,
                metadatas=metadatas,
                embedding=self.llm.get_embedding
            )

        else:
            raise ValueError(f"Invalid layer: {layer}")

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
                
                # Detect layer type dynamically based on the data structure
                if isinstance(vector_store, dict):
                    # Handle dictionary of FAISS instances (typically chunk layer)
                    for node_id, vs in vector_store.items():
                        if vs is not None:
                            save_path = os.path.join(layer_path, f"{node_id}.faiss")
                            vs.save_local(save_path)
                            self.logger.debug(f"Saved {layer} index for node {node_id} to {save_path}")
                else:
                    # Handle single FAISS instance (typically doc layer)
                    save_path = os.path.join(layer_path, f"{layer}.faiss")
                    vector_store.save_local(save_path)
                    self.logger.debug(f"Saved {layer} index to {save_path}")
            
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
            
            # Track which layers were loaded
            loaded_layers = {layer: False for layer in self.layered_vector_stores.keys()}
            
            # Load each layer based on its expected structure (from self.layered_vector_stores)
            for layer, store_structure in self.layered_vector_stores.items():
                layer_path = os.path.join(self.vectorstore_path, layer)
                
                if not os.path.exists(layer_path):
                    self.logger.debug(f"No index directory found for layer {layer}")
                    continue
                
                # Determine if this layer should be a dictionary or single instance based on its initial structure
                if isinstance(store_structure, dict):
                    # Get all .faiss files in the directory
                    faiss_files = [f for f in os.listdir(layer_path) if f.endswith('.faiss')]
                    if faiss_files:  # Only mark as loaded if we found some files
                        loaded_layers[layer] = True
                        for faiss_file in faiss_files:
                            node_id = faiss_file[:-6]  # Remove .faiss extension
                            index_path = os.path.join(layer_path, faiss_file)
                            self.layered_vector_stores[layer][node_id] = FAISSIVFCustom.load_local(
                                index_path,
                                self.llm.get_embedding,
                                allow_dangerous_deserialization=True
                            )
                            self.logger.debug(f"Loaded {layer} index for node {node_id} from {index_path}")
                else:
                    # This is a single instance layer (like "doc")
                    index_path = os.path.join(layer_path, f"{layer}.faiss")
                    if os.path.exists(index_path):
                        self.layered_vector_stores[layer] = FAISSIVFCustom.load_local(
                            index_path,
                            self.llm.get_embedding,
                            allow_dangerous_deserialization=True
                        )
                        self.logger.debug(f"Loaded {layer} index from {index_path}")
                        loaded_layers[layer] = True
            
            loaded_count = sum(loaded_layers.values())
            if loaded_count == 0:
                self.logger.info("No indexes were loaded")
            else:
                self.logger.info(f"Successfully loaded {loaded_count} layers: {[layer for layer, loaded in loaded_layers.items() if loaded]}")
            
            return loaded_layers
            
        except Exception as e:
            self.logger.error(f"Error loading indexes: {str(e)}")
            raise

    def retrieve(
        self, 
        query: str, 
        top_k_per_layer: Optional[Dict[AVAILABLE_LAYERS, int]] = None, 
    ) -> Dict[AVAILABLE_LAYERS, List[LangChainDocument]]:
        """
        Retrieve relevant documents using SCALER's hierarchical approach.
        
        This method:
        1. Retrieves top_k documents from the top node FAISS vector store.
        2. For each retrieved document, retrieves top_k chunks from the corresponding chunk node FAISS vector store.
        """
        try:
            self.logger.debug(f"Starting retrieval for query: {query}")
            
            # Use config's top_k if not specified
            if top_k_per_layer is None:
                config = self.retrieval_generation_config
                top_k_per_layer = {
                    "doc": getattr(config, "top_k_doc", None) or 5,
                    "chunk": getattr(config, "top_k", None) or 10
                }
            # Get the query embedding
            query_embedding = self.llm.embedding.embed_query(query)
            results = {}
            previous_layer = None
            for layer, top_k in top_k_per_layer.items():
                # Skip if the vectorstore is not created
                if not self.layered_vector_stores.get(layer, None):
                    continue

                vectorstore = self.layered_vector_stores[layer]
                self.logger.debug(f"Processing {layer} layer")

                # if vectorsore is a single FAISS instance, use similarity_search_with_score
                if isinstance(vectorstore, FAISS):
                    results[layer] = vectorstore.similarity_search_with_score_by_vector(
                        query_embedding,
                        k=top_k
                    )
                
                # if vectorsore is a dictionary of FAISS instances, use similarity_search_with_score_by_vector
                elif isinstance(vectorstore, dict):
                    # Initialize the results for the current layer
                    layer_results = []
                    # get the parent node ides from the previous layer
                    parent_node_ids = [doc.metadata.get("node_id") for doc, _ in results[previous_layer]]
                    for node_id in parent_node_ids:
                        layer_results.extend(vectorstore[node_id].similarity_search_with_score_by_vector(
                            query_embedding,
                            k=top_k
                        ))
                    # Take top_k - no need to sort since similarity_search_by_vector already returns sorted results
                    results[layer] = [doc for doc, _ in sorted(layer_results, key=lambda x: x[1])][:top_k]
                
                # update the previous layer name
                previous_layer = layer

            return results["chunk"]

        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise
