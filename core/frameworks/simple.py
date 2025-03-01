# core/frameworks/simple.py

from typing import List, Optional, Union, Generator, Iterable
import gc
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument

from .base import BaseRAGFramework
from core.datasets import Document as SchemaDocument

class SimpleRAG(BaseRAGFramework):
    def __init__(self, config_name: str, config_path: str = "core/configs/simple_rag/test.yaml", is_save_vectorstore: bool = False):
        super().__init__(config_name, config_path, is_save_vectorstore)

    def index(
        self, 
        docs: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]], 
    ):
        """Index the documents using FAISS index
        
        Args:
            docs: A single document, a generator of documents, or an iterable of documents to index

        Profiling:
            - index.chunking: Time taken for chunking documents and related operations
            - index.vectorstore: Time taken for embedding documents and creating the vector store
        """
        gen_docs = self._ensure_document_generator(docs)
        
        # Index documents
        try:
            self.logger.debug("Starting document indexing")
            
            # Load index if exists
            if os.path.exists(self.vectorstore_path):
                self._load_index(self.vectorstore_path)
                return
            
            # Execute chunking
            self.logger.debug("Splitting documents into chunks")
           
            # Process documents one at a time to maintain memory efficiency
            for doc in gen_docs:
                # Execute simple chunking for each document
                with self.profiler.track("index.chunking"):
                    chunks = self.chunker.run([doc], mode=self.chunker_config.mode)
                
                if not hasattr(self, 'vector_store') or self.vector_store is None:
                    # Initialize vector store with first batch
                    with self.profiler.track("index.vectorstore"):
                        self.vector_store = FAISS.from_texts(
                            texts=[chunk.page_content for chunk in chunks],
                            embedding=self.llm.get_embedding,
                            metadatas=[chunk.metadata for chunk in chunks]
                        )
                else:
                    # Add subsequent batches
                    with self.profiler.track("index.vectorstore"):
                        self.vector_store.add_texts(
                            texts=[chunk.page_content for chunk in chunks],
                            metadatas=[chunk.metadata for chunk in chunks]
                        )

                gc.collect()
            
            if self.is_save_vectorstore:
                # Ensure vectorstore directory exists
                self.logger.debug(f"Creating directory: {os.path.dirname(self.vectorstore_path)}")
                os.makedirs(os.path.dirname(self.vectorstore_path), exist_ok=True)
                
                # Save vector store
                self.logger.debug(f"Saving vector store to {self.vectorstore_path}")
                self.vector_store.save_local(self.vectorstore_path)
                self.logger.info("Indexing completed successfully")

        except Exception as e:
            self.logger.error(f"Error during indexing: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[LangChainDocument]:
        """Retrieve relevant documents using FAISS search."""
        try:
            self.logger.debug(f"Starting retrieval for query: {query}")
            
            if not self.vector_store:
                raise ValueError("Index not created. Please run the 'index' method first.")
            
            # Use config's top_k if not specified
            if top_k is None:
                top_k = self.retrieval_generation_config.top_k
            
            self.logger.debug(f"Retrieving top {top_k} documents")
            results = self.vector_store.similarity_search(query, k=top_k)
            self.logger.info(f"Retrieved {len(results)} documents")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise
