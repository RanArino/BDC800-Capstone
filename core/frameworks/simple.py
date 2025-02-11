# core/frameworks/simple.py

from typing import List, Dict, Any, Optional
import os
import gc

from langchain_community.vectorstores import FAISS

from .base import BaseRAGFramework
from core.datasets import Document as SchemaDocument

class SimpleRAG(BaseRAGFramework):
    def __init__(self, config_name: str):
        super().__init__(config_name, config_path = "core/configs/simple_rag.yaml")
    
    def index(self, documents: List[SchemaDocument]):
        """Index the documents using FAISS index"""
        try:
            self.logger.debug("Starting document indexing")
            
            # Load index if exists
            if os.path.exists(self.vectorstore_path):
                self.load_index(self.vectorstore_path)
                return
            
            # Split documents into chunks using specified mode
            self.logger.debug("Splitting documents into chunks")
            chunks = self.chunker.run(documents, mode=self.chunker_config.mode)
            self.logger.info(f"Created {len(chunks)} chunks")

            # Process chunks in smaller batches
            BATCH_SIZE = 5
            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, len(chunks))
                batch_chunks = chunks[start_idx:end_idx]
                
                self.logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}")
                
                if batch_idx == 0:
                    # Initialize vector store with first batch
                    self.vector_store = FAISS.from_texts(
                        texts=[chunk.page_content for chunk in batch_chunks],
                        embedding=self.llm.get_embedding,
                        metadatas=[chunk.metadata for chunk in batch_chunks]
                    )
                else:
                    # Add subsequent batches
                    self.vector_store.add_texts(
                        texts=[chunk.page_content for chunk in batch_chunks],
                        metadatas=[chunk.metadata for chunk in batch_chunks]
                    )
                
                # Force garbage collection after each batch
                gc.collect()
            
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
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using FAISS search."""
        try:
            self.logger.debug(f"Starting retrieval for query: {query}")
            
            if not self.vector_store:
                raise ValueError("Index not created. Please run the 'index' method first.")
            
            # Use config's top_k if not specified
            if top_k is None:
                top_k = self.retrieval_config.top_k
            
            self.logger.debug(f"Retrieving top {top_k} documents")
            results = self.vector_store.similarity_search(query, k=top_k)
            self.logger.info(f"Retrieved {len(results)} documents")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the RAG pipeline on a query."""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(query)
            
            # Generate answer
            answer = self.generate(query, retrieved_docs)
            
            return {
                "query": query,
                "answer": answer,
                "context": retrieved_docs
            }
            
        except Exception as e:
            self.logger.error(f"Error during RAG execution: {str(e)}")
            raise

    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM."""
        try:
            self.logger.debug("Starting answer generation")
            
            # Extract content from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            self.logger.debug(f"Created context from {len(retrieved_docs)} documents")
            
            # Generate prompt
            prompt = f"""Based on the following context, please answer the question.
            
Context:
{context}

Question: {query}

Answer:"""
            
            # Generate answer using LLM
            self.logger.debug("Generating answer using LLM")
            answer = self.llm.generate_text(prompt)
            self.logger.info("Answer generated successfully")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error during answer generation: {str(e)}")
            raise
    
    def evaluate(self, dataset):
        # To be implemented later
        pass
    
    def cleanup(self):
        """Explicitly clean up resources."""
        try:
            self.logger.debug("Starting cleanup")
            if self.vector_store:
                self.logger.debug("Cleaning up vector store")
                del self.vector_store
                self.vector_store = None
            if self.faiss_index:
                self.logger.debug("Cleaning up FAISS index")
                del self.faiss_index
                self.faiss_index = None
            gc.collect()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise