# core/frameworks/simple.py

from typing import List, Optional
import os
import gc

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument

from .base import BaseRAGFramework
from .schema import RAGResponse
from core.datasets import Document as SchemaDocument

class SimpleRAG(BaseRAGFramework):
    def __init__(self, config_name: str):
        super().__init__(config_name, config_path = "core/configs/simple_rag.yaml")
    
    def index_preprocessing(self, documents: List[SchemaDocument]) -> List[LangChainDocument]:
        """Index the documents using FAISS index"""
        try:
            # Execute simple chunking
            chunks = self.chunker.run(documents, mode=self.chunker_config.mode)
            return chunks
        
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
                top_k = self.retrieval_config.top_k
            
            self.logger.debug(f"Retrieving top {top_k} documents")
            results = self.vector_store.similarity_search(query, k=top_k)
            self.logger.info(f"Retrieved {len(results)} documents")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise

    def generate(self, query: str, retrieved_docs: List[LangChainDocument]) -> RAGResponse:
        """Generate answer using LLM with retrieved langchain documents as context."""
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
            llm_answer = self.llm.generate_text(prompt)
            self.logger.info("Answer generated successfully")
            
            return RAGResponse(
                query=query,
                llm_answer=llm_answer,
                context=retrieved_docs
            )
            
        except Exception as e:
            self.logger.error(f"Error during answer generation: {str(e)}")
            raise
    
    def evaluate(self, dataset):
        # To be implemented later
        pass
