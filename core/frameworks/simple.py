# core/frameworks/simple.py

from typing import List, Optional, Union, Generator, Iterable

from langchain_core.documents import Document as LangChainDocument

from .base import BaseRAGFramework
from core.datasets import Document as SchemaDocument

class SimpleRAG(BaseRAGFramework):
    def __init__(self, config_name: str, config_path: str = "core/configs/simple_rag/test.yaml", is_save_vectorstore: bool = False):
        super().__init__(config_name, config_path, is_save_vectorstore)
    
    def index_preprocessing(
            self, 
            documents: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]]
        ) -> Generator[LangChainDocument, None, None]:
        """Index the documents using FAISS index
        
        Args:
            documents: A single document, a generator of documents, or an iterable of documents
            
        Returns:
            A generator of LangChainDocument chunks
        """
        try:
            # Convert input to generator if needed
            doc_generator = self._ensure_document_generator(documents)
            
            # Process documents one at a time to maintain memory efficiency
            for doc in doc_generator:
                # Execute simple chunking for each document
                chunks = self.chunker.run([doc], mode=self.chunker_config.mode)
                # Yield each chunk
                for chunk in chunks:
                    yield chunk
        
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
