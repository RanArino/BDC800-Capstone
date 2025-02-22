# core/frameworks/base.py

from abc import ABC, abstractmethod
from typing import List, Optional, Generator, Tuple
from datetime import datetime
from collections import deque
import gc
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
import yaml
# from core.utils.profiler import Profiler
# from core.utils.metrics import TimingMetrics

from core.utils import get_project_root
from core.rag_core import LLMController, Chunker
from core.logger.logger import get_logger
from core.datasets import (
    get_dataset,
    BaseDataset,
    IntraDocumentQA, 
    InterDocumentQA, 
    Document as SchemaDocument
)

from .schema import RAGConfig, DatasetConfig, ChunkerConfig, ModelConfig, RetrievalConfig, RAGResponse

logger = get_logger(__name__)


class BaseRAGFramework(ABC):
    def __init__(self, config_name: str, config_path: str):
        self.logger = logger
        self.config_name = config_name
        self.config_path = config_path

        self.config: RAGConfig = self._load_config()
        self.vectorstore_path: str = self._define_vectorstore_path()
        # self.profiler = Profiler().            # Performance Profiler
        # self.timing_metrics = TimingMetrics()  # Timing Metrics

        # Initialize variables that are defined in index() method
        self.dataset: BaseDataset = None
        self.docs: Generator[SchemaDocument, None, None] = None   
        self.qas: Generator[IntraDocumentQA | InterDocumentQA, None, None] = None
        self.vector_store: FAISS = None

        # Initialize LLMController
        self.logger.info("Initializing LLMController with models: %s (LLM), %s (Embedding)", 
                        self.model_config.llm_id, self.model_config.embedding_id)
        self.llm = LLMController(
            llm_id=self.model_config.llm_id, 
            embedding_id=self.model_config.embedding_id
        )

        # Initialize Chunker based on config
        self.logger.info("Initializing Chunker with size: %d, overlap: %d", 
                        self.chunker_config.size, self.chunker_config.overlap)
        self.chunker = Chunker(
            chunk_size=self.chunker_config.size, 
            chunk_overlap=self.chunker_config.overlap
        )
        self.logger.info("BaseRAGFramework initialization completed")

    def load_dataset(
            self, 
            number_of_docs: Optional[int] = None, 
            number_of_qas: Optional[int] = None, 
            select_mode: Optional[str] = None
        ) -> Tuple[Generator[SchemaDocument, None, None], Generator[IntraDocumentQA | InterDocumentQA, None, None]]:
        """Load the dataset from the given path.
        
        Args:
            number_of_docs: Optional number of documents to load. If specified, related QAs will be loaded through doc IDs.
            number_of_qas: Optional number of QAs to load. If specified (and number_of_docs not specified), related docs will be loaded through QA doc IDs.
            select_mode: Optional selection mode ('sequential' or 'random'). Defaults to config value or 'sequential'.
            
        Returns:
            A tuple of (document generator, QA generator)
        """
        self.logger.debug("Loading dataset with params: docs=%s, qas=%s, mode=%s", number_of_docs, number_of_qas, select_mode)
        
        # Initialize dataset
        self.dataset = get_dataset(self.dataset_config.name)
        self.dataset.load()  # Ensure dataset is loaded

        # TODO: get documents and qas generators
        # return doc_generator, qa_generator

    def run(self, qa: IntraDocumentQA|InterDocumentQA) -> RAGResponse:
        """Run the RAG pipeline on a query."""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(qa.q)
            
            # Generate answer
            llm_answer = self.generate(qa.q, retrieved_docs)
            
            return llm_answer
            
            
        except Exception as e:
            self.logger.error(f"Error during RAG execution: {str(e)}")
            raise

    def index(self, gen_docs: Generator[SchemaDocument, None, None], is_update: bool = False):
        """Index the documents using FAISS index
        
        Args:
            gen_docs: Generator of documents to index
            is_update: Whether to update the existing index
        """
        # Index documents
        try:
            self.logger.debug("Starting document indexing")
            
            # Load index if exists
            if os.path.exists(self.vectorstore_path) and not is_update:
                self._load_index(self.vectorstore_path)
                return
            
            # Execute chunking
            self.logger.debug("Splitting documents into chunks")
            gen_chunks = self.index_preprocessing(gen_docs)
            
            # Process chunks in batches while maintaining generator pattern
            BATCH_SIZE = 5
            current_batch = deque(maxlen=BATCH_SIZE)
            
            for chunk in gen_chunks:
                current_batch.append(chunk)
                
                if len(current_batch) >= BATCH_SIZE:
                    self._process_chunk_batch(list(current_batch))
                    current_batch.clear()  # Clear the deque
                    gc.collect()  # Force garbage collection after each batch
            
            # Process any remaining chunks
            if current_batch:
                self._process_chunk_batch(list(current_batch))
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

    def _process_chunk_batch(self, batch_chunks: List[LangChainDocument]):
        """Process a batch of chunks and add them to the vector store.
        
        Args:
            batch_chunks: List of chunks to process in this batch
        """
        self.logger.debug(f"Processing batch of {len(batch_chunks)} chunks")
        
        if not hasattr(self, 'vector_store') or self.vector_store is None:
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

    @abstractmethod
    def index_preprocessing(
        self, 
        documents: Generator[SchemaDocument, None, None]
    ) -> Generator[LangChainDocument, None, None]:
        """Preprocess the documents before indexing.
        
        Args:
            documents: A generator of documents to preprocess.
            
        Returns:
            A generator of preprocessed documents.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None):
        pass

    @abstractmethod
    def generate(self, query: str, retrieved_docs: List[LangChainDocument]) -> RAGResponse:
        pass

    @abstractmethod
    def evaluate(self, dataset):
        pass
    
    def _load_config(self) -> RAGConfig:
        """Load the config for the given config name."""
        self.logger.debug("Loading configuration from: %s", self.config_path)
        if self.config_path is None:
            self.logger.error("Config path is not set")
            raise ValueError("Config path is not set")
        
        with open(self.config_path, "r") as f:
            config: dict = yaml.safe_load(f)
        
        if config.get(self.config_name) is None:
            self.logger.error("Config name %s not found in %s", self.config_name, self.config_path)
            raise ValueError(f"Config name {self.config_name} not found in {self.config_path}")
        
        self.logger.info("Configuration loaded successfully")
        return RAGConfig(**config[self.config_name])

    def _define_vectorstore_path(self) -> str:
        """Generate a structured path for vector store."""
        self.logger.debug("Generating vectorstore path")
        # base path
        base_path = get_project_root() / "core/vectorstore"
        # Format: config_name-dataset-indextype-YYYYMMDD (i.e., simple_rag_01-qasper-flatl2-20250211)
        # date_str = datetime.now().strftime("%Y%m%d")
        dataset_name = self.dataset_config.name
        faiss_search = self.retrieval_config.faiss_search
        # Get dataset name
        filename = f"{self.config_name}-{dataset_name}-{faiss_search}"
        full_path = f"{base_path}/{filename}"
        self.logger.info("Generated vectorstore path: %s", full_path)
        return full_path

    def _load_index(self, index_path: str):
        """Load the index from the given path. Called by index() method."""
        self.logger.debug("Loading vector store from: %s", index_path)
        self.vector_store = FAISS.load_local(
           index_path, 
           self.llm.get_embedding, 
           allow_dangerous_deserialization=True
       )
        # self.vector_store = FAISS.load_local(index_path, self.llm.get_embedding)
        self.logger.info("Vector store loaded successfully")


    @property
    def dataset_config(self) -> DatasetConfig:
        return self.config.dataset
    
    @property
    def chunker_config(self) -> ChunkerConfig:
        return self.config.chunker
    
    @property
    def model_config(self) -> ModelConfig:
        return self.config.model
    
    @property
    def retrieval_config(self) -> RetrievalConfig:
        return self.config.retrieval
