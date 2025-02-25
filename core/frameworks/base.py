# core/frameworks/base.py

from abc import ABC, abstractmethod
from typing import List, Optional, Generator, Tuple, Union, Iterable, Dict, Any
from datetime import datetime
from collections import deque
import gc
import os
from itertools import tee

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
import yaml
from core.utils import Profiler
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
from core.evaluation.metrics_summary import calculate_metrics_for_qa
from core.evaluation.schema import MetricsSummary

from .schema import RAGConfig, DatasetConfig, ChunkerConfig, ModelConfig, RetrievalConfig, RAGResponse

logger = get_logger(__name__)


class BaseRAGFramework(ABC):
    def __init__(self, config_name: str, config_path: str, is_save_vectorstore: bool = False):
        self.logger = logger
        self.config_name = config_name
        self.config_path = config_path
        self.is_save_vectorstore = is_save_vectorstore

        self.config: RAGConfig = self._load_config()
        self.vectorstore_path: str = self._define_vectorstore_path()
        self.profiler = Profiler(reset_on_init=True)

        # Initialize variables that are defined in index() method
        self.dataset: BaseDataset = None
        self.vector_store: FAISS = None

        # Initialize LLMController
        self.logger.info("Initializing LLMController with models: %s (LLM), %s (Embedding)", 
                        self.model_config.llm_id, self.model_config.embedding_id)
        self.llm = LLMController(
            llm_id=self.model_config.llm_id, 
            embedding_id=self.model_config.embedding_id
        )

        # Initialize Chunker based on config
        self.logger.info("Initializing Chunker with size: %d, overlap: %.2f%%", 
                        self.chunker_config.size, self.chunker_config.overlap * 100)
        self.chunker = Chunker(
            chunk_size=self.chunker_config.size, 
            chunk_overlap=self.chunker_config.overlap
        )
        self.logger.info("BaseRAGFramework initialization completed")

    def load_dataset(
            self, 
            number_of_docs: Optional[int] = None, 
            number_of_qas: Optional[int] = None, 
            selection_mode: Optional[str] = None
        ) -> Tuple[Generator[SchemaDocument, None, None], Union[Generator[List[IntraDocumentQA], None, None], Generator[InterDocumentQA, None, None]]]:
        """Load the dataset from the given path.
        
        Args:
            number_of_docs: Optional number of documents to load. Required for IntraDocumentQA datasets.
            number_of_qas: Optional number of QAs to load. Required for InterDocumentQA datasets.
            selection_mode: Optional selection mode ('sequential' or 'random'). Defaults to config value or 'sequential'.
            
        Returns:
            A tuple of (document generator, QA generator). For IntraDocumentQA, the QA generator yields lists of QAs per document.
            For InterDocumentQA, it yields individual QAs.
            
        Raises:
            ValueError: If number_of_docs is not specified for IntraDocumentQA datasets or
                      if number_of_qas is not specified for InterDocumentQA datasets.
        """
        self.logger.debug("Loading dataset with params: docs=%s, qas=%s, mode=%s", number_of_docs, number_of_qas, selection_mode)
        
        # Load dataset
        self.dataset = get_dataset(self.dataset_config.name)
        self.dataset.load()

        # Check config
        if not number_of_docs:
            number_of_docs = self.dataset_config.number_of_docs
        if not number_of_qas:
            number_of_qas = self.dataset_config.number_of_qas
        
        # Call appropriate loader based on QA type
        if self.dataset.qa_type == IntraDocumentQA:
            if number_of_qas is not None and number_of_docs is None:
                raise ValueError("For IntraDocumentQA datasets, please specify number_of_docs instead of number_of_qas")
            return self._load_intra_docs_and_qas(number_of_docs, selection_mode)
        else:
            if number_of_docs is not None and number_of_qas is None:
                raise ValueError("For InterDocumentQA datasets, please specify number_of_qas instead of number_of_docs")
            return self._load_inter_docs_and_qas(number_of_qas, selection_mode)

    def run(
        self, 
        qas: Union[List[IntraDocumentQA|InterDocumentQA], Generator[IntraDocumentQA|InterDocumentQA, None, None], Iterable[IntraDocumentQA|InterDocumentQA]]
    ) -> Tuple[List[RAGResponse], List[MetricsSummary]]:
        """Run the RAG pipeline on queries.
        
        Args:
            qas: A list, generator, or any iterable of QA pairs to process
            
        Returns:
            A tuple containing:
                - List of RAGResponses corresponding to each input QA pair
                - List of individual metrics dictionaries for each QA pair
            
        Profiling:
            - retrieval: time for overall retrieval process
            - generation: time for overall generation process
        """
        responses = []
        metrics_list = []

        for qa in qas:
            try:
                # Retrieve relevant documents
                with self.profiler.track("retrieval"):
                    retrieved_docs = self.retrieve(qa.q)
                
                # Generate answer
                with self.profiler.track("generation"):
                    llm_answer = self.generate(qa.q, retrieved_docs)

                # Calculate metrics
                metrics = calculate_metrics_for_qa(
                    qa=qa,
                    response=llm_answer
                )
                metrics_list.append(metrics)
                responses.append(llm_answer)
                
            except Exception as e:
                self.logger.error(f"Error during RAG execution for question '{qa.q}': {str(e)}")
                raise
        
        return responses, metrics_list
  
    def index(
        self, 
        docs: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]], 
        is_update: bool = False
    ):
        """Index the documents using FAISS index
        
        Args:
            docs: A single document, a generator of documents, or an iterable of documents to index
            is_update: Whether to update the existing index

        Profiling:
            - index.preprocessing: time for chunking and its depending operations
            - index.vectorstore: time for embedding and creating vector store
        """
        gen_docs = self._ensure_document_generator(docs)
        
        # Index documents
        try:
            self.logger.debug("Starting document indexing")
            
            # Load index if exists
            if os.path.exists(self.vectorstore_path) and not is_update:
                self._load_index(self.vectorstore_path)
                return
            
            # Execute chunking
            self.logger.debug("Splitting documents into chunks")

            # Preprocessing
            with self.profiler.track("index.preprocessing"):
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

    def _process_chunk_batch(self, batch_chunks: List[LangChainDocument]):
        """Process a batch of chunks and add them to the vector store.
        
        Args:
            batch_chunks: List of chunks to process in this batch
        """
        self.logger.debug(f"Processing batch of {len(batch_chunks)} chunks")
        
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            # Initialize vector store with first batch
            with self.profiler.track("index.vectorstore"):
                self.vector_store = FAISS.from_texts(
                    texts=[chunk.page_content for chunk in batch_chunks],
                    embedding=self.llm.get_embedding,
                    metadatas=[chunk.metadata for chunk in batch_chunks]
                )
        else:
            # Add subsequent batches
            with self.profiler.track("index.vectorstore"):
                self.vector_store.add_texts(
                    texts=[chunk.page_content for chunk in batch_chunks],
                    metadatas=[chunk.metadata for chunk in batch_chunks]
                )

    def _ensure_document_generator(
        self, 
        documents: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]]
    ) -> Generator[SchemaDocument, None, None]:
        """Convert a single document or generator into a generator.
        
        Args:
            documents: A single document, a generator of documents, or an iterable of documents
            
        Returns:
            A generator of documents
        """
        if isinstance(documents, SchemaDocument):
            yield documents
        elif isinstance(documents, Generator):
            yield from documents
        else:
            yield from documents

    @abstractmethod
    def index_preprocessing(
        self, 
        documents: Union[SchemaDocument, Generator[SchemaDocument, None, None], Iterable[SchemaDocument]]
    ) -> Generator[LangChainDocument, None, None]:
        """Preprocess the documents before indexing.
        
        Args:
            documents: A single document, a generator of documents, or an iterable of documents to preprocess.
            
        Returns:
            A generator of preprocessed documents.
        """
        pass

    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[LangChainDocument]:
        """Retrieve relevant documents from the vector store.
        
        Args:
            query: The query to retrieve documents for.
            top_k: The number of documents to retrieve.
            
        Returns:
            A list of documents that are relevant to the query.
        """
        pass

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
        # Format: config_name-dataset-indextype-YYYYMMDD (i.e., simple_rag_01-qasper-fixed100T(20%)-flatl2)
        # date_str = datetime.now().strftime("%Y%m%d")
        dataset_name = self.dataset_config.name
        chunk_mode = self.chunker_config.mode
        chunk_size = self.chunker_config.size
        chunk_overlap = self.chunker_config.overlap * 100
        faiss_search = self.retrieval_config.faiss_search
        # Get dataset name
        filename = f"{self.config_name}-{dataset_name}-{chunk_mode}{chunk_size}T({chunk_overlap}%)-{faiss_search}"
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

    def _load_intra_docs_and_qas(
        self,
        number_of_docs: Optional[int] = None,
        selection_mode: Optional[str] = None
    ) -> Tuple[Generator[SchemaDocument, None, None], Generator[List[IntraDocumentQA], None, None]]:
        """Load document and QA generators for IntraDocumentQA type datasets.
        Groups QAs by document and yields them as lists."""
        selection_mode = selection_mode or self.dataset_config.selection_mode or 'sequential'
        self.logger.debug(f"Loading IntraDocumentQA generators with selection mode: {selection_mode}")
        
        # Get docs and clone generator for IDs
        gen_docs, docs_for_ids = tee(
            self.dataset.get_documents(
                num_docs=number_of_docs, 
                selection_mode=selection_mode
            )
        )
        
        # Create a generator that yields lists of QAs grouped by document
        def group_qas_by_doc():
            for doc in docs_for_ids:
                qas_for_doc = list(self.dataset.get_queries(doc_ids=[doc.id]))
                # Always yield the list of QAs, even if empty
                yield qas_for_doc
        
        return gen_docs, group_qas_by_doc()

    def _load_inter_docs_and_qas(
        self,
        number_of_qas: Optional[int] = None,
        selection_mode: Optional[str] = None
    ) -> Tuple[Generator[SchemaDocument, None, None], Generator[InterDocumentQA, None, None]]:
        """Load document and QA generators for InterDocumentQA type datasets."""
        selection_mode = selection_mode or self.dataset_config.selection_mode or 'sequential'
        self.logger.debug(f"Loading InterDocumentQA generators with selection mode: {selection_mode}")
        
        # Get QAs and clone generator for IDs
        gen_qas, qas_for_ids = tee(
            self.dataset.get_queries(
                num_qas=number_of_qas,
                selection_mode=selection_mode
            )
        )
        
        # Get documents referenced by these QAs
        return (
            self.dataset.get_documents(doc_ids=(
                doc_id for qa in qas_for_ids 
                for doc_id in qa.document_ids
            )),
            gen_qas
        )

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
