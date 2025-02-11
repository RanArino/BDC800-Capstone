# core/frameworks/base.py

from abc import ABC, abstractmethod
from datetime import datetime
from langchain_community.vectorstores import FAISS
import yaml
# from core.utils.profiler import Profiler
# from core.utils.metrics import TimingMetrics

from core.utils import get_project_root
from core.rag_core import LLMController, Chunker
from core.logger.logger import get_logger
from .schema import RAGConfig, DatasetConfig, ChunkerConfig, ModelConfig, RetrievalConfig

logger = get_logger(__name__)


class BaseRAGFramework(ABC):
    def __init__(self, config_name: str, config_path: str):
        self.logger = logger
        self.config_name = config_name
        self.config_path = config_path

        self.vector_store = None
        self.faiss_index = None
        
        self.config: RAGConfig = self._load_config()
        self.vectorstore_path: str = self._define_vectorstore_path()
        # self.profiler = Profiler().            # Performance Profiler
        # self.timing_metrics = TimingMetrics()  # Timing Metrics

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

    def run(self, query):
        self.logger.debug("Processing query: %s", query)
        retrieved_docs = self.retrieve(query)
        self.logger.debug("Retrieved %d documents", len(retrieved_docs))
        generated_text = self.generate(query, retrieved_docs)
        self.logger.info("Generated response")
        return {
            "answer": generated_text,
            "context": retrieved_docs,
            # "timing": self.profiler.get_metrics(),
        }
    
    def load_index(self, index_path: str):
        """Load the index from the given path."""
        self.logger.debug("Loading vector store from: %s", index_path)
        self.vector_store = FAISS.load_local(
           index_path, 
           self.llm.get_embedding, 
           allow_dangerous_deserialization=True
       )
        # self.vector_store = FAISS.load_local(index_path, self.llm.get_embedding)
        self.logger.info("Vector store loaded successfully")

    @abstractmethod
    def index(self, documents):
        pass

    @abstractmethod
    def retrieve(self, query):
        pass

    @abstractmethod
    def generate(self, query, retrieved_docs):
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
