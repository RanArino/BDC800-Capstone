# core/datasets/base.py

"""
Base dataset class and interfaces for RAG framework datasets.
"""
from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, List, Union, Literal
from pathlib import Path
import json
import ijson
import random

from core.datasets.schema import Document, IntraDocumentQA, InterDocumentQA
from core.logger.logger import get_logger

# Let Python automatically determine the module name
logger = get_logger(__name__)

class BaseDataset(ABC):
    """Base class for all datasets used in the RAG framework."""
    
    def __init__(self, name: str, qa_type: Union[type[IntraDocumentQA], type[InterDocumentQA]]):
        """Initialize dataset.
        
        Args:
            name: Name of the dataset
            qa_type: Type of QA pairs in this dataset (IntraDocumentQA or InterDocumentQA)
        """
        self.name = name
        self.qa_type = qa_type
        self.data_dir = Path(__file__).parent / "data" / name
        self.docs_file = self.data_dir / "documents.json"
        self.qas_file = self.data_dir / "qas.json"
        self._test_mode = False
        logger.debug(f"Initialized {name} dataset with data directory: {self.data_dir}")
    
    def load(self, mode: Optional[str] = None) -> None:
        """Process raw data and save to JSON if not exists.
        
        Args:
            mode: Optional mode for loading data. If 'test', only processes first 5 items.
        """
        logger.info(f"Loading {self.name} dataset")
        self._test_mode = mode == "test"
        test_suffix = "_test" if self._test_mode else ""
        self.docs_file = self.data_dir / f"documents{test_suffix}.json"
        self.qas_file = self.data_dir / f"qas{test_suffix}.json"
        
        if not (self.docs_file.exists() and self.qas_file.exists()):
            logger.info(f"Data files not found, processing raw data...")
            documents, qas = self._process_raw_data()
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            with open(self.docs_file, 'w') as f:
                json.dump([doc.model_dump() for doc in documents], f, indent=2)
            logger.info(f"Processed and saved documents to {self.docs_file}")
            
            # Save QAs - now using a simpler format since we know the type
            with open(self.qas_file, 'w') as f:
                json.dump([qa.model_dump() for qa in qas], f, indent=2)
            logger.info(f"Processed and saved QAs to {self.qas_file}")
        else:
            logger.info(f"Using existing data files")

    def _count_items(self, key: str) -> int:
        """Count number of items in a JSON array without loading entire file."""
        count = 0
        file_path = self.qas_file if key in ['intra_qas', 'inter_qas'] else self.docs_file
        with open(file_path, 'rb') as f:
            items = ijson.items(f, f"{key}.item" if key in ['intra_qas', 'inter_qas'] else "item")
            for _ in items:
                count += 1
        return count

    def get_documents(
            self,
            num_docs: Optional[int] = None,
            selection_mode: Optional[Literal["sequential", "random"]] = "sequential",
            doc_ids: Optional[List[str]] = None
            ) -> Generator[Document, None, None]:
        """Get documents one by one from the dataset.
        
        Args:
            num_docs: Optional number of documents to get. If None, yields all documents.
                     Ignored if doc_ids is provided.
            selection_mode: Optional selection mode. Can be 'sequential' or 'random'.
                          Ignored if doc_ids is provided.
            doc_ids: Optional list of document IDs to retrieve. If provided, other parameters are ignored.
                    If empty list is provided, returns without yielding any documents.
        """
        logger.debug(f"Starting document iteration for {self.name} dataset")
        if not self.docs_file.exists():
            logger.info("Documents file not found, triggering load()")
            self.load()

        # If specific IDs are requested
        if doc_ids is not None:
            # Return immediately if empty list provided
            if not doc_ids:
                logger.warning("Empty document ID list provided, returning without documents")
                return
                
            doc_ids_set = set(doc_ids)  # Convert to set for O(1) lookup
            with open(self.docs_file, 'rb') as f:
                for doc in ijson.items(f, "item"):
                    if doc['id'] in doc_ids_set:
                        yield Document.model_validate(doc)
            return

        # For random selection, we need to know total count first
        if selection_mode == "random" and num_docs is not None:
            total_docs = self._count_items("")  # Empty key for documents
            if total_docs == 0:
                return
                
            # Generate random indices
            indices = set(random.sample(range(total_docs), min(num_docs, total_docs)))
            
            # Stream and yield only selected indices
            with open(self.docs_file, 'rb') as f:
                for idx, doc in enumerate(ijson.items(f, "item")):
                    if idx in indices:
                        yield Document.model_validate(doc)
        else:
            # For sequential access, stream directly
            with open(self.docs_file, 'rb') as f:
                for idx, doc in enumerate(ijson.items(f, "item")):
                    if num_docs is not None and idx >= num_docs:
                        break
                    yield Document.model_validate(doc)

    def get_queries(
            self,
            num_qas: Optional[int] = None,
            selection_mode: Optional[Literal["sequential", "random"]] = "sequential",
            qa_ids: Optional[List[str]] = None,
            doc_ids: Optional[List[str]] = None
            ) -> Generator[IntraDocumentQA | InterDocumentQA, None, None]:
        """Get queries (QA pairs) one by one from the dataset.
        
        Args:
            num_qas: Optional number of QAs to get. If None, yields all QAs.
                    Ignored if qa_ids or doc_ids is provided.
            selection_mode: Optional selection mode. Can be 'sequential' or 'random'.
                          Ignored if qa_ids or doc_ids is provided.
            qa_ids: Optional list of QA IDs to retrieve. If provided, other parameters except doc_ids are ignored.
                   If empty list is provided, returns without yielding any QAs.
            doc_ids: Optional list of document IDs to retrieve QAs for. For IntraDocumentQA, yields QAs where
                    document_id matches any of the provided IDs. For InterDocumentQA, yields QAs where any
                    of the document_ids is in the provided list. If empty list is provided, returns without
                    yielding any QAs.
        """
        logger.debug(f"Starting query iteration for {self.name} dataset")
        if not self.qas_file.exists():
            logger.info("QAs file not found, triggering load()")
            self.load()

        # Helper function to check if a QA matches document IDs
        def matches_doc_ids(qa: dict, doc_ids_set: set) -> bool:
            if self.qa_type == IntraDocumentQA:
                return qa['document_id'] in doc_ids_set
            else:  # InterDocumentQA
                return any(doc_id in doc_ids_set for doc_id in qa['document_ids'])

        # If specific QA IDs are requested
        if qa_ids is not None:
            # Return immediately if empty list provided
            if not qa_ids:
                logger.warning("Empty QA ID list provided, returning without QAs")
                return
                
            qa_ids_set = set(qa_ids)  # Convert to set for O(1) lookup
            with open(self.qas_file, 'rb') as f:
                for qa in ijson.items(f, "item"):
                    if qa['id'] in qa_ids_set:
                        # If doc_ids is also specified, check document match
                        if doc_ids:
                            if matches_doc_ids(qa, set(doc_ids)):
                                yield self.qa_type.model_validate(qa)
                        else:
                            yield self.qa_type.model_validate(qa)
            return

        # If specific document IDs are requested
        if doc_ids is not None:
            # Return immediately if empty list provided
            if not doc_ids:
                logger.warning("Empty document ID list provided, returning without QAs")
                return
                
            doc_ids_set = set(doc_ids)  # Convert to set for O(1) lookup
            with open(self.qas_file, 'rb') as f:
                for qa in ijson.items(f, "item"):
                    if matches_doc_ids(qa, doc_ids_set):
                        yield self.qa_type.model_validate(qa)
            return

        # For random selection, we need to know total count first
        if selection_mode == "random" and num_qas is not None:
            total_qas = self._count_items("")  # Empty key since we're using simple array
            if total_qas == 0:
                return
                
            # Generate random indices
            indices = set(random.sample(range(total_qas), min(num_qas, total_qas)))
            
            # Stream and yield only selected indices
            with open(self.qas_file, 'rb') as f:
                for idx, qa in enumerate(ijson.items(f, "item")):
                    if idx in indices:
                        yield self.qa_type.model_validate(qa)
        else:
            # For sequential access, stream directly
            with open(self.qas_file, 'rb') as f:
                for idx, qa in enumerate(ijson.items(f, "item")):
                    if num_qas is not None and idx >= num_qas:
                        break
                    yield self.qa_type.model_validate(qa)

    @abstractmethod
    def _process_raw_data(self) -> Tuple[List[Document], List[Union[IntraDocumentQA, InterDocumentQA]]]:
        """Process raw data into documents and QAs. Must be implemented by each dataset."""
        logger.info(f"Processing raw data for {self.name} dataset")
        pass 