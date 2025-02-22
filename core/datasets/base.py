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
            
            # Save QAs
            with open(self.qas_file, 'w') as f:
                qa_data = {
                    'intra_qas': [qa.model_dump() for qa in qas if isinstance(qa, IntraDocumentQA)],
                    'inter_qas': [qa.model_dump() for qa in qas if isinstance(qa, InterDocumentQA)]
                }
                json.dump(qa_data, f, indent=2)
            logger.info(f"Processed and saved QAs to {self.qas_file}")
        else:
            logger.info(f"Using existing data files")

    def get_documents(self) -> Generator[Document, None, None]:
        """Get documents one by one from the dataset."""
        logger.debug(f"Starting document iteration for {self.name} dataset")
        if not self.data_file.exists():
            logger.info("Data file not found, triggering load()")
            self.load()
            
        with open(self.data_file, 'r') as f:
            data = json.load(f)
            logger.info(f"Found documents in dataset")
            for doc_dict in data['documents']:
                yield Document.model_validate(doc_dict)
    
    def get_queries(self) -> Generator[IntraDocumentQA | InterDocumentQA, None, None]:
        """Get queries (QA pairs) one by one from the dataset."""
        logger.debug(f"Starting query iteration for {self.name} dataset")
        if not self.data_file.exists():
            logger.info("Data file not found, triggering load()")
            self.load()
            
        with open(self.data_file, 'r') as f:
            data = json.load(f)
            # Yield from whichever QA list is non-empty
            if data['intra_qas']:
                logger.info(f"Found intra-document QA pairs")
                for qa_dict in data['intra_qas']:
                    yield IntraDocumentQA.model_validate(qa_dict)
            else:
                logger.info(f"Found inter-document QA pairs")
                for qa_dict in data['inter_qas']:
                    yield InterDocumentQA.model_validate(qa_dict)

    @abstractmethod
    def _process_raw_data(self) -> Tuple[List[Document], List[Union[IntraDocumentQA, InterDocumentQA]]]:
        """Process raw data into documents and QAs. Must be implemented by each dataset."""
        logger.info(f"Processing raw data for {self.name} dataset")
        pass 