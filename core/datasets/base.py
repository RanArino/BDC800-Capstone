"""
Base dataset class and interfaces for RAG framework datasets.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, Optional
from pathlib import Path
import json

from core.datasets.schema import Dataset, Document

class BaseDataset(ABC):
    """Base class for all datasets used in the RAG framework."""
    
    def __init__(self, name: str):
        self.name = name
        self.data_dir = Path(__file__).parent / "data" / name
        self.data_file = self.data_dir / "data.json"
        self._test_mode = False
    
    def load(self, mode: Optional[str] = None) -> None:
        """Process raw data and save to JSON if not exists.
        
        Args:
            mode: Optional mode for loading data. If 'test', only processes first 5 items.
        """
        self._test_mode = mode == "test"
        test_suffix = "_test" if self._test_mode else ""
        self.data_file = self.data_dir / f"data{test_suffix}.json"
        
        if not self.data_file.exists():
            dataset = self._process_raw_data()
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_file, 'w') as f:
                json.dump(dataset.model_dump(), f, indent=2)

    def get_documents(self) -> Generator[Dict[str, Any], None, None]:
        """Get documents one by one from the dataset."""
        if not self.data_file.exists():
            self.load()
            
        with open(self.data_file, 'r') as f:
            data = json.load(f)
            for doc in data['documents']:
                yield doc
    
    def get_queries(self) -> Generator[Dict[str, Any], None, None]:
        """Get queries (QA pairs) one by one from the dataset."""
        if not self.data_file.exists():
            self.load()
            
        with open(self.data_file, 'r') as f:
            data = json.load(f)
            # Yield from whichever QA list is non-empty
            if data['intra_qas']:
                for qa in data['intra_qas']:
                    yield qa
            else:
                for qa in data['inter_qas']:
                    yield qa

    @abstractmethod
    def _process_raw_data(self) -> Dataset:
        """Process raw data into our schema format. Must be implemented by each dataset."""
        pass 