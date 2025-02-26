# core/utils/process_bar.py

import os
from tqdm import tqdm
from core.datasets import IntraDocumentQA
from core.utils import get_project_root
import logging

class ProgressTracker:
    """Progress tracking utility for RAG frameworks."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize progress tracker.
        
        Args:
            logger: Logger instance to use for logging
        """
        self.logger = logger
        self.progress_bar = None
        self.total_qa_count = 0
        self.processed_qa_count = 0
    
    def initialize(self, dataset, dataset_config, number_of_docs=None, number_of_qas=None):
        """Initialize progress tracking based on dataset type and configuration.
        
        Args:
            dataset: The dataset instance
            dataset_config: Dataset configuration
            number_of_docs: Optional number of documents to process
            number_of_qas: Optional number of QAs to process
        """
        # Reset progress tracking
        self.processed_qa_count = 0
        
        # Get dataset info from YAML config if available
        dataset_info = self._get_dataset_info_from_yaml(dataset_config.name)
        
        if dataset.qa_type == IntraDocumentQA:
            # For IntraDoc, we track by documents
            # Prioritize explicitly passed number_of_docs, then dataset_config, then yaml info
            doc_count = number_of_docs or dataset_config.number_of_docs or dataset_info.get("number_of_docs", 0)
            
            # Use doc_count directly for IntraDocumentQA
            self.total_qa_count = doc_count
                
            self.logger.info(f"Tracking progress for {doc_count} documents")
            # Create progress bar for documents
            self.progress_bar = tqdm(total=self.total_qa_count, desc="Processing documents")
        else:
            # For InterDoc, we track by questions directly
            # Prioritize explicitly passed number_of_qas, then dataset_config, then yaml info
            self.total_qa_count = number_of_qas or dataset_config.number_of_qas or dataset_info.get("number_of_qas", 0)
            self.logger.info(f"Tracking progress for {self.total_qa_count} questions")
            # Create progress bar for questions
            self.progress_bar = tqdm(total=self.total_qa_count, desc="Processing questions")
    
    def update(self, count=1):
        """Update progress by the specified count.
        
        Args:
            count: Number of items to increment progress by
        """
        if self.progress_bar is not None:
            self.progress_bar.update(count)
            self.processed_qa_count += count
    
    def close(self):
        """Close the progress bar if it exists."""
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
    
    def _get_dataset_info_from_yaml(self, dataset_name) -> dict:
        """Get dataset information from datasets.yaml file.
        
        Args:
            dataset_name: Name of the dataset to get info for
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            datasets_yaml_path = os.path.join(get_project_root(), "core/configs/datasets.yaml")
            if os.path.exists(datasets_yaml_path):
                # Read the file manually and parse it
                with open(datasets_yaml_path, "r") as f:
                    content = f.read()
                
                # Parse the custom YAML format
                datasets_info = {}
                current_dataset = None
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    # Skip empty lines and comments
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    
                    # If line is not indented, it's a dataset name
                    if not line.startswith(' ') and not line.startswith('\t'):
                        current_dataset = line.strip()
                        datasets_info[current_dataset] = {}
                    # If line is indented and has a colon, it's a property
                    elif current_dataset and (':' in line):
                        parts = line.strip().split(':', 1)
                        if len(parts) == 2:
                            key, value = parts
                            key = key.strip()
                            value = value.strip()
                            
                            # Try to convert to appropriate type
                            try:
                                value = int(value)
                            except ValueError:
                                pass
                                
                            datasets_info[current_dataset][key] = value
                
                # Return the info for the requested dataset
                if dataset_name in datasets_info:
                    return datasets_info[dataset_name]
            
            return {}
        except Exception as e:
            self.logger.warning(f"Could not load dataset info from YAML: {str(e)}")
            return {}