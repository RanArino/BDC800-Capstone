# test/simple_rag.py

"""
Simple RAG test
Run the script to debug;
```
python -m pdb test/simple_rag.py
```
"""

import sys
import os
import gc
import warnings
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.frameworks import SimpleRAG
from core.datasets import Qasper, IntraDocumentQA
from core.logger.logger import get_logger

# Set up logging
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    try:
        # Initialize RAG
        logger.info("Initializing SimpleRAG")
        simple_rag = SimpleRAG("simple_rag_01")
        
        # Load dataset
        dataset = Qasper()
        dataset.load("test")
        documents = list(dataset.get_documents())[0:1]  # Get first document
        logger.info(f"Loaded {len(documents)} documents")
        
        # Index documents
        logger.info("Indexing documents")
        simple_rag.index(documents)
        logger.info("Indexing completed")
        
        # Test retrieval and generation
        qa = list(dataset.get_queries())[0]
        logger.info(f"Testing RAG with query: {qa.q}")
        result = simple_rag.run(qa)
        logger.info("RAG test completed")
        logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
