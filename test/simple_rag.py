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
from core.logger.logger import get_logger

# Set up logging
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

def qasper_test():
    try:
        # Initialize RAG
        logger.info("Initializing SimpleRAG")
        simple_rag = SimpleRAG("simple_rag_qasper_test")
        gen_docs, gen_qas = simple_rag.load_dataset()

        # Check if gen_docs and gen_qas are not empty
        try:
            first_doc = next(gen_docs)
            print(f"First document: \n id: {first_doc.id} \n text: {first_doc.content[:120]}")
        except StopIteration:
            print("No documents found in gen_docs")
        
        try:
            first_qa_list = next(gen_qas)
            print(f"QA IDs: {[qa.document_id for qa in first_qa_list]}")
        except StopIteration:
            print("No QAs found in gen_qas")
        
        # # Index documents
        # logger.info("Indexing documents")
        # simple_rag.index(gen_docs)
        # logger.info("Indexing completed")
        
        # # Test retrieval and generation
        # qa = list(dataset.get_queries())[0]
        # logger.info(f"Testing RAG with query: {qa.q}")
        # result = simple_rag.run(qa)
        # logger.info("RAG test completed")
        # logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        qasper_test()
