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
import json
from pathlib import Path

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
        gen_docs, gen_qas = simple_rag.load_dataset(number_of_docs=1)

        # # Check if gen_docs and gen_qas are not empty
        # try:
        #     first_doc = next(gen_docs)
        #     print(f"First document: \n id: {first_doc.id} \n text: {first_doc.content[:120]}")
        # except StopIteration:
        #     print("No documents found in gen_docs")
        
        # try:
        #     first_qa_list = next(gen_qas)
        #     print(f"QA IDs: {[qa.document_id for qa in first_qa_list]}")
        # except StopIteration:
        #     print("No QAs found in gen_qas")
        
        # load each document and qa pair
        for doc, qas in zip(gen_docs, gen_qas):
            # Index documents
            logger.info("Indexing documents")
            simple_rag.index(doc)
            logger.info("Indexing completed")
            # Test retrieval and generation
            logger.info(f"Testing RAG with query")
            response_list = []
            for qa in qas:
                response = simple_rag.run(qa)
                response_list.append(response)
            logger.info("RAG test completed")

            # print the metrics
            print(simple_rag.profiler.get_metrics(include_counts=True))

            # store the response_list in a json file, use it for evaluation test
            response_dir = Path("test/input_data")
            response_dir.mkdir(parents=True, exist_ok=True)
            response_file = response_dir / f"RAGResponse_qasper_test.json"
            with open(response_file, 'w') as f:
                json.dump([response.model_dump() for response in response_list], f, indent=2)
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        gc.collect()

def multihoprag_test():
    try:
        # Initialize RAG
        logger.info("Initializing SimpleRAG")
        simple_rag = SimpleRAG("simple_rag_multihoprag_test")
        gen_docs, gen_qas = simple_rag.load_dataset(number_of_qas=10)

        # Index documents
        logger.info("Indexing documents")
        simple_rag.index(gen_docs)
        logger.info("Indexing completed")

        # Test retrieval and generation
        logger.info(f"Testing RAG with query")
        response_list = []
        for qa in gen_qas:
            response = simple_rag.run(qa)
            response_list.append(response)
        logger.info("RAG test completed")
        
        # store the response_list in a json file, use it for evaluation test
        response_dir = Path("test/input_data")
        response_dir.mkdir(parents=True, exist_ok=True)
        response_file = response_dir / f"RAGResponse_multihoprag_test.json"
        with open(response_file, 'w') as f:
            json.dump([response.model_dump() for response in response_list], f, indent=2)

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    import time
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        qasper_test()
        print("\n===== qasper_test completed =====")
        print("start multihoprag_test in 3 seconds...")
        time.sleep(3)
        multihoprag_test()
