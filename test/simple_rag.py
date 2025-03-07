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
import json
from pathlib import Path
from typing import List
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.frameworks import SimpleRAG, RAGResponse
from core.logger.logger import get_logger
from core.evaluation.metrics_summary import accumulate_and_summarize_metrics
from core.evaluation.schema import MetricsSummary
# Set up logging
logger = get_logger(__name__)


def qasper_test():
    try:
        # Initialize RAG
        logger.info("Initializing SimpleRAG")
        simple_rag = SimpleRAG("TEST_simple_rag_qasper", is_save_vectorstore=True)
        gen_docs, gen_qas = simple_rag.load_dataset()
        
        # Initialize accumulators for metrics
        all_responses: List[RAGResponse] = []
        all_metrics: List[MetricsSummary] = []

        # load each document and qa pair
        for doc, qas in zip(gen_docs, gen_qas):
            # Index documents
            logger.info("Indexing documents")
            simple_rag.index(doc)
            logger.info("Indexing completed")
            
            # Test retrieval and generation
            logger.info("Testing RAG with queries")
            response_list, metrics_list = simple_rag.run(qas)  # Process list of QAs
            logger.info("RAG test completed")

            # Update accumulators
            all_responses.extend(response_list)
            all_metrics.extend(metrics_list)

        # Build overall metrics summary
        overall_metrics_summary, detailed_df = accumulate_and_summarize_metrics(
            metrics_list=all_metrics,
            profiler_metrics=simple_rag.profiler.get_metrics()
        ) 
        
        # print the metrics
        print(simple_rag.profiler.get_metrics(include_counts=True))

        # store the response_list in a json file, use it for evaluation test
        response_dir = Path("test/input_data")
        response_dir.mkdir(parents=True, exist_ok=True)
        response_file = response_dir / f"RAGResponse_qasper_test.json"
        with open(response_file, 'w') as f:
            json.dump([response.model_dump() for response in all_responses], f, indent=2)

        # save the metrics summary
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary_file = output_dir / f"qasper_overall_metrics_summary.json"
        with open(metrics_summary_file, 'w') as f:
            json.dump(overall_metrics_summary.model_dump(), f, indent=2)

        # save the detailed dataframe
        detailed_df_file = output_dir / f"qasper_detailed_metrics_summary.csv"
        detailed_df.to_csv(detailed_df_file, index=False)

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        gc.collect()

def multihoprag_test():
    try:
        # Initialize RAG
        logger.info("Initializing SimpleRAG")
        simple_rag = SimpleRAG("TEST_simple_rag_multihoprag")
        gen_docs, gen_qas = simple_rag.load_dataset()

        # Index documents
        logger.info("Indexing documents")
        simple_rag.index(gen_docs)
        logger.info("Indexing completed")

        # Test retrieval and generation
        logger.info("Testing RAG with queries")
        all_responses, all_metrics = simple_rag.run(gen_qas) 
        logger.info("RAG test completed")

        # Build overall metrics summary
        overall_metrics_summary, detailed_df = accumulate_and_summarize_metrics(
            metrics_list=all_metrics,
            profiler_metrics=simple_rag.profiler.get_metrics()
        ) 

        # print the metrics
        print(simple_rag.profiler.get_metrics(include_counts=True))
        
        # store the response_list in a json file, use it for evaluation test
        response_dir = Path("test/input_data")
        response_dir.mkdir(parents=True, exist_ok=True)
        response_file = response_dir / f"RAGResponse_multihoprag_test.json"
        with open(response_file, 'w') as f:
            json.dump([response.model_dump() for response in all_responses], f, indent=2)

        # save the metrics summary
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary_file = output_dir / f"multihoprag_overall_metrics_summary.json"
        with open(metrics_summary_file, 'w') as f:
            json.dump(overall_metrics_summary.model_dump(), f, indent=2)

        # save the detailed dataframe
        detailed_df_file = output_dir / f"multihoprag_detailed_metrics_summary.csv"
        detailed_df.to_csv(detailed_df_file, index=False)

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
        