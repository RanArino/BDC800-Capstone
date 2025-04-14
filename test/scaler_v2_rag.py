# test/scaler_v1_2_rag.py

"""
SCALER V1.2 RAG test
Run the script to debug;
```
python -m pdb test/scaler_v1_2_rag.py
```
"""

import sys
import os
import gc
import warnings
import json
import yaml
from pathlib import Path
from typing import List
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.frameworks import ScalerV2RAG, RAGResponse
from core.logger.logger import get_logger
from core.evaluation.metrics_summary import accumulate_and_summarize_metrics, MetricsSummary

# Set up logging
logger = get_logger(__name__)

def multihoprag_test():
    try:
        # Initialize RAG
        logger.info("Initializing SCALER V1.2 RAG")
        scaler_rag = ScalerV2RAG("TEST02_scaler_v2_rag_01", is_save_vectorstore=True)
        gen_docs, gen_qas = scaler_rag.load_dataset()

        # Index documents
        logger.info("Indexing documents")
        scaler_rag.index(gen_docs)
        logger.info("Indexing completed")

        # for qa in gen_qas:
        #     retrieved_docs = scaler_rag.retrieve(qa.q)

        #     print("Completed")

        # Test retrieval and generation
        logger.info("Testing RAG with queries")
        all_responses, all_metrics = scaler_rag.run(gen_qas) 
        logger.info("RAG test completed")


        # Build overall metrics summary
        overall_metrics_summary, detailed_df = accumulate_and_summarize_metrics(
            metrics_list=all_metrics,
            profiler_metrics=scaler_rag.profiler.get_metrics()
        ) 

        # print the metrics
        print(scaler_rag.profiler.get_metrics(include_counts=True))
        
        # store the response_list in a json file, use it for evaluation test
        response_dir = Path("test/input_data")
        response_dir.mkdir(parents=True, exist_ok=True)
        response_file = response_dir / f"RAGResponse_scaler_v1_2_multihoprag_test.json"
        with open(response_file, 'w') as f:
            json.dump([response.model_dump() for response in all_responses], f, indent=2)

        # save the metrics summary
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary_file = output_dir / f"scaler_v1_2_multihoprag_overall_metrics_summary.json"
        with open(metrics_summary_file, 'w') as f:
            json.dump(overall_metrics_summary.model_dump(), f, indent=2)

        # save the detailed dataframe
        detailed_df_file = output_dir / f"scaler_v1_2_multihoprag_detailed_metrics_summary.csv"
        detailed_df.to_csv(detailed_df_file, index=False)

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        gc.collect()

def frames_test():
    try:
        # Initialize RAG
        logger.info("Initializing SCALER V2 RAG")
        scaler_rag = ScalerV2RAG("TEST02_scaler_v2_rag_02", is_save_vectorstore=True)
        gen_docs, gen_qas = scaler_rag.load_dataset()

        # Index documents
        logger.info("Indexing documents")
        scaler_rag.index(gen_docs)
        logger.info("Indexing completed")

        # check retrieval method
        # for qa in gen_qas:
        #     retrieved_docs = scaler_rag.retrieve(qa.q)

        #     print("Completed")

        # Test retrieval and generation
        logger.info("Testing RAG with queries")
        all_responses, all_metrics = scaler_rag.run(gen_qas, llm_generation=False) 
        logger.info("RAG test completed")


        # Build overall metrics summary
        overall_metrics_summary, detailed_df = accumulate_and_summarize_metrics(
            metrics_list=all_metrics,
            profiler_metrics=scaler_rag.profiler.get_metrics()
        ) 

        # print the metrics
        print(scaler_rag.profiler.get_metrics(include_counts=True))
        
        # store the response_list in a json file, use it for evaluation test
        response_dir = Path("test/input_data")
        response_dir.mkdir(parents=True, exist_ok=True)
        response_file = response_dir / f"RAGResponse_scaler_v1_2_multihoprag_test.json"
        with open(response_file, 'w') as f:
            json.dump([response.model_dump() for response in all_responses], f, indent=2)

        # save the metrics summary
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary_file = output_dir / f"scaler_v1_2_multihoprag_overall_metrics_summary.json"
        with open(metrics_summary_file, 'w') as f:
            json.dump(overall_metrics_summary.model_dump(), f, indent=2)

        # save the detailed dataframe
        detailed_df_file = output_dir / f"scaler_v1_2_multihoprag_detailed_metrics_summary.csv"
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
        frames_test()
        print("\n===== frames_test completed =====")
        print("start multihoprag_test in 3 seconds...")
        # time.sleep(3)
        # multihoprag_test()
        # print("\n===== multihoprag_test completed =====")
        # print("start comparison test in 3 seconds...")
        # time.sleep(3)