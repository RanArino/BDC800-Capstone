# experiments/base.py

"""
Run Simple RAG experiments with memory-efficient metrics calculation
"""

import sys
import os
import gc
import warnings
import json
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.frameworks import SimpleRAG, ScalerRAG, RAGResponse
from core.datasets import IntraDocumentQA, InterDocumentQA
from core.logger.logger import get_logger
from core.evaluation.metrics_summary import accumulate_and_summarize_metrics
from core.evaluation.schema import MetricsSummary

# Set up logging
logger = get_logger(__name__)

def run_experiment(
        config_path: str,
        config_name: str,
        store_responses: bool = True,
        store_metrics: bool = True,
        store_detailed_df: bool = True,
        llm_generation: bool = True
    ):
    """
    Run RAG test for a specific dataset type with memory-efficient metrics calculation.
    """
    try:
        # Initialize RAG
        
        if Path(config_path).parent.name == "simple_rag":
            logger.info(f"Initializing SimpleRAG for {config_name}")
            rag = SimpleRAG(config_name, config_path, True)
        elif Path(config_path).parent.name == "scaler_rag":
            logger.info(f"Initializing ScalerRAG for {config_name}")
            rag = ScalerRAG(config_name, config_path, True)
        else:
            raise ValueError(f"Invalid config path: {config_path}. Must be in the simple_rag directory.")

        # Load dataset
        gen_docs, gen_qas = rag.load_dataset()

        # For IntraDocumentQA
        if rag.dataset.qa_type == IntraDocumentQA:
            logger.info("Running RAG for IntraDocumentQA")
            # Initialize accumulators for metrics
            all_responses: List[RAGResponse] = []
            all_metrics: List[MetricsSummary] = []
            
            for doc, qas in zip(gen_docs, gen_qas):
                if qas == []:
                    rag.progress_tracker.update(1)
                    continue
                
                # Indexing
                with rag.profiler.track("indexing"):
                    rag.index(doc)
                # Retrieval & Generation
                response_list, metrics_list = rag.run(qas, llm_generation)

                all_responses.extend(response_list)
                all_metrics.extend(metrics_list)

                # Update progress for each document
                rag.progress_tracker.update(1)

                # Force garbage collection after each document
                gc.collect()
        
        # For InterDocumentQA
        elif rag.dataset.qa_type == InterDocumentQA:
            logger.info("Running RAG for InterDocumentQA")
            rag.index(gen_docs)
            logger.info("Indexing completed")

            # Test retrieval and generation
            all_responses, all_metrics = rag.run(gen_qas, llm_generation) 
            logger.info("RAG test completed")

            # Force garbage collection after each document
            gc.collect()
            
        else:
            raise ValueError(f"Invalid dataset type: {rag.dataset.qa_type}. Must be one of {IntraDocumentQA, InterDocumentQA}")
            
        # Build overall metrics summary
        overall_metrics_summary, detailed_df = accumulate_and_summarize_metrics(
            metrics_list=all_metrics,
            profiler_metrics=rag.profiler.get_metrics()
        ) 

        # get date and time as a string (i.e., 20250223-091030)
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Store response_list if requested
        if store_responses:
            response_dir = Path("experiments/responses")
            response_dir.mkdir(parents=True, exist_ok=True)
            response_file = response_dir / f"{config_name}-{date_time}.json"
            logger.info(f"Storing responses to {response_file}")
            with open(response_file, 'w') as f:
                json.dump([response.model_dump() for response in all_responses], f, indent=2)
        
        # Store metrics_summary if requested
        if store_metrics:
            metrics_dir = Path("experiments/metrics")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_summary_file = metrics_dir / f"{config_name}-{date_time}.json"
            logger.info(f"Storing metrics summary to {metrics_summary_file}")
            with open(metrics_summary_file, 'w') as f:
                json.dump(overall_metrics_summary.model_dump(), f, indent=2)
        
        # Store detailed_df if requested
        if store_detailed_df and detailed_df is not None:
            detailed_dir = Path("experiments/detailed_dfs")
            detailed_dir.mkdir(parents=True, exist_ok=True)
            detailed_df_file = detailed_dir / f"{config_name}-{date_time}.csv"
            logger.info(f"Storing detailed metrics to {detailed_df_file}")
            detailed_df.to_csv(detailed_df_file, index=False)
        
    except Exception as e:
        logger.error(f"Error occurred in {config_name} test: {e}")
        raise
    finally:
        gc.collect()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # define config_path and config_name
        config_path = "core/configs/simple_rag/test.yaml"
        config_name = ["TEST02_simple_rag_01", "TEST02_simple_rag_02"]
        # run the simple_rag experiment
        for config in config_name:
            print(f"\n===== Starting {config} test =====")
            run_experiment(config_path, config, llm_generation=True)
            print(f"\n===== {config} test completed =====")

        # run the scaler_rag experiment
        config_path = "core/configs/scaler_rag/test.yaml"
        config_name = ["TEST02_scaler_rag_01", "TEST02_scaler_rag_02"]
        for config in config_name:
            print(f"\n===== Starting {config} test =====")
            run_experiment(config_path, config, llm_generation=True)
            print(f"\n===== {config} test completed =====")
