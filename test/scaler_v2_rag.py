"""
SCALER V2 RAG test
Run the script to debug;
```
python -m pdb test/scaler_v2_rag.py
```
"""

import sys
import os
import gc
import warnings
import json
import yaml
from pathlib import Path
from typing import List, Dict
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.frameworks import ScalerV2RAG, RAGResponse
from core.logger.logger import get_logger
from core.evaluation.metrics_summary import accumulate_and_summarize_metrics, MetricsSummary

from langchain_core.documents import Document as LangChainDocument

# Set up logging
logger = get_logger(__name__)

def multihoprag_test():
    try:
        # Initialize RAG
        logger.info("Initializing SCALER V2 RAG")
        scaler_v2_rag = ScalerV2RAG("TEST_scaler_v2_rag_multihoprag", is_save_vectorstore=True)
        gen_docs, gen_qas = scaler_v2_rag.load_dataset()

        # Index documents
        logger.info("Indexing documents")
        scaler_v2_rag.index(gen_docs)
        logger.info("Indexing completed")

        # Test retrieval and generation
        logger.info("Testing RAG with queries")
        all_responses, all_metrics = scaler_v2_rag.run(gen_qas) 
        logger.info("RAG test completed")

        # Build overall metrics summary
        overall_metrics_summary, detailed_df = accumulate_and_summarize_metrics(
            metrics_list=all_metrics,
            profiler_metrics=scaler_v2_rag.profiler.get_metrics()
        ) 

        # print the metrics
        print(scaler_v2_rag.profiler.get_metrics(include_counts=True))
        
        # store the response_list in a json file, use it for evaluation test
        response_dir = Path("test/input_data")
        response_dir.mkdir(parents=True, exist_ok=True)
        response_file = response_dir / f"RAGResponse_scaler_v2_multihoprag_test.json"
        with open(response_file, 'w') as f:
            json.dump([response.model_dump() for response in all_responses], f, indent=2)

        # save the metrics summary
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_summary_file = output_dir / f"scaler_v2_multihoprag_overall_metrics_summary.json"
        with open(metrics_summary_file, 'w') as f:
            json.dump(overall_metrics_summary.model_dump(), f, indent=2)

        # save the detailed dataframe
        detailed_df_file = output_dir / f"scaler_v2_multihoprag_detailed_metrics_summary.csv"
        detailed_df.to_csv(detailed_df_file, index=False)

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        gc.collect()

def comparison_test():
    """Run both ScalerV1 and ScalerV2 and compare their performance"""
    try:
        from core.frameworks import ScalerV1RAG
        import time
        import pandas as pd
        
        logger.info("Starting comparison test between ScalerV1 and ScalerV2")
        
        # Initialize both RAG systems
        logger.info("Initializing SCALER V1 and V2 RAG")
        scaler_v1_rag = ScalerV1RAG("TEST_scaler_v1_rag_multihoprag", is_save_vectorstore=False)
        scaler_v2_rag = ScalerV2RAG("TEST_scaler_v2_rag_multihoprag", is_save_vectorstore=False)
        
        # Load dataset (use the same dataset for both)
        gen_docs, gen_qas = scaler_v1_rag.load_dataset()
        # Convert to list to reuse
        docs_list = list(gen_docs)
        qas_list = list(gen_qas)
        
        # Index documents
        logger.info("Indexing documents for both systems")
        
        # Time ScalerV1 indexing
        start_time = time.time()
        scaler_v1_rag.index(docs_list)
        v1_index_time = time.time() - start_time
        logger.info(f"ScalerV1 indexing completed in {v1_index_time:.2f} seconds")
        
        # Time ScalerV2 indexing
        start_time = time.time()
        scaler_v2_rag.index(docs_list)
        v2_index_time = time.time() - start_time
        logger.info(f"ScalerV2 indexing completed in {v2_index_time:.2f} seconds")
        
        # Test retrieval (without generation to focus on retrieval performance)
        logger.info("Testing retrieval performance")
        
        # Time ScalerV1 retrieval
        start_time = time.time()
        v1_responses, v1_metrics = scaler_v1_rag.run(qas_list, llm_generation=False)
        v1_retrieval_time = time.time() - start_time
        logger.info(f"ScalerV1 retrieval completed in {v1_retrieval_time:.2f} seconds")
        
        # Time ScalerV2 retrieval
        start_time = time.time()
        v2_responses, v2_metrics = scaler_v2_rag.run(qas_list, llm_generation=False)
        v2_retrieval_time = time.time() - start_time
        logger.info(f"ScalerV2 retrieval completed in {v2_retrieval_time:.2f} seconds")
        
        # Build metrics summaries
        v1_summary, v1_df = accumulate_and_summarize_metrics(
            metrics_list=v1_metrics,
            profiler_metrics=scaler_v1_rag.profiler.get_metrics()
        )
        
        v2_summary, v2_df = accumulate_and_summarize_metrics(
            metrics_list=v2_metrics,
            profiler_metrics=scaler_v2_rag.profiler.get_metrics()
        )
        
        # Create comparison report
        comparison = {
            "indexing": {
                "v1_time": v1_index_time,
                "v2_time": v2_index_time,
                "speedup": v1_index_time / v2_index_time if v2_index_time > 0 else float('inf')
            },
            "retrieval": {
                "v1_time": v1_retrieval_time,
                "v2_time": v2_retrieval_time,
                "speedup": v1_retrieval_time / v2_retrieval_time if v2_retrieval_time > 0 else float('inf')
            },
            "metrics": {
                "v1": v1_summary.model_dump(),
                "v2": v2_summary.model_dump()
            }
        }
        
        # Save comparison report
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_file = output_dir / "scaler_v1_vs_v2_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
            
        # Print summary
        print("\n===== Comparison Summary =====")
        print(f"Indexing: ScalerV2 is {comparison['indexing']['speedup']:.2f}x faster than ScalerV1")
        print(f"Retrieval: ScalerV2 is {comparison['retrieval']['speedup']:.2f}x faster than ScalerV1")
        
        # Create and save comparison chart
        try:
            import matplotlib.pyplot as plt
            
            # Prepare data
            categories = ['Indexing Time', 'Retrieval Time']
            v1_times = [v1_index_time, v1_retrieval_time]
            v2_times = [v2_index_time, v2_retrieval_time]
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(categories))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], v1_times, width, label='ScalerV1')
            ax.bar([i + width/2 for i in x], v2_times, width, label='ScalerV2')
            
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Performance Comparison: ScalerV1 vs ScalerV2')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            
            # Add speedup annotations
            for i in range(len(categories)):
                speedup = v1_times[i] / v2_times[i] if v2_times[i] > 0 else float('inf')
                ax.annotate(f'{speedup:.2f}x faster',
                            xy=(i, min(v1_times[i], v2_times[i]) / 2),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / "scaler_v1_vs_v2_comparison.png")
            print("Comparison chart saved to test/output_data/scaler_v1_vs_v2_comparison.png")
        except Exception as e:
            logger.warning(f"Could not create comparison chart: {e}")
        
    except Exception as e:
        logger.error(f"Error in comparison test: {e}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ScalerV2 RAG tests')
    parser.add_argument('--comparison', action='store_true', help='Run comparison test between ScalerV1 and ScalerV2')
    parser.add_argument('--multihop', action='store_true', help='Run multihop RAG test for ScalerV2')
    args = parser.parse_args()
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        if args.multihop or (not args.comparison and not args.multihop):
            multihoprag_test()
            print("\n===== multihoprag_test completed =====")
        
        # if args.comparison:
        #     print("Starting comparison test in 3 seconds...")
        #     time.sleep(3)
        #     comparison_test()
        #     print("\n===== comparison_test completed =====")
