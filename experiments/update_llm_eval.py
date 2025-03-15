import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from core.evaluation.metrics.self_checker import check_retrieval_chunks, SelfCheckerModel
from core.evaluation.schema import SelfCheckerAnswer
from langchain.schema import Document
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(experiment_id: str) -> tuple[pd.DataFrame, dict, list]:
    """
    Load data from detailed_dfs, metrics, and responses directories for a given experiment ID.
    
    Args:
        experiment_id (str): The experiment ID (e.g., 'simple_rag_02_04-20250315-232123')
        
    Returns:
        tuple: (detailed_df, metrics_data, responses_data)
    """
    base_path = Path("experiments")
    
    try:
        # Load detailed DataFrame
        detailed_df = pd.read_csv(base_path / "detailed_dfs" / f"{experiment_id}.csv")
        
        # Load metrics
        with open(base_path / "metrics" / f"{experiment_id}.json", 'r') as f:
            metrics_data = json.load(f)
        
        # Load responses
        with open(base_path / "responses" / f"{experiment_id}.json", 'r') as f:
            responses_data = json.load(f)
        
        return detailed_df, metrics_data, responses_data
    except Exception as e:
        logging.error(f"Error loading data for experiment {experiment_id}: {str(e)}")
        return None, None, None

def create_document_from_context(context_obj: dict) -> Document:
    """Create a Langchain Document from context object in the response JSON."""
    # Extract metadata and page_content from the context object
    metadata = context_obj.get('metadata', {})
    page_content = context_obj.get('page_content', '')
    
    return Document(page_content=page_content, metadata=metadata)

def evaluate_llm(
    qa_id: str,
    query: str,
    ground_truth: str,
    contexts: List[Document],  # Changed to accept Documents directly
    top_k_values: List[int] = [1, 3, 5, 10],
    model: Optional[SelfCheckerModel] = None
) -> Dict[int, SelfCheckerAnswer]:
    """
    Evaluate retrieved contexts using LLM evaluation.
    
    Args:
        qa_id (str): Question ID
        query (str): Original question
        ground_truth (str): Ground truth answer
        contexts (List[Document]): List of retrieved Langchain Documents
        top_k_values (List[int]): List of k values to evaluate
        model (Optional[SelfCheckerModel]): Model to use for evaluation
        
    Returns:
        Dict[int, SelfCheckerAnswer]: Evaluation results for each k value
    """
    # Evaluate using check_retrieval_chunks
    try:
        results = check_retrieval_chunks(
            qa_id=qa_id,
            question=query,
            ground_truth_answer=ground_truth,
            retrieval_chunks=contexts,  # Pass documents directly
            top_k=top_k_values,
            model=model
        )
        return results
    except Exception as e:
        logging.error(f"Error evaluating QA pair {qa_id}: {str(e)}")
        return {}

def update_metrics_with_llm_eval(metrics_data: dict, llm_eval_results: Dict[str, Dict[int, float]]) -> dict:
    """
    Update metrics data with LLM evaluation results.
    
    Args:
        metrics_data (dict): Original metrics data
        llm_eval_results (Dict[str, Dict[int, float]]): LLM evaluation results per QA pair
        
    Returns:
        dict: Updated metrics data
    """
    # Calculate average LLM eval scores for each k
    k_values = set()
    for results in llm_eval_results.values():
        k_values.update(results.keys())
    
    avg_scores = {}
    for k in k_values:
        scores = [results[k] for results in llm_eval_results.values() if k in results]
        avg_scores[str(k)] = sum(scores) / len(scores) if scores else 0.0
    
    # Update metrics
    if 'retrieval' not in metrics_data:
        metrics_data['retrieval'] = {}
    elif metrics_data['retrieval'] is None:
        # Handle the case where 'retrieval' exists but is None
        metrics_data['retrieval'] = {}
    
    metrics_data['retrieval']['llm_eval'] = avg_scores
    
    return metrics_data

def process_experiment(experiment_id: str):
    """
    Process a single experiment.
    
    Args:
        experiment_id (str): The experiment ID to process
    """
    print(f"\n\n{'='*80}")
    print(f"Processing experiment: {experiment_id}")
    print(f"{'='*80}")
    
    # Load data
    detailed_df, metrics_data, responses_data = load_data(experiment_id)
    
    if detailed_df is None or metrics_data is None or responses_data is None:
        logging.error(f"Skipping experiment {experiment_id} due to data loading errors")
        return
    
    # Check if LLM evaluation columns already exist in the detailed dataframe
    llm_eval_columns = [col for col in detailed_df.columns if col.startswith('llm_eval@')]
    if llm_eval_columns:
        print(f"LLM evaluation columns {llm_eval_columns} already exist in {experiment_id}. Skipping...")
        return
    
    # Note: We no longer check if metrics has LLM evaluation - we only care about the detailed dataframe
    
    # Store LLM evaluation results
    llm_eval_results = {}
    
    # Create a mapping from query to response data for easier lookup
    query_to_response = {item['query']: item for item in responses_data}
    
    # Process each QA pair with progress bar
    total_rows = len(detailed_df)
    processed_rows = 0
    skipped_rows = 0
    
    # Create progress bar
    pbar = tqdm(total=total_rows, desc=f"Evaluating QA pairs", unit="pair")
    
    for _, row in detailed_df.iterrows():
        qa_id = row['qa_id']
        query = row['query']
        ground_truth = row['ground_truth']
        
        # Find the corresponding response data by query
        if query not in query_to_response:
            logging.warning(f"No response data found for query: {query}")
            skipped_rows += 1
            pbar.update(1)
            continue
            
        response_item = query_to_response[query]
        context_objects = response_item.get('context', [])
        
        # Skip if no contexts found
        if not context_objects:
            logging.warning(f"No contexts found for query: {query}")
            skipped_rows += 1
            pbar.update(1)
            continue
        
        # Convert context objects to Langchain Documents
        documents = [create_document_from_context(ctx) for ctx in context_objects]
        
        # Evaluate using LLM
        results = evaluate_llm(
            qa_id=qa_id,
            query=query,
            ground_truth=ground_truth,
            contexts=documents,
            top_k_values=[1, 3, 5, 10]
        )
        
        if results:  # Only add results if evaluation was successful
            llm_eval_results[qa_id] = results
            
            # Add results to detailed DataFrame
            for k, result in results.items():
                column_name = f'llm_eval@{k}'
                detailed_df.loc[detailed_df['qa_id'] == qa_id, column_name] = result
            
            processed_rows += 1
        else:
            skipped_rows += 1
        
        pbar.update(1)
    
    pbar.close()
    
    print(f"Processed {processed_rows} QA pairs, skipped {skipped_rows} QA pairs")
    
    # Save detailed DataFrame
    base_path = Path("experiments")
    try:
        detailed_df.to_csv(base_path / "detailed_dfs" / f"{experiment_id}.csv", index=False)
        print(f"Updated detailed DataFrame saved for {experiment_id}")
    except Exception as e:
        logging.error(f"Error saving detailed DataFrame for {experiment_id}: {str(e)}")

    try:
        # Update metrics with LLM evaluation results
        updated_metrics = update_metrics_with_llm_eval(metrics_data, llm_eval_results)
    
        # Save updated metrics
        with open(base_path / "metrics" / f"{experiment_id}.json", 'w') as f:
            json.dump(updated_metrics, f, indent=2)
        print(f"Updated metrics saved for {experiment_id}")
    except Exception as e:
        logging.error(f"Error saving data for experiment {experiment_id}: {str(e)}")

def get_source_experiment_id(experiment_id: str) -> Optional[str]:
    """
    Get the source experiment ID for copying LLM evaluation results.
    
    Args:
        experiment_id (str): The experiment ID to check
        
    Returns:
        Optional[str]: Source experiment ID if there's a mapping, None otherwise
    """
    # Mapping rules as per requirements:
    # - simple_rag_02_01, simple_rag_02_02 are the same retrieval llm_eval as simple_rag_01_02
    # - simple_rag_02_03, simple_rag_02_04 are the same retrieval llm_eval as simple_rag_01_04
    mapping = {
        'simple_rag_02_01': 'simple_rag_01_02',
        'simple_rag_02_02': 'simple_rag_01_02',
        'simple_rag_02_03': 'simple_rag_01_04',
        'simple_rag_02_04': 'simple_rag_01_04'
    }
    
    # Extract the experiment name without the timestamp
    base_experiment_id = experiment_id.split('-')[0]
    
    return mapping.get(base_experiment_id)

def copy_llm_eval_from_source(target_experiment_id: str, source_experiment_id: str) -> bool:
    """
    Copy LLM evaluation results from source experiment to target experiment.
    
    Args:
        target_experiment_id (str): Target experiment ID
        source_experiment_id (str): Source experiment ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Copying LLM evaluation from {source_experiment_id} to {target_experiment_id}")
    
    base_path = Path("experiments")
    
    try:
        # Find the full source experiment ID with timestamp
        source_files = list(base_path.glob(f"detailed_dfs/{source_experiment_id}-*.csv"))
        if not source_files:
            print(f"Source experiment {source_experiment_id} not found")
            return False
            
        source_file = source_files[0]
        full_source_id = source_file.stem
        
        # Load source detailed DataFrame
        source_df = pd.read_csv(base_path / "detailed_dfs" / f"{full_source_id}.csv")
        
        # Check if source has LLM evaluation columns
        llm_eval_columns = [col for col in source_df.columns if col.startswith('llm_eval@')]
        if not llm_eval_columns:
            print(f"Source experiment {full_source_id} doesn't have LLM evaluation columns")
            return False
            
        # Load source metrics
        with open(base_path / "metrics" / f"{full_source_id}.json", 'r') as f:
            source_metrics = json.load(f)
            
        # Check if source metrics has LLM evaluation
        if ('retrieval' not in source_metrics or 
            source_metrics['retrieval'] is None or 
            'llm_eval' not in source_metrics['retrieval']):
            print(f"Source experiment {full_source_id} doesn't have LLM evaluation in metrics")
            return False
            
        # Load target detailed DataFrame
        target_df = pd.read_csv(base_path / "detailed_dfs" / f"{target_experiment_id}.csv")
        
        # Load target metrics
        with open(base_path / "metrics" / f"{target_experiment_id}.json", 'r') as f:
            target_metrics = json.load(f)
            
        # Copy LLM evaluation columns from source to target
        for col in llm_eval_columns:
            target_df[col] = source_df[col].values[:len(target_df)]
            
        # Ensure retrieval section exists in target metrics
        if 'retrieval' not in target_metrics:
            target_metrics['retrieval'] = {}
        elif target_metrics['retrieval'] is None:
            target_metrics['retrieval'] = {}
            
        # Copy LLM evaluation metrics from source to target
        target_metrics['retrieval']['llm_eval'] = source_metrics['retrieval']['llm_eval']
        
        # Save updated target detailed DataFrame
        target_df.to_csv(base_path / "detailed_dfs" / f"{target_experiment_id}.csv", index=False)
        
        # Save updated target metrics
        with open(base_path / "metrics" / f"{target_experiment_id}.json", 'w') as f:
            json.dump(target_metrics, f, indent=2)
            
        print(f"Successfully copied LLM evaluation from {full_source_id} to {target_experiment_id}")
        return True
        
    except Exception as e:
        logging.error(f"Error copying LLM evaluation: {str(e)}")
        return False

def main():
    """Process all experiments in the detailed_dfs directory."""
    experiments_dir = Path("experiments/detailed_dfs")
    
    # Get all experiment IDs (excluding hidden files) and sort them to maintain consistent order
    experiment_files = sorted(
        [f for f in experiments_dir.glob("*.csv") if not f.name.startswith('.')],
        key=lambda x: x.name
    )
    
    if not experiment_files:
        logging.error("No experiment files found")
        return
    
    total_experiments = len(experiment_files)
    print(f"Found {total_experiments} experiments to process")
    
    # Process each experiment with progress bar
    for i, experiment_file in enumerate(experiment_files):
        experiment_id = experiment_file.stem
        print(f"\nExperiment {i+1}/{total_experiments}: {experiment_id}")
        
        # Check if this experiment should use evaluation from another experiment
        source_experiment_id = get_source_experiment_id(experiment_id)
        
        if source_experiment_id:
            # Try to copy LLM evaluation from source experiment
            if copy_llm_eval_from_source(experiment_id, source_experiment_id):
                continue  # Skip to next experiment if copying was successful
            else:
                print(f"Failed to copy LLM evaluation, falling back to normal processing")
                
        # Process experiment normally
        process_experiment(experiment_id)

if __name__ == "__main__":
    main() 