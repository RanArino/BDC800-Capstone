# test/eval_generation.py

"""
Generation evaluation test

- Run the script to debug;
    ```
    python -m pdb test/generation.py
    ```
- Input examples are stored in `test/input_data/qasper_list.json` and `test/input_data/RAGResponse_qasper_test.json`
- Output example is generated in `test/output_data/metrics_list.json`
"""

import sys
import os
import json
from typing import List, Dict
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.evaluation import calculate_generation_metrics

if __name__ == "__main__":
    # Load response list
    qa_file = "test/input_data/qasper_list.json"
    with open(qa_file, 'r') as f:
        qa_list: List[Dict] = json.load(f)

    response_file = "test/input_data/RAGResponse_qasper_test.json"
    with open(response_file, 'r') as f:
        # dictionary of RAGResponse objects
        response_list: List[Dict] = json.load(f) 

    # Calculate generation metrics
    metrics_list = []
    for response, qa in zip(response_list, qa_list):
        metrics = calculate_generation_metrics(
            qa.get('id'),
            qa.get('q'),
            response.get('llm_answer'), 
            qa.get('a'),
            rouge_types=['rouge1', 'rouge2', 'rougeL'],
            rouge_metric_types=['precision', 'recall', 'fmeasure']
        )
        metrics_list.append(metrics)

    # Save metrics to JSON file
    metrics_dir = Path("test/output_data")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_list_file = metrics_dir / "generation_metrics_list.json"
    with open(metrics_list_file, 'w') as f:
        json.dump([m.model_dump() for m in metrics_list], f, indent=2)
