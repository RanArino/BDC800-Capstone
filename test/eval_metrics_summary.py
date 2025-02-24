# test/eval_metrics_summary.py

"""
Metrics summary evaluation test

- Run the script to debug:
    ```
    python -m pdb test/eval_metrics_summary.py
    ```
- Input examples are stored in:
    - `test/input_data/qasper_list.json` (for IntraDocumentQA)
    - `test/input_data/RAGResponse_qasper_test.json` (for responses)
- Output example is generated in `test/output_data/metrics_summary.json`
"""

import sys
import os
import json
from typing import Literal
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.datasets.schema import IntraDocumentQA, InterDocumentQA
from core.frameworks.schema import RAGResponse
from core.evaluation.metrics_summary import calculate_metrics_for_qa

def test_script(mode: Literal["intra", "inter"] = "intra"):
    # Load QA list
    if mode == "intra":
        qa_file = "test/input_data/qasper_list.json"
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
            qa_list = [IntraDocumentQA(**qa) for qa in qa_data]

        response_file = "test/input_data/RAGResponse_qasper_test.json"

    elif mode == "inter":
        qa_file = "test/input_data/multihoprag_qa_list.json"
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
            qa_list = [InterDocumentQA(**qa) for qa in qa_data]

        response_file = "test/input_data/RAGResponse_multihoprag_test.json"
    
    with open(response_file, 'r') as f:
        response_data = json.load(f)
        response_list = [RAGResponse(**response) for response in response_data]

    # Calculate metrics for each QA pair
    metrics_list = []
    for qa, response in zip(qa_list, response_list):
        metrics = calculate_metrics_for_qa(
            qa=qa,
            response=response,
        )
        metrics_list.append(metrics)

    # Save metrics to JSON file
    output_dir = Path("test/output_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / f"{mode}_metrics_summary.json"
    with open(metrics_file, 'w') as f:
        json.dump([metrics.model_dump() for metrics in metrics_list], f, indent=2)

    print(f"Metrics saved to {metrics_file}")
    
if __name__ == "__main__":
    # evaluate inter-document QA
    test_script(mode="inter")

    # evaluate intra-document QA
    test_script(mode="intra")

    