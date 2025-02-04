# test/dataset_qasper.py

"""
Qasper dataset test
Run the script to debug;
```
python -m pdb test/datasets_qasper.py
```
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.datasets.data.qasper.processor import Qasper

if __name__ == "__main__":
    qasper = Qasper()

    # Debugging: Load the dataset and print the first document and query
    qasper.load('test')

    # Print the first document
    documents = list(qasper.get_documents())
    if documents:
        print("First document:", documents[0])
    else:
        print("No documents found.")

    # Print the first query
    queries = list(qasper.get_queries())
    if queries:
        print("First query:", queries[0])
    else:
        print("No queries found.")
