# test/datasets_multihoprag.py

"""
MultiHopRAG dataset test
Run the script to debug;
```
python -m pdb test/datasets_multihoprag.py
```
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.datasets.data.multihoprag.processor import MultiHopRAG

if __name__ == "__main__":
    multihoprag = MultiHopRAG()

    # Debugging: Load the dataset and print the first document and query
    multihoprag.load('test')

    # Print the first document
    documents = list(multihoprag.get_documents())
    if documents:
        print("First document:", documents[0])
    else:
        print("No documents found.")

    # Print the first query
    queries = list(multihoprag.get_queries())
    if queries:
        print("First query:", queries[0])
    else:
        print("No queries found.")
