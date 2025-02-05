# test/datasets_narrativeqa.py

"""
NarrativeQA dataset test
Run the script to debug;
```
python -m pdb test/datasets_narrativeqa.py
```
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.datasets.data.narrativeqa.processor import NarrativeQA


if __name__ == "__main__":
    narrativeqa = NarrativeQA()

    # Debugging: Load the dataset and print the first document and query
    narrativeqa.load('test')

    # Print the first document
    documents = list(narrativeqa.get_documents())
    if documents:
        print("First document:", documents[0])
    else:
        print("No documents found.")

    # Print the first query
    queries = list(narrativeqa.get_queries())
    if queries:
        print("First query:", queries[0])
    else:
        print("No queries found.")