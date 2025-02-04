# test/datasets_frames.py

"""
Frames dataset test
Run the script to debug;
```
python -m pdb test/datasets_frames.py
```
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.datasets.data.frames.processor import Frames

if __name__ == "__main__":
    frames = Frames()

    # Debugging: Load the dataset and print the first document and query
    frames.load('test')

    # Print the first document
    documents = list(frames.get_documents())
    if documents:
        print("First document:", documents[0])
    else:
        print("No documents found.")

    # Print the first query
    queries = list(frames.get_queries())
    if queries:
        print("First query:", queries[0])
    else:
        print("No queries found.")
