# test/chunker.py

"""
Chunker test
Run the script to debug;
```
python -m pdb test/chunker.py
```
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.datasets import Frames
from core.rag_core import Chunker

if __name__ == "__main__":
    # Initialize chunker
    chunk_size = 100
    chunk_overlap = 0.2
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Load frames dataset
    frames = Frames()
    frames.load('test')
    documents = list(frames.get_documents())

    # Chunk documents
    # Fixed-size chunks
    # chunks = chunker.run(documents, mode='fixed')

    # Sentence-aware chunks
    chunks = chunker.run(documents, mode='sentence')


    # Print the first 3 chunks
    for i in range(3):
        print(f"Chunk {i+1}:")
        print(chunks[i])
        print("\n")

    