"""
Test for the document summarizer functionality.
This script tests the run_doc_summary method by retrieving documents from the Frames dataset
and generating summaries for them.

To run this test:
```
python test/summarizer.py
```
"""

import sys
import os
import time
import tiktoken
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.rag_core.llm.summarizer import run_doc_summary
from core.datasets import get_dataset


def num_tokens_from_string(string: str, encoding_name: str = "o200k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def test_document_summarization():
    """Test document summarization with Frames dataset documents."""
    print("\n=== Testing Document Summarization ===")
    
    # Initialize Frames dataset
    print("Loading Frames dataset...")
    frames = get_dataset("frames")
    frames.load(mode="test")  # Use test mode to load a smaller dataset
    
    # Get a few documents to summarize
    print("Retrieving documents...")
    documents = list(frames.get_documents(num_docs=2, selection_mode="sequential"))
    
    if not documents:
        print("No documents found in the dataset. Exiting test.")
        return
    
    # Test summarization with different parameters
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1} ID: {doc.id}")
        print(f"Document Content Token Length: {num_tokens_from_string(doc.content)} tokens")
        
        # Generate summary with default parameters
        print("\nGenerating summary with default parameters...")
        start = time.time()
        summary = run_doc_summary(doc.content, temperature=1.0)
        print("Complete summary in ", time.time() - start)
        print(f"Summary: {summary}")
        print(f"Summary Token Length: {num_tokens_from_string(summary)} tokens")
        
        # Generate summary with custom parameters
        print("\nGenerating summary with custom parameters (shorter, more creative)...")
        start = time.time()
        short_summary = run_doc_summary(doc.content)
        print("Complete summary in ", time.time() - start)
        print(f"Summary: {summary}")
        print(f"Summary Length: {num_tokens_from_string(short_summary)} tokens")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_document_summarization()
    print("\n=== Summarization Tests Completed ===")
