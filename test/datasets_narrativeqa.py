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

    # Load the dataset and store in local file
    narrativeqa.load()

    # Print the first document
    print("\n=== Testing with get_documents(num_docs=2, selection_mode='sequential') ===")
    documents_1 = list(narrativeqa.get_documents(num_docs=2, selection_mode="sequential"))
    if documents_1:
        for i, doc in enumerate(documents_1):
            print(f"Document {i+1}:")
            print(f"Document ID: {doc.id}")
            print(f"Document Content: {doc.content[:20]}")
    else:
        print("No documents found.")

    # Print random document
    print("\n=== Testing with get_documents(num_docs=2, selection_mode='random') ===")
    documents_2 = list(narrativeqa.get_documents(num_docs=2, selection_mode="random"))
    if documents_2:
        for i, doc in enumerate(documents_2):
            print(f"Document {i+1}:")
            print(f"Document ID: {doc.id}")
            print(f"Document Content: {doc.content[:20]}")
    else:
        print("No documents found.")

    # Print first two queries
    print("\n=== Testing with get_queries(num_qas=2, selection_mode='sequential') ===")
    queries_1 = list(narrativeqa.get_queries(num_qas=2, selection_mode="sequential"))
    if queries_1:
        for i, qa in enumerate(queries_1):
            print(f"Query {i+1}:")
            print(f"Query ID: {qa.id}")
            print(f"Query Question: {qa.q}")
            print(f"Document ID: {qa.document_id}")  # NarrativeQA has single doc per query
    else:
        print("No queries found.")

    # Print randomly selected two queries
    print("\n=== Testing with get_queries(num_qas=2, selection_mode='random') ===")
    queries_2 = list(narrativeqa.get_queries(num_qas=2, selection_mode="random"))
    if queries_2:
        for i, qa in enumerate(queries_2):
            print(f"Query {i+1}:")
            print(f"Query ID: {qa.id}")
            print(f"Query Question: {qa.q}")
            print(f"Document ID: {qa.document_id}")  # NarrativeQA has single doc per query
    else:
        print("No queries found.")

    # Print documents based on queries_2 document_ids
    print("\n=== Testing with get_documents(doc_ids=queries_2.document_id) ===")
    documents_3 = list(narrativeqa.get_documents(doc_ids=[qa.document_id for qa in queries_2]))
    if documents_3:
        for i, doc in enumerate(documents_3):
            print(f"Document {i+1}:")
            print(f"Document ID: {doc.id}")
            print(f"Document Content: {doc.content[:20]}")
    else:
        print("No documents found.")

    # Print queries based on documents_3 ids
    print("\n=== Testing with get_queries(doc_ids=documents_3.id) ===")
    queries_3 = list(narrativeqa.get_queries(doc_ids=[doc.id for doc in documents_3]))
    if queries_3:
        for i, qa in enumerate(queries_3):
            print(f"Query {i+1}:")
            print(f"Query ID: {qa.id}")
            print(f"Query Question: {qa.q}")
            print(f"Document ID: {qa.document_id}")  # NarrativeQA has single doc per query
            if i == 4:
                print(f"Query Evidence: {qa.e}")
    else:
        print("No queries found.")

    print("\n=== Completed testing ===")