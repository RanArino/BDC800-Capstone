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

    # Load the dataset and store in local file
    multihoprag.load()

    # # Print the first document
    # print("\n=== Testing with get_documents(num_docs=2, selection_mode='sequential') ===")
    # documents_1 = list(multihoprag.get_documents(num_docs=2, selection_mode="sequential"))
    # if documents_1:
    #     for i, doc in enumerate(documents_1):
    #         print(f"Document {i+1}:")
    #         print(f"Document ID: {doc.id}")
    #         print(f"Document Content: {doc.content[:20]}")
    # else:
    #     print("No documents found.")

    # # Print random document
    # print("\n=== Testing with get_documents(num_docs=2, selection_mode='random') ===")
    # documents_2 = list(multihoprag.get_documents(num_docs=2, selection_mode="random"))
    # if documents_2:
    #     for i, doc in enumerate(documents_2):
    #         print(f"Document {i+1}:")
    #         print(f"Document ID: {doc.id}")
    #         print(f"Document Content: {doc.content[:20]}")
    # else:
    #     print("No documents found.")

    # # Print first two queries
    # print("\n=== Testing with get_queries(num_qas=2, selection_mode='sequential') ===")
    # queries_1 = list(multihoprag.get_queries(num_qas=2, selection_mode="sequential"))
    # if queries_1:
    #     for i, qa in enumerate(queries_1):
    #         print(f"Query {i+1}:")
    #         print(f"Query ID: {qa.id}")
    #         print(f"Query Question: {qa.q}")
    #         print(f"Document IDs: {qa.document_ids}")  # MultiHopRAG has multiple docs per query
    # else:
    #     print("No queries found.")

    # Print randomly selected two queries
    print("\n=== Testing with get_queries(num_qas=1, selection_mode='random') ===")
    queries_2 = list(multihoprag.get_queries(num_qas=1, selection_mode="random"))
    if queries_2:
        for i, qa in enumerate(queries_2):
            print(f"Query {i+1}:")
            print(f"Query ID: {qa.id}")
            print(f"Query Question: {qa.q}")
            print(f"Document IDs: {qa.document_ids}")  # MultiHopRAG has multiple docs per query
    else:
        print("No queries found.")

    # Print documents based on queries_2 document_ids
    print("\n=== Testing with get_documents(doc_ids=queries_2.document_ids) ===")
    # Flatten the list of document IDs from all queries
    doc_ids = [doc_id for qa in queries_2 for doc_id in qa.document_ids]
    documents_3 = list(multihoprag.get_documents(doc_ids=doc_ids))
    if documents_3:
        for i, doc in enumerate(documents_3):
            print(f"Document {i+1}:")
            print(f"Document ID: {doc.id}")
            print(f"Document Content: {doc.content[:20]}")
            if i == 4:
                break

    # Print queries based on documents_3 ids
    print("\n=== Testing with get_queries(doc_ids=documents_3.id) ===")
    queries_3 = list(multihoprag.get_queries(doc_ids=[doc.id for doc in documents_3]))
    if queries_3:
        for i, qa in enumerate(queries_3):
            print(f"Query {i+1}:")
            print(f"Query ID: {qa.id}")
            print(f"Query Question: {qa.q}")
            print(f"Document IDs: {qa.document_ids}")  # MultiHopRAG has multiple docs per query
            if i == 4:
                break
    else:
        print("No queries found.")
    
    print("\n=== Completed testing ===")
