# core/datasets/data/qasper/processor.py

"""
Qasper dataset processor
"""

from typing import List
from tqdm import tqdm

from core.utils import load_hf_dataset
from core.datasets import (
    BaseDataset,
    Document, 
    Metadata, 
    IntraDocumentQA, 
    Dataset
)

def flatten_and_join(lst: List[List[str]]) -> str:
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_and_join(item))
        else:
            flattened.append(item)
    return flattened

class Qasper(BaseDataset):
    def __init__(self):
        super().__init__("qasper")
    
    def _process_raw_data(self):
        """Process raw data from HuggingFace into our schema format"""
        raw_data = load_hf_dataset("allenai/qasper", split="train")
        total_items = len(raw_data)
        
        # If in test mode, only process first 5 items
        if self._test_mode:
            total_items = min(5, total_items)
            raw_data = raw_data.select(range(total_items))
            print(f"\nTest Mode: Processing first {total_items} items from QASPER dataset...")
        else:
            print(f"\nProcessing {total_items} items from QASPER dataset...")
        
        # Track processed document IDs for deduplication
        processed_docs = set()
        documents = []
        qas = []
        
        for item in tqdm(raw_data, total=total_items, desc="Processing documents", unit="item"):
            doc_id = str(item['id'])

            # Process document only if not seen before
            if doc_id not in processed_docs:
                processed_docs.add(doc_id)
                
                # Process document content - flatten paragraphs if needed
                content = item['full_text']['paragraphs']
                flat_content = ' '.join(flatten_and_join(content))
                
                # Create metadata
                metadata = Metadata(
                    title=item['title'],
                    abstract=item['abstract'],
                )
                
                # Create document
                doc = Document(
                    id=doc_id,
                    content=flat_content,
                    metadata=metadata
                )
                documents.append(doc)
                
                # Process QA pairs
                for question, answers in zip(item['qas']['question'], item['qas']['answers']):
                    # Find a valid answer from the answer list
                    valid_answer = None
                    valid_evidence = None

                    # Process each answer
                    for ans in answers['answer']:
                        # Skip if marked as unanswerable
                        if ans['unanswerable']:
                            continue
                                
                        # Skip if free_form_answer is empty
                        if not ans['free_form_answer']:
                            continue
                                
                        # We found a valid answer
                        valid_answer = ans['free_form_answer']
                        
                        # Process evidence for this valid answer
                        if ans.get('evidence'):
                            evidence = []
                            for ev in ans['evidence']:
                                if isinstance(ev, list):
                                    evidence.extend(ev)
                                else:
                                    evidence.append(ev)
                            valid_evidence = ' '.join(evidence) if evidence else None
                        
                        # Break once we find the first valid answer
                        break
                    
                    # Skip if no valid answer was found
                    if valid_answer is None:
                        continue
                        
                    # Create QA pair
                    qa = IntraDocumentQA(
                        q=question,
                        a=valid_answer,
                        e=valid_evidence,
                        document_id=doc_id
                    )
                    qas.append(qa)
        
        # Create and return dataset
        return Dataset(
            documents=documents,
            intra_qas=qas
        )