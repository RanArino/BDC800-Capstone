# core/datasets/data/narrativeqa/processor.py

"""
NarrativeQA dataset processor
"""
from tqdm import tqdm

from core.utils import load_hf_dataset
from core.datasets import (
    BaseDataset, 
    Document, 
    Metadata, 
    IntraDocumentQA, 
    Dataset
)

def extract_text(text: str, start_marker: str, end_marker: str) -> str:
    """Extract text between start and end markers.
    
    Args:
        text: The full text content
        start_marker: The starting marker to find
        end_marker: The ending marker to find
        
    Returns:
        The extracted text between markers. Returns full text if markers not found.
    """
    try:
        # Find the first occurrence of start marker
        start_pos = text.find(start_marker)
        if start_pos == -1:
            return text
            
        # Find the last occurrence of end marker
        end_pos = text.rfind(end_marker)
        if end_pos == -1:
            return text
            
        # Include end marker in the extraction
        end_pos += len(end_marker)
            
        # Validate positions
        if start_pos >= end_pos:
            return text
            
        return text[start_pos:end_pos].strip()
    except Exception:
        # Fallback to full text if any error occurs
        return text

class NarrativeQA(BaseDataset):
    def __init__(self):
        super().__init__("narrativeqa")
    
    def _process_raw_data(self) -> Dataset:
        """Process raw data from HuggingFace into our schema format"""
        # "split" can be "train"(14.7k), "validation"(3.46k), or "test"(10.6k)
        raw_data = load_hf_dataset("deepmind/narrativeqa", split="train")
        total_items = len(raw_data)
        
        # If in test mode, only process first 5 items
        if self._test_mode:
            total_items = min(5, total_items)
            raw_data = raw_data.select(range(total_items))
            print(f"\nTest Mode: Processing first {total_items} items from NarrativeQA dataset...")
        else:
            print(f"\nProcessing {total_items} items from NarrativeQA dataset...")
        
        # Track processed document IDs for deduplication
        processed_docs = set()
        documents = []
        qas = []
        qa_counter = 0  # Counter for generating QA IDs
        
        for item in tqdm(raw_data, total=total_items, desc="Processing documents", unit="item"):
            doc_id = str(item['document']['id'])
            
            # Process document only if not seen before
            if doc_id not in processed_docs:
                processed_docs.add(doc_id)
                
                # Extract text between markers
                text = item['document']['text']
                start_marker = item['document']['start']
                end_marker = item['document']['end']
                extracted_text = extract_text(text, start_marker, end_marker)
                
                doc = Document(
                    id=doc_id,
                    content=extracted_text,
                    metadata=Metadata(
                        url=item['document']['url'],
                        summary=item['document']['summary']['text'],
                        source=item['document']['kind']
                    )
                )
                documents.append(doc)
            
            # Store QA pairs with unique ID
            answers = [ans['text'] for ans in item['answers']]
            qa = IntraDocumentQA(
                id=f"q{qa_counter}",  # Add unique ID
                q=item['question']['text'],
                a=' '.join(answers),
                document_id=doc_id
            )
            qa_counter += 1
            qas.append(qa)
            
        return Dataset(documents=documents, intra_qas=qas) 