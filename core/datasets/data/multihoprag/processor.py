# core/datasets/data/multihoprag/processor.py

"""
MultiHopRAG dataset processor
"""

import hashlib
from typing import Dict, Tuple, List
from tqdm import tqdm

from core.utils import load_hf_dataset
from core.datasets import (
    BaseDataset, 
    Document, 
    Metadata, 
    InterDocumentQA,
)

class MultiHopRAG(BaseDataset):
    def __init__(self):
        super().__init__("multihoprag")
    
    def _process_raw_data(self) -> Tuple[List[Document], List[InterDocumentQA]]:
        """Process raw data from HuggingFace into our schema format"""
        raw_data_corpus = load_hf_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
        raw_data_qas = load_hf_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
        
        total_corpus = len(raw_data_corpus)
        total_qas = len(raw_data_qas)
        
        # If in test mode, only process first 5 items of each
        if self._test_mode:
            total_corpus = min(5, total_corpus)
            total_qas = min(5, total_qas)
            raw_data_corpus = raw_data_corpus.select(range(total_corpus))
            raw_data_qas = raw_data_qas.select(range(total_qas))
            print(f"\nTest Mode: Processing first {total_corpus} documents and {total_qas} QAs from MultiHopRAG dataset...")
        else:
            print(f"\nProcessing {total_corpus} documents and {total_qas} QAs from MultiHopRAG dataset...")
        
        # Keep track of URL to document ID mapping
        url_to_id: Dict[str, str] = {}
        
        # Process corpus with progress bar
        documents = []
        for item in tqdm(raw_data_corpus, total=total_corpus, desc="Processing documents", unit="doc"):
            # Generate unique ID from URL using hash if not already exists
            if item['url'] not in url_to_id:
                doc_id = hashlib.md5(item['url'].encode()).hexdigest()
                url_to_id[item['url']] = doc_id
            
            # Create metadata
            metadata = Metadata(
                url=item['url'],
                title=item['title'],
                category=item['category'],
                source=item['source']
            )
            
            # Create document
            doc = Document(
                id=url_to_id[item['url']],
                content=item['body'],
                metadata=metadata
            )
            documents.append(doc)
            
        # Process QA pairs with progress bar
        inter_qas = []
        qa_counter = 0  # Counter for generating QA IDs
        
        for item in tqdm(raw_data_qas, total=total_qas, desc="Processing QA pairs", unit="qa"):
            # Get document IDs from evidence list
            doc_ids = []
            for evidence in item['evidence_list']:
                url = evidence['url']
                # Generate ID if URL not seen before
                if url not in url_to_id:
                    doc_id = hashlib.md5(url.encode()).hexdigest()
                    url_to_id[url] = doc_id
                doc_ids.append(url_to_id[url])
            
            # Create QA pair with unique ID
            qa = InterDocumentQA(
                id=f"q{qa_counter}",  # Add unique ID
                q=item['query'],
                a=item['answer'],
                document_ids=doc_ids
            )
            qa_counter += 1
            inter_qas.append(qa)
        
        return documents, inter_qas

