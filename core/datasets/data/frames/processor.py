# core/datasets/data/frames/processor.py

"""
Frames dataset processor
"""

import requests
from bs4 import BeautifulSoup
import re
import hashlib
import ast
from typing import Dict, List, Union
from tqdm import tqdm

from core.utils import load_hf_dataset
from core.datasets import (
    BaseDataset, 
    Document, 
    Metadata, 
    InterDocumentQA, 
    Dataset, 
    WikipediaContent
)

def _parse_wiki_links(wiki_links: Union[str, List[str]]) -> List[str]:
    """Parse wiki links that might come as string or list.
    
    Args:
        wiki_links: Either a string representation of a list or an actual list of URLs
        
    Returns:
        List of wiki link strings
    """
    if isinstance(wiki_links, str):
        try:
            # Handle string representation of list using ast.literal_eval
            return ast.literal_eval(wiki_links)
        except (ValueError, SyntaxError):
            # If parsing fails, return empty list
            return []
    return wiki_links if isinstance(wiki_links, list) else []

def _extract_wikipedia_content(url: str) -> WikipediaContent:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_element = soup.find(id="firstHeading")
        if not title_element:
            return WikipediaContent(error="Could not find title element on page")
        title = title_element.text
        
        content_div = soup.find(id="mw-content-text")
        if not content_div:
            return WikipediaContent(error="Could not find content element on page")
            
        paragraphs = content_div.find_all('p')
        if not paragraphs:
            return WikipediaContent(error="No paragraphs found in content")
            
        text_content = "\n".join(re.sub(r'\[.*?\]', '', p.get_text()) for p in paragraphs).strip()
        text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
        
        if not text_content:
            return WikipediaContent(error="Extracted content is empty")
            
        return WikipediaContent(title=title, content=text_content, error=None)
    except requests.RequestException as e:
        return WikipediaContent(error=f"Request failed: {str(e)}")
    except Exception as e:
        return WikipediaContent(error=f"Unexpected error: {str(e)}")


class Frames(BaseDataset):
    def __init__(self):
        super().__init__("frames")
    
    def _process_raw_data(self):
        """Process raw data from HuggingFace into our schema format"""
        raw_data = load_hf_dataset("google/frames-benchmark", split="test")
        total_items = len(raw_data)
        
        # If in test mode, only process first 5 items
        if self._test_mode:
            total_items = min(5, total_items)
            raw_data = raw_data.select(range(total_items))
            print(f"\nTest Mode: Processing first {total_items} items from FRAMES dataset...")
        else:
            print(f"\nProcessing {total_items} items from FRAMES dataset...")
        
        # Keep track of URL to document ID mapping
        url_to_id: Dict[str, str] = {}
        documents = []
        inter_qas = []
        qa_counter = 0  # Counter for generating QA IDs
        
        # Process documents with progress bar
        for item in tqdm(raw_data, total=total_items, desc="Processing documents", unit="item"):
            # Get list of wiki links and parse if needed
            wiki_links = _parse_wiki_links(item['wiki_links'])
            doc_ids = []
            
            # Process each wiki link
            for url in wiki_links:
                if not url:  # Skip empty URLs
                    continue
                    
                # Check if we already have this URL processed
                if url in url_to_id:
                    doc_ids.append(url_to_id[url])
                    continue
                
                # Extract Wikipedia content first
                wiki_content = _extract_wikipedia_content(url)
                if wiki_content.error is None:  # Only process successful extractions
                    # Generate ID only for successful content retrieval
                    doc_id = hashlib.md5(url.encode()).hexdigest()
                    url_to_id[url] = doc_id
                    doc_ids.append(doc_id)
                    
                    # Create metadata
                    metadata = Metadata(
                        url=url,
                        title=wiki_content.title,
                        source="wikipedia"
                    )
                    
                    # Create document
                    doc = Document(
                        id=doc_id,
                        content=wiki_content.content,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            # Create QA pair if we have valid documents
            if doc_ids:
                qa = InterDocumentQA(
                    id=f"q{qa_counter}",  # Add unique ID
                    q=item['Prompt'],
                    a=item['Answer'],
                    e=None,  # No explicit evidence provided in FRAMES
                    document_ids=doc_ids
                )
                qa_counter += 1
                inter_qas.append(qa)
        
        # Create and return dataset
        dataset = Dataset(
            documents=documents,
            inter_qas=inter_qas
        )
        
        return dataset