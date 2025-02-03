# core/datasets/schema.py

"""
Common schema definitions for all datasets.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    """Metadata for a document. Fields may vary by dataset source."""
    url: Optional[str] = None
    title: Optional[str] = None
    # Allow additional fields
    extra_fields: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    """A single document in the knowledge base."""
    id: str
    content: str
    metadata: Metadata

class BaseQA(BaseModel):
    """Base class for question-answer pairs."""
    q: str
    a: str

class IntraDocumentQA(BaseQA):
    """QA pair that references a single source document (e.g., NarrativeQA, Qasper)."""
    document_id: str

class InterDocumentQA(BaseQA):
    """QA pair that may reference multiple source documents (e.g., Multihop-RAG, FRAMES)."""
    document_ids: List[str]

class Dataset(BaseModel):
    """The complete dataset containing documents and QA pairs."""
    documents: List[Document]
    intra_qas: List[IntraDocumentQA] = []  # For single-document QA pairs
    inter_qas: List[InterDocumentQA] = []  # For multi-document QA pairs

    def validate_document(self, document: Document) -> bool:
        """
        Validate a document against the schema.
        
        Args:
            document: Document to validate
            
        Returns:
            bool: True if valid
        """
        try:
            Document.model_validate(document)
            return True
        except:
            return False
    
    def validate_qa(self, qa: BaseQA) -> bool:
        """
        Validate a QA pair against the schema.
        
        Args:
            qa: QA pair to validate
            
        Returns:
            bool: True if valid
        """
        try:
            if isinstance(qa, IntraDocumentQA):
                IntraDocumentQA.model_validate(qa)
            elif isinstance(qa, InterDocumentQA):
                InterDocumentQA.model_validate(qa)
            else:
                return False
            return True
        except:
            return False