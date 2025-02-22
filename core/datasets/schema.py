# core/datasets/schema.py

"""
Common schema definitions for all datasets.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class WikipediaContent(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    error: Optional[str] = None

class Metadata(BaseModel):
    """Metadata for a document. Fields may vary by dataset source."""
    url: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None

class Document(BaseModel):
    """A single document in the knowledge base."""
    id: str
    content: str
    metadata: Metadata

class BaseQA(BaseModel):
    """Base class for question-answer pairs."""
    id: str
    q: str # question
    a: str # answer
    e: Optional[str] = None # evidence

class IntraDocumentQA(BaseQA):
    """QA pair that references a single source document (e.g., NarrativeQA, Qasper)."""
    document_id: str

class InterDocumentQA(BaseQA):
    """QA pair that may reference multiple source documents (e.g., Multihop-RAG, FRAMES)."""
    document_ids: List[str]
