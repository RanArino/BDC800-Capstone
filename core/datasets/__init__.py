# core/datasets/__init__.py
"""
Dataset implementations for RAG framework.
"""

from core.datasets.base import BaseDataset
from core.datasets.schema import Dataset, Document, Metadata, BaseQA, IntraDocumentQA, InterDocumentQA, WikipediaContent
from core.datasets.data.narrativeqa.processor import NarrativeQA
from core.datasets.data.qasper.processor import Qasper
from core.datasets.data.multihoprag.processor import MultiHopRAG
from core.datasets.data.frames.processor import Frames

__all__ = [
    'BaseDataset',
    'Dataset',
    'Document',
    'Metadata',
    'BaseQA',
    'IntraDocumentQA',
    'InterDocumentQA',
    'WikipediaContent',
    'NarrativeQA',
    'Qasper',
    'MultiHopRAG',
    'Frames'
] 