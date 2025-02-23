# core/datasets/__init__.py
"""
Dataset implementations for RAG framework.
"""

from core.datasets.base import BaseDataset
from core.datasets.schema import Document, Metadata, BaseQA, IntraDocumentQA, InterDocumentQA, WikipediaContent
from core.datasets.data.narrativeqa.processor import NarrativeQA
from core.datasets.data.qasper.processor import Qasper
from core.datasets.data.multihoprag.processor import MultiHopRAG
from core.datasets.data.frames.processor import Frames

__all__ = [
    'BaseDataset',
    'Document',
    'Metadata',
    'BaseQA',
    'IntraDocumentQA',
    'InterDocumentQA',
    'WikipediaContent',
    'NarrativeQA',
    'Qasper',
    'MultiHopRAG',
    'Frames',
    'get_dataset'
]

def get_dataset(name: str) -> BaseDataset:
    """
    Retrieves a dataset instance by name.

    Args:
        name: The name of the dataset.
        **kwargs: Additional keyword arguments to pass to the dataset constructor (e.g., test_mode).

    Returns:
        An instance of the requested dataset.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if name == "narrativeqa":
        return NarrativeQA()
    elif name == "qasper":
        return Qasper()
    elif name == "multihoprag":
        return MultiHopRAG()
    elif name == "frames":
        return Frames()
    else:
        raise ValueError(f"Unknown dataset: {name}") 