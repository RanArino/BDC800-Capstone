# core/utils/libs/custom_faiss.py
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Iterable
import uuid
import warnings
import torch

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# Suppress semaphore leak warnings
warnings.filterwarnings("ignore", message=".*resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown.*")

# Limit number of threads to prevent threading issues
torch.set_num_threads(1)

class FAISSIVFCustom(FAISS):
    """Custom FAISS implementation with IVF configuration based on dataset size."""
    
    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[Iterable[dict]] = None,
        ids: Optional[List[str]] = None,
        nlist: Optional[int] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Create FAISS instance with custom IVF configuration based on dataset size"""
        
        # Extract texts and embeddings
        texts = []
        embeddings = []
        for text, embedding_vector in text_embeddings:
            texts.append(text)
            embeddings.append(embedding_vector)
        
        # Get embedding dimension
        dim = len(embeddings[0]) if embeddings else 0
        
        # Create a FAISS index - using conservative settings for memory usage
        if len(embeddings) > 10000 and dim > 0:  # Only use IVF for larger collections
            if nlist is None:
                # Use a more conservative nlist value to reduce memory usage
                nlist = min(1024, max(4, int(4 * np.sqrt(len(embeddings)))))
                
            # Create quantizer for IVF
            quantizer = faiss.IndexFlatL2(dim)
            # Create the IVF index
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            # Train the index (required for IVF)
            if not index.is_trained and len(embeddings) > nlist:
                # Convert to numpy array once to avoid multiple conversions
                embeddings_array = np.array(embeddings, dtype=np.float32)
                index.train(embeddings_array)
                # Add embeddings to the index
                if len(embeddings) > 0:
                    index.add(embeddings_array)
        else:
            # For smaller collections, use a flat index
            index = faiss.IndexFlatL2(dim)
            # Add embeddings to the index
            if len(embeddings) > 0:
                index.add(np.array(embeddings, dtype=np.float32))
        
        # Create the docstore and mapping
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}
        
        # Process texts, embeddings, and metadata
        metadatas = metadatas or [{} for _ in texts]
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
            docstore.add({doc_id: Document(page_content=text, metadata=metadata)})
            index_to_docstore_id[i] = doc_id
        
        # Return the FAISS instance
        return cls(
            embedding,
            index,
            docstore,
            index_to_docstore_id,
            **kwargs
        )
        
    def __del__(self):
        """Ensure proper resource cleanup when object is deleted."""
        try:
            if hasattr(self, 'index') and self.index is not None:
                # Reset the FAISS index to release resources
                self.index.reset()
        except:
            pass