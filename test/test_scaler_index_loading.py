"""Test the SCALER RAG index loading and saving functionality.

This test suite verifies:
1. Full index saving and loading
2. Partial index loading (when only some layers exist)
3. Incremental index building
"""

import os
import shutil
import pytest
from pathlib import Path
from typing import List, Dict

from core.frameworks import ScalerRAG
from core.datasets import Document as SchemaDocument

class TestDocument(SchemaDocument):
    """Test document class for creating mock documents"""
    def __init__(self, id: str, content: str):
        super().__init__(
            id=id,
            content=content,
            metadata={"source": "test"}
        )

def create_test_docs() -> List[TestDocument]:
    """Create test documents"""
    return [
        TestDocument(
            id="doc1",
            content="This is a test document about machine learning. It contains information about neural networks and deep learning."
        ),
        TestDocument(
            id="doc2",
            content="This document discusses natural language processing and transformers. It explains how BERT and GPT models work."
        ),
        TestDocument(
            id="doc3",
            content="This document covers reinforcement learning and genetic algorithms. It explains concepts like Q-learning and evolutionary strategies."
        )
    ]

@pytest.fixture
def test_config_path(tmp_path) -> str:
    """Create a temporary config file for testing"""
    config_content = """
TEST_scaler_rag:
  dataset:
    name: "test_dataset"
    number_of_docs: 3
    number_of_qas: 10
    selection_mode: "sequential"
  chunker:
    mode: "fixed"
    size: 100
    overlap: 0.2
    embedding_id: "huggingface-multi-qa-mpnet"
    clustering:
      method: "kmeans"
      n_clusters: 1
    dim_reduction:
      method: "pca"
      n_components: 50
  retrieval_generation:
    faiss_search: "flatl2"
    top_k: 3
    llm_id: "llama3.1"
    """
    config_file = tmp_path / "test_scaler_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)

@pytest.fixture
def scaler_rag(test_config_path):
    """Create a ScalerRAG instance with test configuration"""
    return ScalerRAG("TEST_scaler_rag", config_path=test_config_path, is_save_vectorstore=True)

def test_full_index_save_load(scaler_rag, tmp_path):
    """Test saving and loading of complete index"""
    # Index test documents
    test_docs = create_test_docs()
    scaler_rag.index(test_docs)
    
    # Verify all layers were created
    assert scaler_rag.layered_vector_stores["doc"] is not None
    assert scaler_rag.layered_vector_stores["doc_cc"] is not None
    assert len(scaler_rag.layered_vector_stores["chunk"]) > 0
    assert len(scaler_rag.layered_vector_stores["chunk_cc"]) > 0
    
    # Create new instance and load indexes
    new_rag = ScalerRAG("TEST_scaler_rag", config_path=scaler_rag.config_path, is_save_vectorstore=True)
    loaded_layers = new_rag._load_all_indexes()
    
    # Verify all layers were loaded
    assert all(loaded_layers.values())
    assert new_rag.layered_vector_stores["doc"] is not None
    assert new_rag.layered_vector_stores["doc_cc"] is not None
    assert len(new_rag.layered_vector_stores["chunk"]) > 0
    assert len(new_rag.layered_vector_stores["chunk_cc"]) > 0

def test_partial_index_loading(scaler_rag, tmp_path):
    """Test loading when only some layers exist"""
    # Index test documents
    test_docs = create_test_docs()
    scaler_rag.index(test_docs)
    
    # Remove chunk and chunk_cc directories to simulate partial index
    chunk_path = Path(scaler_rag.vectorstore_path) / "chunk"
    chunk_cc_path = Path(scaler_rag.vectorstore_path) / "chunk_cc"
    shutil.rmtree(chunk_path)
    shutil.rmtree(chunk_cc_path)
    
    # Create new instance and load indexes
    new_rag = ScalerRAG("TEST_scaler_rag", config_path=scaler_rag.config_path, is_save_vectorstore=True)
    loaded_layers = new_rag._load_all_indexes()
    
    # Verify only doc and doc_cc were loaded
    assert loaded_layers["doc"]
    assert loaded_layers["doc_cc"]
    assert not loaded_layers["chunk"]
    assert not loaded_layers["chunk_cc"]
    
    # Try indexing the same documents again
    new_rag.index(test_docs)
    
    # Verify all layers are now present
    assert new_rag.layered_vector_stores["doc"] is not None
    assert new_rag.layered_vector_stores["doc_cc"] is not None
    assert len(new_rag.layered_vector_stores["chunk"]) > 0
    assert len(new_rag.layered_vector_stores["chunk_cc"]) > 0

def test_no_existing_index(scaler_rag):
    """Test behavior when no index exists"""
    # Try loading non-existent index
    loaded_layers = scaler_rag._load_all_indexes()
    
    # Verify no layers were loaded
    assert not any(loaded_layers.values())
    
    # Verify vector stores are properly initialized
    assert isinstance(scaler_rag.layered_vector_stores["doc"], dict)
    assert scaler_rag.layered_vector_stores["doc_cc"] is None
    assert isinstance(scaler_rag.layered_vector_stores["chunk"], dict)
    assert isinstance(scaler_rag.layered_vector_stores["chunk_cc"], dict)
    assert len(scaler_rag.layered_vector_stores["doc"]) == 0
    assert len(scaler_rag.layered_vector_stores["chunk"]) == 0
    assert len(scaler_rag.layered_vector_stores["chunk_cc"]) == 0

def test_corrupted_index(scaler_rag, tmp_path):
    """Test handling of corrupted index files"""
    # Index test documents
    test_docs = create_test_docs()
    scaler_rag.index(test_docs)
    
    # Corrupt one of the index files
    doc_index_path = Path(scaler_rag.vectorstore_path) / "doc" / "doc.faiss"
    with open(doc_index_path, 'w') as f:
        f.write("corrupted data")
    
    # Create new instance and try loading indexes
    new_rag = ScalerRAG("TEST_scaler_rag", config_path=scaler_rag.config_path, is_save_vectorstore=True)
    
    with pytest.raises(Exception):
        new_rag._load_all_indexes()

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test artifacts after each test"""
    yield
    # Clean up vectorstore directory after tests
    vectorstore_path = Path("core/vectorstore")
    if vectorstore_path.exists():
        shutil.rmtree(vectorstore_path) 