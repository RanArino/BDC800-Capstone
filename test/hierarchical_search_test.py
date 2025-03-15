"""
Test for the simplified hierarchical search implementation.

This script compares the performance of hierarchical search vs. flat search,
focusing specifically on the efficiency gains from pruning the search space
based on document similarity.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import sys
import os
import argparse
from tqdm import tqdm

# Add parent directory to Python path for imports to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the hierarchical search implementation
from core.rag_core.indexing import HierarchicalIndexer
from core.rag_core.retrieval import HierarchicalRetriever

# Define test document structure
class TestDocument:
    """Simple document class for testing"""
    def __init__(self, doc_id: str, embedding: np.ndarray, chunk_embeddings: List[np.ndarray]):
        self.id = doc_id
        self.embedding = embedding
        self.chunk_embeddings = chunk_embeddings
        self.chunks = [f"chunk_{i}" for i in range(len(chunk_embeddings))]

# Baseline flat search implementation
class FlatVectorStore:
    """Simple flat vector store for baseline comparison"""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = None
        self.ids = []
        
    def add_vectors(self, vectors: np.ndarray, ids: List[str]):
        """Add vectors to the store"""
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            
        self.ids.extend(ids)
        
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[str]]:
        """Search for the most similar vectors"""
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([]), []
            
        # Ensure query is properly shaped
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
            
        # Calculate distances (L2)
        distances = np.sqrt(np.sum((self.vectors - query) ** 2, axis=1))
        
        # Get top k indices
        if len(distances) <= k:
            top_indices = np.argsort(distances)
        else:
            top_indices = np.argsort(distances)[:k]
            
        # Return results
        result_distances = distances[top_indices]
        result_ids = [self.ids[i] for i in top_indices]
        
        return result_distances, result_ids

def generate_test_data(num_docs: int, chunks_per_doc: int, dimension: int = 768) -> List[TestDocument]:
    """Generate test documents with random embeddings"""
    np.random.seed(42)  # For reproducibility
    
    documents = []
    for i in range(num_docs):
        doc_id = f"doc_{i}"
        
        # Generate document embedding
        doc_embedding = np.random.random(dimension).astype('float32')
        
        # Generate chunk embeddings
        # Make chunks somewhat similar to their parent document
        chunk_embeddings = []
        for j in range(chunks_per_doc):
            # Mix document embedding with random noise
            chunk_embedding = 0.7 * doc_embedding + 0.3 * np.random.random(dimension)
            chunk_embedding = chunk_embedding.astype('float32')
            chunk_embeddings.append(chunk_embedding)
            
        documents.append(TestDocument(doc_id, doc_embedding, chunk_embeddings))
        
    return documents

def index_documents(documents: List[TestDocument]) -> Tuple[HierarchicalRetriever, FlatVectorStore]:
    """Index documents in both hierarchical and flat stores"""
    # Initialize stores
    dimension = documents[0].embedding.shape[0]
    hierarchical_indexer = HierarchicalIndexer(dimension=dimension)
    hierarchical_retriever = HierarchicalRetriever(indexer=hierarchical_indexer)
    flat_store = FlatVectorStore(dimension=dimension)
    
    # Prepare document vectors
    doc_vectors = []
    doc_ids = []
    
    # Prepare chunk vectors
    all_chunk_vectors = []
    all_chunk_ids = []
    all_chunk_parent_ids = []
    
    # Process documents
    for doc in documents:
        # Add document
        doc_vectors.append(doc.embedding)
        doc_ids.append(doc.id)
        
        # Add chunks
        for i, chunk_embedding in enumerate(doc.chunk_embeddings):
            chunk_id = f"{doc.id}_chunk_{i}"
            all_chunk_vectors.append(chunk_embedding)
            all_chunk_ids.append(chunk_id)
            all_chunk_parent_ids.append(doc.id)
    
    # Convert to numpy arrays
    doc_vectors_np = np.array(doc_vectors)
    all_chunk_vectors_np = np.array(all_chunk_vectors)
    
    # Index in hierarchical store
    hierarchical_indexer.add_documents(doc_vectors_np, doc_ids)
    hierarchical_indexer.add_chunks(all_chunk_vectors_np, all_chunk_ids, all_chunk_parent_ids)
    
    # Index in flat store (all vectors together)
    all_vectors = np.vstack([doc_vectors_np, all_chunk_vectors_np])
    all_ids = doc_ids + all_chunk_ids
    flat_store.add_vectors(all_vectors, all_ids)
    
    return hierarchical_retriever, flat_store

def generate_test_queries(num_queries: int, dimension: int = 768) -> List[np.ndarray]:
    """Generate random test queries"""
    np.random.seed(43)  # Different seed from documents
    return [np.random.random(dimension).astype('float32') for _ in range(num_queries)]

def performance_test(
    num_docs: int = 100, 
    chunks_per_doc: int = 30, 
    num_queries: int = 10, 
    k: int = 10,
    top_k_docs: int = 3
) -> Dict:
    """
    Test the performance of hierarchical vs. flat search
    
    Args:
        num_docs: Number of documents to generate
        chunks_per_doc: Number of chunks per document
        num_queries: Number of test queries
        k: Number of results to return
        top_k_docs: Number of top documents to consider in hierarchical search
        
    Returns:
        Dict with performance metrics
    """
    print(f"\n=== Performance Test ===")
    print(f"Configuration:")
    print(f"- Documents: {num_docs}")
    print(f"- Chunks per document: {chunks_per_doc}")
    print(f"- Total vectors: {num_docs + (num_docs * chunks_per_doc)}")
    print(f"- Queries: {num_queries}")
    print(f"- Top-k: {k}")
    print(f"- Top-k docs for hierarchical: {top_k_docs}")
    
    # Generate test data
    print("\nGenerating test data...")
    documents = generate_test_data(num_docs, chunks_per_doc)
    
    # Index documents
    print("Indexing documents...")
    hierarchical_retriever, flat_store = index_documents(documents)
    
    # Generate test queries
    print("Generating test queries...")
    queries = generate_test_queries(num_queries)
    
    # Test flat search
    print("\nTesting flat search...")
    flat_times = []
    for query in tqdm(queries):
        start_time = time.time()
        flat_store.search(query, k)
        flat_times.append(time.time() - start_time)
    
    # Test hierarchical search
    print("\nTesting hierarchical search...")
    hierarchical_times = []
    for query in tqdm(queries):
        start_time = time.time()
        hierarchical_retriever.hierarchical_search(query, top_k_docs, k)
        hierarchical_times.append(time.time() - start_time)
    
    # Calculate statistics
    flat_avg = np.mean(flat_times)
    hierarchical_avg = np.mean(hierarchical_times)
    
    print("\n=== Results ===")
    print(f"Flat search average time: {flat_avg:.6f} seconds")
    print(f"Hierarchical search average time: {hierarchical_avg:.6f} seconds")
    
    if hierarchical_avg < flat_avg:
        speedup = flat_avg / hierarchical_avg
        print(f"Hierarchical search is {speedup:.2f}x faster than flat search")
    else:
        slowdown = hierarchical_avg / flat_avg
        print(f"Hierarchical search is {slowdown:.2f}x slower than flat search")
    
    # Calculate theoretical speedup
    total_vectors = num_docs + (num_docs * chunks_per_doc)
    hierarchical_vectors = top_k_docs + (top_k_docs * chunks_per_doc)
    theoretical_speedup = total_vectors / hierarchical_vectors
    
    print("\n=== Vector Operations Analysis ===")
    print(f"Flat search compares against all {total_vectors} vectors")
    print(f"Hierarchical search compares against {top_k_docs} doc vectors + ~{top_k_docs * chunks_per_doc} chunks = ~{hierarchical_vectors} vectors")
    print(f"Theoretical speedup: {theoretical_speedup:.2f}x")
    print(f"Actual speedup: {flat_avg / hierarchical_avg:.2f}x")
    
    # Generate performance plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([flat_times, hierarchical_times], labels=["Flat Search", "Hierarchical Search"])
    plt.title(f"Search Performance (docs={num_docs}, chunks={chunks_per_doc})")
    plt.ylabel("Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("hierarchical_search_performance.png")
    print("\nPerformance plot saved as 'hierarchical_search_performance.png'")
    
    return {
        "flat_times": flat_times,
        "hierarchical_times": hierarchical_times,
        "flat_avg": flat_avg,
        "hierarchical_avg": hierarchical_avg,
        "theoretical_speedup": theoretical_speedup,
        "actual_speedup": flat_avg / hierarchical_avg
    }

def run_scaling_tests():
    """Run tests with different dataset sizes to analyze scaling behavior"""
    configurations = [
        {"num_docs": 10, "chunks_per_doc": 10, "num_queries": 5},
        {"num_docs": 50, "chunks_per_doc": 20, "num_queries": 5},
        {"num_docs": 100, "chunks_per_doc": 30, "num_queries": 5}, 
        {"num_docs": 200, "chunks_per_doc": 30, "num_queries": 3},
        {"num_docs": 10, "chunks_per_doc": 1000, "num_queries": 3}  # Large number of chunks per doc
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n\n{'='*80}")
        print(f"Running test with {config['num_docs']} documents and {config['chunks_per_doc']} chunks per document")
        print(f"{'='*80}")
        
        result = performance_test(**config)
        results.append((config, result))
    
    print("\n\n=== Scaling Test Summary ===")
    for config, result in results:
        num_docs = config["num_docs"]
        chunks_per_doc = config["chunks_per_doc"]
        actual_speedup = result["actual_speedup"]
        theoretical_speedup = result["theoretical_speedup"]
        
        print(f"- {num_docs} docs × {chunks_per_doc} chunks: theoretical {theoretical_speedup:.2f}x, actual {actual_speedup:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hierarchical search performance")
    parser.add_argument("--docs", type=int, default=100, help="Number of documents")
    parser.add_argument("--chunks", type=int, default=30, help="Chunks per document")
    parser.add_argument("--queries", type=int, default=10, help="Number of test queries")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--top-k-docs", type=int, default=3, help="Number of top documents to consider in hierarchical search")
    parser.add_argument("--scaling", action="store_true", help="Run multiple tests with different sizes")
    
    args = parser.parse_args()
    
    if args.scaling:
        run_scaling_tests()
    else:
        performance_test(
            num_docs=args.docs,
            chunks_per_doc=args.chunks,
            num_queries=args.queries,
            k=args.k,
            top_k_docs=args.top_k_docs
        ) 