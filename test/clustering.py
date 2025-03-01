# test/clustering.py

"""
Clustering and Dimensional Reduction Tests
Run the script to debug:
```
python -m pdb test/clustering.py
```
"""

import sys
import os
import gc
import warnings
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.rag_core.indexing.dim_reduction import run_dim_reduction, apply_pca, apply_umap
from core.rag_core.indexing.clustering import run_clustering, kmeans_clustering, gmm_clustering
from core.logger.logger import get_logger

# Set up logging
logger = get_logger(__name__)

def test_dim_reduction():
    """Test dimensional reduction functionality."""
    try:
        logger.info("Starting dimensional reduction tests")
        
        # Create synthetic embeddings for testing
        np.random.seed(42)
        embeddings = np.random.rand(100, 768).astype(np.float32).tolist()
        
        # Test PCA reduction
        logger.info("Testing PCA dimensional reduction")
        pca_start_time = time.time()
        reduced_embeddings_pca, pca_model = run_dim_reduction(
            embeddings=embeddings,
            method="pca",
            n_components=50
        )
        pca_time = time.time() - pca_start_time
        
        # Verify PCA output
        assert isinstance(reduced_embeddings_pca, np.ndarray), "PCA output should be a numpy array"
        assert reduced_embeddings_pca.shape == (100, 50), f"Expected shape (100, 50), got {reduced_embeddings_pca.shape}"
        
        logger.info(f"PCA reduction completed in {pca_time:.4f} seconds")
        logger.info(f"PCA output shape: {reduced_embeddings_pca.shape}")
        
        # Test UMAP reduction
        logger.info("Testing UMAP dimensional reduction")
        umap_start_time = time.time()
        reduced_embeddings_umap, umap_model = run_dim_reduction(
            embeddings=embeddings,
            method="umap",
            n_components=10
        )
        umap_time = time.time() - umap_start_time
        
        # Verify UMAP output
        assert isinstance(reduced_embeddings_umap, np.ndarray), "UMAP output should be a numpy array"
        assert reduced_embeddings_umap.shape == (100, 10), f"Expected shape (100, 10), got {reduced_embeddings_umap.shape}"
        
        logger.info(f"UMAP reduction completed in {umap_time:.4f} seconds")
        logger.info(f"UMAP output shape: {reduced_embeddings_umap.shape}")
        
        # Test edge cases
        logger.info("Testing edge cases")
        
        # Test with small number of samples
        small_embeddings = embeddings[:5]
        reduced_small, _ = run_dim_reduction(
            embeddings=small_embeddings,
            method="pca",
            n_components=10
        )
        assert reduced_small.shape == (5, 5), f"Expected shape (5, 5), got {reduced_small.shape}"
        
        # Test invalid method
        try:
            run_dim_reduction(
                embeddings=embeddings,
                method="invalid_method",
                n_components=50
            )
            assert False, "Should have raised ValueError for invalid method"
        except ValueError:
            pass
        
        logger.info("All dimensional reduction tests passed")
        
        # Save results
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "pca_time": pca_time,
            "umap_time": umap_time,
            "pca_shape": reduced_embeddings_pca.shape,
            "umap_shape": reduced_embeddings_umap.shape,
        }
        
        with open(output_dir / "dim_reduction_results.json", "w") as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        
        return reduced_embeddings_pca, reduced_embeddings_umap
        
    except Exception as e:
        logger.error(f"Error in dimensional reduction test: {e}")
        raise
    finally:
        gc.collect()

def test_clustering():
    """Test clustering functionality."""
    try:
        logger.info("Starting clustering tests")
        
        # Create synthetic embeddings for testing
        np.random.seed(42)
        embeddings = np.random.rand(100, 50).astype(np.float32).tolist()
        
        # Test K-means clustering
        logger.info("Testing K-means clustering")
        kmeans_start_time = time.time()
        labels_kmeans, centroids_kmeans, clusters_kmeans = run_clustering(
            embeddings=embeddings,
            method="kmeans",
            n_clusters=5
        )
        kmeans_time = time.time() - kmeans_start_time
        
        # Verify K-means output
        assert isinstance(labels_kmeans, list), "K-means labels should be a list"
        assert len(labels_kmeans) == 100, f"Expected 100 labels, got {len(labels_kmeans)}"
        assert isinstance(centroids_kmeans, dict), "K-means centroids should be a dictionary"
        assert len(centroids_kmeans) == 5, f"Expected 5 centroids, got {len(centroids_kmeans)}"
        
        # Check that all embeddings are assigned to a cluster
        assigned_indices = []
        for cluster_indices in clusters_kmeans.values():
            assigned_indices.extend(cluster_indices)
        assert len(set(assigned_indices)) == 100, f"Expected 100 assigned indices, got {len(set(assigned_indices))}"
        
        logger.info(f"K-means clustering completed in {kmeans_time:.4f} seconds")
        logger.info(f"K-means cluster sizes: {[len(indices) for indices in clusters_kmeans.values()]}")
        
        # Test GMM clustering
        logger.info("Testing GMM clustering")
        gmm_start_time = time.time()
        labels_gmm, centroids_gmm, clusters_gmm = run_clustering(
            embeddings=embeddings,
            method="gmm",
            n_clusters=5
        )
        gmm_time = time.time() - gmm_start_time
        
        # Verify GMM output
        assert isinstance(labels_gmm, list), "GMM labels should be a list"
        assert len(labels_gmm) == 100, f"Expected 100 labels, got {len(labels_gmm)}"
        assert isinstance(centroids_gmm, dict), "GMM centroids should be a dictionary"
        assert len(centroids_gmm) == 5, f"Expected 5 centroids, got {len(centroids_gmm)}"
        
        # Check that all embeddings are assigned to a cluster
        assigned_indices = []
        for cluster_indices in clusters_gmm.values():
            assigned_indices.extend(cluster_indices)
        assert len(set(assigned_indices)) == 100, f"Expected 100 assigned indices, got {len(set(assigned_indices))}"
        
        logger.info(f"GMM clustering completed in {gmm_time:.4f} seconds")
        logger.info(f"GMM cluster sizes: {[len(indices) for indices in clusters_gmm.values()]}")
        
        # Test automatic cluster determination
        logger.info("Testing automatic cluster determination")
        auto_start_time = time.time()
        labels_auto, centroids_auto, clusters_auto = run_clustering(
            embeddings=embeddings,
            method="kmeans",
            n_clusters=None,
            items_per_cluster=20
        )
        auto_time = time.time() - auto_start_time
        
        # Verify automatic clustering output
        expected_clusters = min(max(2, 100 // 20), 20)
        assert len(clusters_auto) == expected_clusters, f"Expected {expected_clusters} clusters, got {len(clusters_auto)}"
        
        logger.info(f"Automatic clustering completed in {auto_time:.4f} seconds")
        logger.info(f"Auto-determined {len(clusters_auto)} clusters with target of 20 items per cluster")
        logger.info(f"Auto cluster sizes: {[len(indices) for indices in clusters_auto.values()]}")
        
        # Test invalid method
        try:
            run_clustering(
                embeddings=embeddings,
                method="invalid_method",
                n_clusters=5
            )
            assert False, "Should have raised ValueError for invalid method"
        except ValueError:
            pass
        
        logger.info("All clustering tests passed")
        
        # Save results
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "kmeans_time": kmeans_time,
            "gmm_time": gmm_time,
            "auto_time": auto_time,
            "kmeans_cluster_sizes": [len(indices) for indices in clusters_kmeans.values()],
            "gmm_cluster_sizes": [len(indices) for indices in clusters_gmm.values()],
            "auto_cluster_sizes": [len(indices) for indices in clusters_auto.values()],
        }
        
        with open(output_dir / "clustering_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return labels_kmeans, centroids_kmeans, clusters_kmeans
        
    except Exception as e:
        logger.error(f"Error in clustering test: {e}")
        raise
    finally:
        gc.collect()

def test_integration():
    """Test integration of dimensional reduction and clustering."""
    try:
        logger.info("Starting integration tests")
        
        # Create synthetic embeddings for testing
        np.random.seed(42)
        embeddings = np.random.rand(100, 768).astype(np.float32).tolist()
        
        # Step 1: Dimensional reduction with PCA
        logger.info("Performing PCA dimensional reduction")
        reduced_embeddings_pca, pca_model = run_dim_reduction(
            embeddings=embeddings,
            method="pca",
            n_components=50
        )
        
        # Step 2: Clustering on reduced embeddings
        logger.info("Performing K-means clustering on PCA-reduced embeddings")
        labels_pca, centroids_pca, clusters_pca = run_clustering(
            embeddings=reduced_embeddings_pca.tolist(),
            method="kmeans",
            n_clusters=5
        )
        
        # Verify output
        assert len(labels_pca) == 100, f"Expected 100 labels, got {len(labels_pca)}"
        assert len(centroids_pca) == 5, f"Expected 5 centroids, got {len(centroids_pca)}"
        
        # Check that centroids have the correct shape
        for centroid in centroids_pca.values():
            assert centroid.shape == (50,), f"Expected centroid shape (50,), got {centroid.shape}"
        
        # Verify all indices are accounted for
        all_indices = []
        for indices in clusters_pca.values():
            all_indices.extend(indices)
        assert len(set(all_indices)) == 100, f"Expected 100 indices, got {len(set(all_indices))}"
        
        logger.info("PCA + K-means integration test passed")
        
        # Step 1: Dimensional reduction with UMAP
        logger.info("Performing UMAP dimensional reduction")
        reduced_embeddings_umap, umap_model = run_dim_reduction(
            embeddings=embeddings,
            method="umap",
            n_components=10
        )
        
        # Step 2: Clustering on reduced embeddings
        logger.info("Performing GMM clustering on UMAP-reduced embeddings")
        labels_umap, centroids_umap, clusters_umap = run_clustering(
            embeddings=reduced_embeddings_umap.tolist(),
            method="gmm",
            n_clusters=5
        )
        
        # Verify output
        assert len(labels_umap) == 100, f"Expected 100 labels, got {len(labels_umap)}"
        assert len(centroids_umap) == 5, f"Expected 5 centroids, got {len(centroids_umap)}"
        
        # Check that centroids have the correct shape
        for centroid in centroids_umap.values():
            assert centroid.shape == (10,), f"Expected centroid shape (10,), got {centroid.shape}"
        
        logger.info("UMAP + GMM integration test passed")
        logger.info("All integration tests passed")
        
        # Save results
        output_dir = Path("test/output_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "pca_kmeans_cluster_sizes": [len(indices) for indices in clusters_pca.values()],
            "umap_gmm_cluster_sizes": [len(indices) for indices in clusters_umap.values()],
        }
        
        with open(output_dir / "integration_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error in integration test: {e}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        # Run dimensional reduction tests
        reduced_embeddings_pca, reduced_embeddings_umap = test_dim_reduction()
        print("\n===== Dimensional Reduction Tests Completed =====")
        
        # Run clustering tests
        test_clustering()
        print("\n===== Clustering Tests Completed =====")
        
        # Run integration tests
        test_integration()
        print("\n===== Integration Tests Completed =====")
        
        print("\nAll tests completed successfully!")
