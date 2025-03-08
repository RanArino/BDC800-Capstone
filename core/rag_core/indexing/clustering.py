"""
Clustering module for the SCALER framework.
Implements K-means and GMM clustering methods for semantic clustering of document chunks.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
    """Perform K-means clustering on embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        n_clusters: Number of clusters (if None, will be estimated)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (cluster labels, cluster centroids, cluster_to_indices mapping)
    """
    # Estimate number of clusters if not provided
    if n_clusters is None:
        n_clusters = max(2, min(int(np.sqrt(len(embeddings))), 20))
    
    logger.info(f"Performing K-means clustering with {n_clusters} clusters")
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    
    # Create mapping from cluster ID to indices
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return labels, centroids, clusters

def gmm_clustering(
    embeddings: np.ndarray,
    n_components: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
    """Perform Gaussian Mixture Model clustering on embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        n_components: Number of components (if None, will be estimated)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (cluster labels, cluster centroids, cluster_to_indices mapping)
    """
    # Estimate number of components if not provided
    if n_components is None:
        n_components = max(2, min(int(np.sqrt(len(embeddings))), 20))
    
    logger.info(f"Performing GMM clustering with {n_components} components")
    
    # Perform GMM clustering
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(embeddings)
    
    # Get centroids (means of the Gaussians)
    centroids = gmm.means_
    
    # Create mapping from cluster ID to indices
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return labels, centroids, clusters

def run_clustering(
    embeddings: List[List[float]],
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    items_per_cluster: int = 15,
    **kwargs
) -> Union[KMeans, GaussianMixture]:
    """Run clustering on embeddings.
    
    Args:
        embeddings: List of embedding vectors
        method: Clustering method ("kmeans" or "gmm")
        n_clusters: Number of clusters (if None, will be estimated)
        items_per_cluster: Target number of items per cluster (used if n_clusters is None).
                          This is a guideline, not a strict requirement - actual clusters may
                          contain fewer or more items depending on the data distribution.
        **kwargs: Additional arguments for the specific method
        
    Returns:
        The clustering model object, which can be:
        - KMeans: with attributes labels_, cluster_centers_
        - GaussianMixture: with methods predict() and attribute means_
    """
    # Convert embeddings to numpy array if not already
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Check if embeddings array is empty
    if embeddings_array.size == 0:
        logger.warning("Empty embeddings array provided to clustering. Returning None.")
        return None
    
    # Estimate number of clusters if not provided
    if n_clusters is None:
        # Calculate based on desired items per cluster (as a guideline)
        n_clusters = max(2, min(len(embeddings) // items_per_cluster, 20))
        logger.info(f"Estimated {n_clusters} clusters for {len(embeddings)} items (target guideline: ~{items_per_cluster} items/cluster)")
    
    # Apply clustering based on method
    if method.lower() == "kmeans":
        # For KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=kwargs.get("random_state", 42),
            n_init=10
        )
        kmeans.fit(embeddings_array)
        
        # Log cluster sizes
        labels = kmeans.labels_
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        cluster_sizes = {cluster_id: len(indices) for cluster_id, indices in clusters.items()}
        logger.info(f"KMeans cluster sizes: {cluster_sizes}")
        
        return kmeans
            
    elif method.lower() == "gmm":
        # For GMM
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=kwargs.get("random_state", 42)
        )
        gmm.fit(embeddings_array)
        
        # Log cluster sizes
        labels = gmm.predict(embeddings_array)
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        cluster_sizes = {cluster_id: len(indices) for cluster_id, indices in clusters.items()}
        logger.info(f"GMM cluster sizes: {cluster_sizes}")
        
        return gmm
            
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
