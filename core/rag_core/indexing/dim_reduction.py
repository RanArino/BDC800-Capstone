"""
Dimensional reduction module for the SCALER framework.
Implements PCA, t-SNE, and UMAP methods for reducing embedding dimensions.
"""

import numpy as np
import logging
from typing import Tuple, Any, List, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)

def apply_pca(
    embeddings: np.ndarray,
    n_components: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """Apply PCA to reduce embedding dimensions.
    
    Args:
        embeddings: Numpy array of embeddings
        n_components: Number of components to keep
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (reduced embeddings, fitted PCA model)
    """
    logger.info(f"Applying PCA to reduce dimensions to {n_components}")
    
    # Check if embeddings array is empty
    if embeddings.size == 0:
        logger.warning("Empty embeddings array provided to PCA. Returning empty array and default PCA model.")
        # Return empty array and initialized PCA model
        pca = PCA(n_components=n_components, random_state=random_state)
        return np.array([], dtype=np.float32).reshape(0, n_components), pca
    
    # Ensure n_components is not larger than the number of samples or features
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Log explained variance
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"PCA explained variance: {explained_variance:.2f}%")
    
    return reduced_embeddings, pca

def apply_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, umap.UMAP]:
    """Apply UMAP to reduce embedding dimensions.
    
    Args:
        embeddings: Numpy array of embeddings
        n_components: Number of components to keep
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (reduced embeddings, fitted UMAP model)
    """
    logger.info(f"Applying UMAP to reduce dimensions to {n_components}")
    
    # Check if embeddings array is empty
    if embeddings.size == 0:
        logger.warning("Empty embeddings array provided to UMAP. Returning empty array and default UMAP model.")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        return np.array([], dtype=np.float32).reshape(0, n_components), reducer
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    return reduced_embeddings, reducer

def run_dim_reduction(
    embeddings: List[List[float]],
    method: str = "pca",
    n_components: int = 50,
    **kwargs
) -> Union[PCA, umap.UMAP]:
    """Reduce dimensions of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        method: Dimensionality reduction method ("pca" or "umap")
        n_components: Number of components to keep
        **kwargs: Additional arguments for the specific method
        
    Returns:
        Fitted dimensionality reduction model (either PCA or UMAP)
        with transform() method for reducing new data
    """
    # Convert embeddings to numpy array if not already
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Apply dimensionality reduction
    if method.lower() == "pca":
        _, model = apply_pca(
            embeddings_array, 
            n_components=n_components,
            random_state=kwargs.get("random_state", 42)
        )
    elif method.lower() == "umap":
        _, model = apply_umap(
            embeddings_array,
            n_components=n_components,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            random_state=kwargs.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    return model

def reduce_query_embedding(
    query_embedding: List[float],
    model: Union[PCA, umap.UMAP]
) -> np.ndarray:
    """Apply a pre-fitted dimensionality reduction model to a query embedding.
    
    Args:
        query_embedding: The original query embedding vector
        model: A fitted dimensionality reduction model (PCA or UMAP)
        
    Returns:
        Reduced query embedding as numpy array
    """
    # Convert query embedding to numpy array
    query_array = np.array([query_embedding], dtype=np.float32)
    
    # Apply the reduction model
    if isinstance(model, PCA):
        reduced_query = model.transform(query_array)
    elif isinstance(model, umap.UMAP):
        # UMAP transform requires more careful handling
        reduced_query = model.transform(query_array)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    return reduced_query[0]  # Return as a 1D array
