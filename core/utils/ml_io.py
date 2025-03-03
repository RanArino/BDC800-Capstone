# core/utils/ml_io.py
"""
Utility functions for saving and loading machine learning models.
Provides functionality to handle dimension reduction and clustering models.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Tuple

# Setup logger
logger = logging.getLogger(__name__)

def save_ml_models(
    models_dir: str,
    prefix: str,
    dim_reduction_model: Any = None,
    clustering_model: Any = None
) -> None:
    """Save dimension reduction and clustering models to disk.
    
    Args:
        models_dir: Directory to save the models
        prefix: Prefix for the model filenames
        dim_reduction_model: Dimension reduction model to save
        clustering_model: Clustering model to save
    """
    from joblib import dump
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Save dimension reduction model if provided
    if dim_reduction_model is not None:
        try:
            dim_path = os.path.join(models_dir, f"{prefix}.dim_reduction.joblib")
            dump(dim_reduction_model, dim_path)
            logger.debug(f"Saved dimension reduction model to {dim_path}")
        except Exception as e:
            logger.warning(f"Failed to save dimension reduction model for {prefix}: {e}")
    
    # Save clustering model if provided
    if clustering_model is not None:
        try:
            cluster_path = os.path.join(models_dir, f"{prefix}.clustering.joblib")
            dump(clustering_model, cluster_path)
            logger.debug(f"Saved clustering model to {cluster_path}")
        except Exception as e:
            logger.warning(f"Failed to save clustering model for {prefix}: {e}")

def load_ml_models(
    models_dir: str,
    prefix: str,
    check_legacy: bool = True
) -> Tuple[Optional[Any], Optional[Any]]:
    """Load dimension reduction and clustering models from disk.
    
    Args:
        models_dir: Directory where models are stored
        prefix: Prefix for the model filenames
        check_legacy: Whether to check for legacy .models.joblib files
        
    Returns:
        Tuple containing (dimension_reduction_model, clustering_model)
        Either model may be None if not found or there was an error loading
    """
    from joblib import load
    
    dim_reduction_model = None
    clustering_model = None
    
    # Check for legacy combined model file first
    if check_legacy:
        legacy_model_path = os.path.join(models_dir, f"{prefix}.models.joblib")
        if os.path.exists(legacy_model_path):
            try:
                models = load(legacy_model_path)
                
                # Extract dimension reduction model
                if 'dim_reduction' in models:
                    dim_reduction_model = models['dim_reduction']
                    logger.debug(f"Loaded dimension reduction model from legacy file {legacy_model_path}")
                
                # Extract clustering model
                if 'clustering' in models:
                    clustering_model = models['clustering']
                    logger.debug(f"Loaded clustering model from legacy file {legacy_model_path}")
                
                return dim_reduction_model, clustering_model
                
            except Exception as e:
                logger.warning(f"Failed to load legacy ML models from {legacy_model_path}: {e}")
    
    # Try to load separate model files
    
    # Load dimension reduction model
    dim_path = os.path.join(models_dir, f"{prefix}.dim_reduction.joblib")
    if os.path.exists(dim_path):
        try:
            dim_reduction_model = load(dim_path)
            logger.debug(f"Loaded dimension reduction model from {dim_path}")
        except Exception as e:
            logger.warning(f"Failed to load dimension reduction model from {dim_path}: {e}")
    
    # Load clustering model
    cluster_path = os.path.join(models_dir, f"{prefix}.clustering.joblib")
    if os.path.exists(cluster_path):
        try:
            clustering_model = load(cluster_path)
            logger.debug(f"Loaded clustering model from {cluster_path}")
        except Exception as e:
            logger.warning(f"Failed to load clustering model from {cluster_path}: {e}")
    
    return dim_reduction_model, clustering_model

def save_layer_models(
    layer: str,
    node_id: str,
    layer_path: str,
    dim_reduction_models: Dict[str, Dict[str, Any]],
    clustering_models: Dict[str, Dict[str, Any]],
) -> None:
    """Save ML models for a specific layer and node.
    
    Args:
        layer: Layer name ('doc' or 'chunk')
        node_id: Node identifier
        layer_path: Path to the layer directory
        dim_reduction_models: Dictionary of dimension reduction models
        clustering_models: Dictionary of clustering models
    """
    # For clustering, we should save models at the document level, not at the cluster level
    # Extract base document ID for chunks
    doc_id = None
    if layer == "chunk":
        # For cluster-specific node (e.g., "doc_1-0"), get the document ID ("doc_1")
        if '-' in node_id:
            doc_id = node_id.split('-')[0]
        else:
            # Direct document node
            doc_id = node_id
            
    # Determine which models to save
    dim_model = None
    cluster_model = None
    
    # For document layer, we have a single model for all documents
    if layer == "doc":
        dim_model = dim_reduction_models[layer]
        cluster_model = clustering_models[layer]
        
        # Save models at the layer level for doc layer
        model_prefix = layer
        save_ml_models(layer_path, model_prefix, dim_model, cluster_model)
    
    # For chunk layer, each document has its own models
    elif layer == "chunk" and doc_id:
        # Only save if we have models for this document
        if doc_id in dim_reduction_models[layer]:
            dim_model = dim_reduction_models[layer][doc_id]
            
            # Only save document models once per document (avoid duplicate saves)
            dim_path = os.path.join(layer_path, f"{doc_id}.dim_reduction.joblib")
            if not os.path.exists(dim_path):
                save_ml_models(layer_path, doc_id, dim_model, None)
        
        if doc_id in clustering_models[layer]:
            cluster_model = clustering_models[layer][doc_id]
            
            # Only save document models once per document
            cluster_path = os.path.join(layer_path, f"{doc_id}.clustering.joblib")
            if not os.path.exists(cluster_path):
                save_ml_models(layer_path, doc_id, None, cluster_model)

def load_layer_models(
    layer: str,
    node_id: str,
    layer_path: str
) -> Tuple[Optional[Any], Optional[Any]]:
    """Load ML models for a specific layer and node.
    
    Args:
        layer: Layer name ('doc' or 'chunk')
        node_id: Node identifier
        layer_path: Path to the layer directory
        
    Returns:
        Tuple containing (dimension_reduction_model, clustering_model)
    """
    # For document layer, we just load the models at the layer level
    if layer == "doc":
        # For legacy reasons, try node_id first
        dim_model, cluster_model = load_ml_models(layer_path, node_id)
        if dim_model is None and cluster_model is None:
            # Try layer name (new approach)
            dim_model, cluster_model = load_ml_models(layer_path, layer)
        return dim_model, cluster_model
    
    # For chunk layer, we need to load the models for the parent document
    elif layer == "chunk":
        # Extract document ID from node ID
        doc_id = node_id
        if '-' in node_id:
            doc_id = node_id.split('-')[0]
        
        # Try to load models for this document
        return load_ml_models(layer_path, doc_id)
    
    # Default: return None for both models
    return None, None 