# Folder Structure

This document outlines the organization of our RAG framework codebase.

## Overview

```
├── core/             # Core framework and operations
│   ├── frameworks/           # All RAG implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Base RAG framework
│   │   ├── simple.py        # Basic vector search
│   │   └── scaler.py        # Our full implementation
│   │
│   ├── rag_core/
│   │   ├── indexing/        # Document processing
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py       # Uses langchain
│   │   │   └── summarizer.py    # Uses core.utils.llm
│   │   │
│   │   ├── retrieval/       # Search operations
│   │   │   ├── __init__.py
│   │   │   ├── search.py        # FAISS operations
│   │   │   └── expander.py      # Optional query expansion, uses core.utils.llm
│   │   │
│   │   ├── clustering/      # Clustering operations
│   │   │   ├── __init__.py
│   │   │   ├── kmeans.py
│   │   │   ├── gmm.py
│   │   │   └── reduction/
│   │   │       ├── __init__.py
│   │   │       ├── pca.py
│   │   │       └── umap.py
│   │   │
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── profiler.py      # Performance tracking
│   │   ├── metrics.py       # Timing metrics collection
│   │   └── llm.py           # Centralized LLM utility
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── timing.py        # Processing time analysis
│   │   │   ├── retrieval.py     # Retrieval quality
│   │   │   └── generation.py    # LLM output quality
│   │   └── viz/
│   │       ├── __init__.py
│   │       ├── performance.py    # Performance visualizations
│   │       └── results.py        # Quality metric plots
│   │
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── rag_config.yaml      # Framework configurations
│   │   └── logging.yaml         # Logging configurations
│   │
│   └── vectorstore/          # Physical storage for indices
│       └── ...    
│
├── datasets/         # Dataset management and storage (framework-agnostic)
│   ├── __init__.py
│   ├── base.py       # Base dataset class and interfaces
│   ├── schema.py     # Common schema for all datasets
│   └── data/         # Dataset implementations
│       ├── qasper/   # Example dataset with detailed structure
│       │   ├── processor.py    # Dataset-specific processing logic
│       │   ├── analysis.ipynb  # Dataset analysis and exploration
│       │   ├── test.json       # Test data sample
│       │   └── data.json       # Processed dataset (gitignored)
│       ├── narrativeqa/        # Question answering on narrative texts
│       ├── multihoprag/        # Multi-hop reasoning dataset
│       └── frames/             # Frames benchmark dataset
│
└── logger/          # Logging system
    ├── __init__.py
    ├── logger.py     # Main logging implementation
    ├── example.py    # Example usage and configuration
    └── logs/        # Log files storage
```

## Module Descriptions

### Core Framework (core/)

#### Framework Implementations (frameworks/)
- `base.py`: Base RAG framework with core interfaces
- `simple.py`: Basic vector search implementation
- `scaler.py`: Our full RAG implementation

#### Core Operations (rag_core/)

##### Indexing Operations
Document processing operations:
- `chunker.py`: Document chunking using langchain
- `summarizer.py`: Document summarization using core.utils.llm

##### Retrieval Operations
Search and query operations:
- `search.py`: FAISS-based search operations
- `expander.py`: Optional query expansion using LLM

##### Clustering Operations
Document clustering and dimensionality reduction:
- `kmeans.py`: K-means clustering implementation
- `gmm.py`: Gaussian Mixture Model clustering
- `reduction/`: Dimensionality reduction techniques
  - `pca.py`: Principal Component Analysis
  - `umap.py`: UMAP dimensionality reduction

#### Utils
Common utilities:
- `profiler.py`: Performance tracking tools
- `metrics.py`: Timing metrics collection
- `llm.py`: Centralized LLM utility

#### Evaluation
Performance measurement and visualization:
##### Metrics
- `timing.py`: Processing time analysis
- `retrieval.py`: Retrieval quality metrics
- `generation.py`: LLM output quality assessment

##### Visualization
- `performance.py`: Performance visualization tools
- `results.py`: Quality metric plotting

#### Configuration
Framework and system configurations:
- `rag_config.yaml`: Framework-specific configurations
- `logging.yaml`: Logging system configurations

#### Vector Store (vectorstore/)
Physical storage for indices and configurations:
- Index storage and management
- Vector database configurations

### Framework-Agnostic Components

#### Datasets Module (datasets/)
Framework-agnostic dataset management:
- `base.py`: Base dataset class and interfaces
- `schema.py`: Common schema for all datasets
- `data/`: Dataset implementations
  - `qasper/`: Intra-document QA dataset about scientific papers
  - `narrativeqa/`:Intra-document QA on narrative texts
  - `multihoprag/`: Multi-hop reasoning dataset
  - `frames/`: Frames benchmark dataset

## Key Files
- `constants.py`: Global constants used across operations
- `exceptions.py`: Custom exception definitions