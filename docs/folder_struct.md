# Folder Structure

This document outlines the organization of our RAG framework codebase.

## Overview

```
├── core/             # Core framework and operations
│   ├── frameworks/           # All RAG implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Base RAG framework
│   │   ├── simple.py        # Basic vector search
│   │   ├── scaler.py        # Our full implementation
│   │   └── schema.py        # Configuration schemas
│   │
│   ├── rag_core/
│   │   ├── indexing/        # Document processing
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py       # Uses langchain
│   │   │   ├── clustering.py    # clustering functions
│   │   │   ├── dim_reduction.py # dimentional reduction functions
│   │   │   └── summarizer.py    # Uses rag_core.llm
│   │   │
│   │   ├── llm/            # LLM and embedding models management
│   │   │   ├── __init__.py
│   │   │   ├── controller.py    # LLM and embedding models controller
│   │   │   └── schema.py        # LLM configurations schema
│   │   │
│   │   ├── retrieval/       # Search operations
│   │   │   ├── __init__.py
│   │   │   ├── search.py        # FAISS operations
│   │   │   └── expander.py      # Optional query expansion, uses \
│   │   │
│   │   │
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── profiler.py      # Performance tracking
│   │   └── metrics.py       # Timing metrics collection
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics_summary.py   # Aggregates metrics and performance data
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── retrieval.py     # Retrieval quality
│   │   │   ├── generation.py    # LLM output quality
│   │   │   └── self_checker.py  # Logical consistency validator for QA pairs
│   │   └── viz/
│   │       ├── __init__.py
│   │       ├── viz_test.ipynb    # Visualization test
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
├── test/            # Test suite
│   ├── chunker.py            # Chunking test
│   ├── simple_rag.py         # SimpleRAG implementation tests
│   ├── eval_retrieval.py     # Retrieval evaluation tests
│   ├── eval_generation.py    # Generation evaluation tests
│   ├── llm_controller.py     # LLM controller test
│   ├── input_data/           # Input data for evaluation test
│   │   └── ...
│   └── output_data/          # Evaluation output data
│       └── ...
│
├── experiments/     # Experiment execution and results storage
│   ├── __init__.py
│   ├── base.py               # Core experiment runner implementation
│   ├── detailed_dfs/         # Detailed metrics dataframes storage
│   │   └── *.csv             # CSV files with detailed metrics
│   ├── metrics/              # Experiment metrics summaries
│   │   └── *.json            # JSON files with metrics summaries
│   └── responses/            # RAG response storage
│       └── *.json            # JSON files with RAG responses
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
- `schema.py`: Configuration schemas and data models for RAG frameworks


#### Core Operations (rag_core/)

##### Indexing Operations
Document processing operations:
- `chunker.py`: Document chunking using langchain
- `summarizer.py`: Document summarization using rag_core.llm

##### Retrieval Operations
Search and query operations:
- `search.py`: FAISS-based search operations
- `expander.py`: Optional query expansion using rag_core.llm

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

#### Evaluation
Performance measurement and visualization:
##### Metrics
- `retrieval.py`: Retrieval quality metrics
- `generation.py`: LLM output quality assessment
- `self_checker.py`: Validates logical consistency between questions, answers, and reasoning using LLM
- `metrics_summary.py`: Aggregates metrics and performance data including processing time and memory usage

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

### Test Suite (test/)
- `test/`: Framework testing and evaluation
  - `chunker.py`: Chunking test
  - `simple_rag.py`: SimpleRAG implementation tests
  - `eval_retrieval.py`: Retrieval evaluation tests
  - `eval_generation.py`: Generation evaluation tests
  - `llm_controller.py`: LLM controller test
  - `input_data/`: Test input data files
  - `output_data/`: Test output and metrics files

### Experiments Module (experiments/)
Experiment execution and results storage:
- `base.py`: Core experiment runner implementation with memory-efficient metrics calculation
- `detailed_dfs/`: Storage for detailed metrics dataframes in CSV format
- `metrics/`: Storage for experiment metrics summaries in JSON format
- `responses/`: Storage for RAG responses in JSON format

## Key Files
- `constants.py`: Global constants used across operations
- `exceptions.py`: Custom exception definitions