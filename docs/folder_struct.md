# Folder Structure

This document outlines the organization of our RAG framework codebase.

## Overview

```
├── core/             # Core framework and operations
│   ├── rag_core/    # Core reusable RAG operations
│   │   ├── indexing/     # Independent indexing operations
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py    # Document chunking operations
│   │   │   ├── embedder.py   # Embedding generation operations
│   │   │   ├── summarizer.py # Summary generation operations
│   │   │   └── clusterer.py  # Clustering operations
│   │   │
│   │   ├── retrieval/    # Independent retrieval operations
│   │   │   ├── __init__.py
│   │   │   ├── processor.py  # Query processing operations
│   │   │   └── search.py     # Search operations
│   │   │
│   │   ├── llm/         # LLM integration operations
│   │   │   ├── __init__.py
│   │   │   ├── base.py      # Abstract base class for LLM operations
│   │   │   ├── gemini.py    # Gemini-specific operations
│   │   │   └── ollama.py    # Ollama-specific operations
│   │   │
│   │   ├── utils/       # Common utilities
│   │   │   ├── __init__.py
│   │   │   ├── profiler.py  # Performance monitoring
│   │   │   └── parser.py    # Document parsing utilities (PDF, DOCX to JSON)
│   │   │
│   │   ├── __init__.py
│   │   ├── constants.py    # Global constants
│   │   └── exceptions.py   # Custom exceptions
│   │
│   ├── frameworks/     # RAG framework implementations
│   │   ├── __init__.py
│   │   ├── simple.py     # Simple RAG implementation
│   │   └── ...           # Other framework implementations
│   │
│   ├── evaluation/    # Evaluation and visualization
│   │   ├── __init__.py
│   │   ├── metrics.py    # Performance metrics calculation
│   │   └── viz.py        # Results visualization
│   │
│   ├── __init__.py
│   ├── base.py        # Base RAG framework class and interfaces
│   └── run.py         # Main entry point for framework selection and evaluation
│
├── vectorstore/      # Vector store operations (framework-agnostic)
│   ├── __init__.py
│   ├── base.py       # Base vector store configuration
│   ├── store.py      # FAISS store implementation
│   ├── config.py     # FAISS-specific configurations
│   └── storage/      # Physical storage for dataset indices
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

#### Base Framework (`base.py`)
- Defines base RAG framework class with required methods:
  - `process_documents()`: Document processing pipeline
  - `query()`: Query processing and retrieval
  - `evaluate()`: Framework evaluation
- Provides common utilities and interfaces

#### Framework Implementations (frameworks/)
Each framework in a single file:
- `simple.py`: Simple RAG implementation

Each implementation inherits from base framework class and defines:
- Framework-specific parameters
- Document processing logic
- Query processing logic
- Custom methods if needed

#### Evaluation and Visualization (evaluation/)
Tools for measuring and visualizing performance:
- `metrics.py`: Performance metrics calculation
  - Retrieval accuracy
  - Response quality
  - Processing time
  - Memory usage
- `viz.py`: Results visualization
  - Performance comparisons
  - Metric plots
  - Analysis tools

#### Execution Entry Point (`run.py`)
Main script for running frameworks:
- Framework selection
- Dataset selection
- Evaluation execution
- Results generation and visualization

#### Logging System (logger/)
Core logging operations:
- `logger.py`: Main logging implementation
- `example.py`: Example usage and configuration
- `logs/`: Log files storage directory

### Core Operations (rag_core/)

#### Indexing Operations
Independent operations for document processing:
- `chunker.py`: Document chunking operations
- `embedder.py`: Vector embedding generation operations
- `summarizer.py`: Document summary generation operations
- `clusterer.py`: Semantic clustering operations

#### Retrieval Operations
Independent operations for retrieval:
- `processor.py`: Query processing operations
- `search.py`: Search strategy operations

#### LLM Operations
Independent LLM provider operations:
- `base.py`: Abstract base class for LLM operations
- `gemini.py`: Google Gemini operations
- `ollama.py`: Local model operations

#### Utils
Common utilities:
- `profiler.py`: Performance monitoring tools
- `parser.py`: Document parsing utilities for preprocessing (PDF, DOCX to JSON)

### Framework-Agnostic Components

#### Vector Store Module (vectorstore/)
Framework-agnostic vector store operations:
- `base.py`: Base vector store configurations
- `store.py`: FAISS store implementation
- `config.py`: FAISS-specific configurations
- `storage/`: Physical storage for dataset indices

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