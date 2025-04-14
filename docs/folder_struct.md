# Folder Structure

This document outlines the organization of our RAG framework codebase.

## Overview

```
├── core/             # Core framework and operations
│   ├── frameworks/           # All RAG implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Base RAG framework
│   │   ├── simple.py        # Basic vector search
│   │   ├── scaler.py        # Our latest full implementation
│   │   ├── scaler_v1.py     # Two layer approach with IVF method
│   │   └── schema.py        # Configuration schemas
│   │
│   ├── rag_core/
│   │   ├── __init__.py
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
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── libs/            # Utility libraries/helpers
│   │   ├── ml_io.py         # Machine learning I/O utilities
│   │   ├── path.py          # Path manipulation utilities
│   │   ├── process_bar.py   # Progress bar utility
│   │   ├── profiler.py      # Performance tracking
│   │   └── README.md        # Utils documentation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics_summary.py   # Aggregates metrics and performance data
│   │   ├── schema.py        # Evaluation configuration schema
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── retrieval.py     # Retrieval quality
│   │   │   ├── generation.py    # LLM output quality
│   │   │   └── self_checker.py  # Logical consistency validator
│   │   └── viz/
│   │       ├── __init__.py
│   │       ├── viz_test.ipynb    # Visualization test
│   │       ├── performance.py    # Performance visualizations
│   │       └── results.py        # Quality metric plots
│   │
│   ├── configs/
│   │   ├── models.yaml          # LLM/Embedding model configurations
│   │   ├── datasets.yaml        # Dataset configurations
│   │   ├── logging.yaml         # Logging configurations
│   │   ├── simple_rag/          # Configs for SimpleRAG
│   │   │    └── rag_config.yaml
│   │   ├── scaler_rag/          # Configs for ScalerRAG
│   │   │    └── rag_config.yaml
│   │   └── scaler_v1_rag/       # Configs for ScalerV1RAG
│   │        └── rag_config.yaml
│   │
│   ├── logger/              # Logging system (moved into core)
│   │   ├── __init__.py
│   │   ├── logger.py        # Main logging implementation
│   │   ├── example.py       # Example usage and configuration
│   │   └── logs/            # Log files storage (gitignored)
│   │       └── ...
│   │
│   ├── datasets/            # Dataset management and storage (framework-agnostic, moved into core)
│   │   ├── __init__.py
│   │   ├── base.py          # Base dataset class and interfaces
│   │   ├── schema.py        # Common schema for all datasets
│   │   ├── README.md        # Datasets documentation
│   │   ├── test.ipynb       # Dataset testing notebook
│   │   └── data/            # Dataset implementations (gitignored)
│   │       ├── qasper/      # Example dataset with detailed structure
│   │       │   └── ...      # (processor.py, analysis.ipynb, test.json, data.json)
│   │       ├── narrativeqa/ # Question answering on narrative texts
│   │       │   └── ...
│   │       ├── multihoprag/ # Multi-hop reasoning dataset
│   │       │   └── ...
│   │       └── frames/      # Frames benchmark dataset
│   │           └── ...
│   │
│   ├── vectorstore/         # Physical storage for indices (gitignored)
│   │   └── ...
│   │
│   └── __init__.py
│
├── test/            # Test suite
│   ├── chunker.py            # Chunking test
│   ├── clustering.py         # Clustering tests
│   ├── datasets_frames.py    # Frames dataset tests
│   ├── datasets_multihoprag.py # MultiHopRAG dataset tests
│   ├── datasets_narrativeqa.py # NarrativeQA dataset tests
│   ├── datasets_qasper.py    # QASPER dataset tests
│   ├── eval_generation.py    # Generation evaluation tests
│   ├── eval_metrics_summary.py # Metrics summary evaluation tests
│   ├── eval_retrieval.py     # Retrieval evaluation tests
│   ├── eval_selfchecker.py   # Self-checker evaluation tests
│   ├── llm_controller.py     # LLM controller test
│   ├── scaler_rag.py         # ScalerRAG implementation tests
│   ├── scaler_v1_rag.py      # ScalerV1RAG implementation tests
│   ├── simple_rag.py         # SimpleRAG implementation tests
│   ├── summarizer.py         # Summarizer tests
│   ├── test.ipynb            # General test notebook
│   ├── test_scaler_index_loading.py # Scaler index loading tests
│   ├── test_scaler_storage.py # Scaler storage tests
│   ├── core/                 # Core component tests
│   │   └── ...
│   ├── input_data/           # Input data for tests (gitignored)
│   │   └── ...
│   └── output_data/          # Test output data (gitignored)
│       └── ...
│
├── experiments/     # Experiment execution and results storage
│   ├── __init__.py
│   ├── base.py               # Core experiment runner implementation
│   ├── main.py               # Main script to run experiments
│   ├── result_intraqa.csv    # Intra-QA results summary
│   ├── result_interqa.csv    # Inter-QA results summary
│   ├── detailed_dfs/         # Detailed metrics dataframes storage (gitignored)
│   │   └── *.csv             # CSV files with detailed metrics
│   ├── metrics/              # Experiment metrics summaries (gitignored)
│   │   └── *.json            # JSON files with metrics summaries
│   └── responses/            # RAG response storage (gitignored)
│       └── *.json            # JSON files with RAG responses
│
├── docs/             # Documentation files
│   └── folder_struct.md # This file
│
├── .gitignore        # Specifies intentionally untracked files
├── README.md         # Project overview and setup instructions
└── requirements.txt  # Project dependencies
```
