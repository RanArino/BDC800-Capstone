# Technical Requirements

## System Requirements

### Hardware
- Apple Silicon (M3)
- Memory: 24GB RAM
- Storage: Sufficient space for model weights and vector database

## Core Dependencies

### RAG Framework
- langchain         # Main library for RAG implementation
  - Docs: https://python.langchain.com/docs/get_started/introduction
- langchain-core   # Core components
  - Docs: https://python.langchain.com/api_reference/core/

### Vector Database
- faiss-cpu        # Primary vector store (CPU version)
- langchain-community  # For FAISS integration with LangChain
  - Docs: https://python.langchain.com/docs/integrations/vectorstores/faiss
- sentence-transformers  # For text embeddings
  - Docs: https://www.sbert.net/docs/pretrained_models.html

### LLM Integration
- google-generativeai  # For Gemini integration
  - Docs: https://ai.google.dev/tutorials/python_quickstart
- ollama             # For local LLM deployment
  - Docs: https://github.com/ollama/ollama
- python-ollama      # Python SDK for Ollama
  - Docs: https://github.com/ollama/ollama-python

### Document Processing
- langchain-document-loaders  # For PDF, DOCX, text file processing
  - Docs: https://python.langchain.com/docs/integrations/document_loaders/

## LLM Models

### Local Models (via Ollama)
1. **Deepseek-r1**
   - Version: 8B and 14B
   - Paper: https://arxiv.org/abs/2501.12948
   - Use Case: Primary local LLM option
   - Status: Pre-downloaded
   - Model Card: https://ollama.com/library/deepseek-r1

2. **Phi4**
   - Version: 14B
   - Paper: https://arxiv.org/abs/2412.08905
   - Use Case: Alternative local LLM option
   - Status: Pre-downloaded
   - Model Card: https://ollama.com/library/phi4

3. **Llama 3.1**
   - Version: 8B
   - Paper: https://arxiv.org/abs/2407.21783
   - Use Case: Multilingual support and long context processing
   - Status: Pre-downloaded
   - Model Card: https://ollama.com/library/llama3.1

### Cloud Models
1. **Gemini**
   - Version: 8B
   - API Integration: google-generativeai package
   - Reference: https://ai.google.dev/gemini-api/docs#python
   - Model Card: https://ai.google.dev/models/gemini

## Vector Store Configuration

### FAISS (Facebook AI Similarity Search)
- Version: Latest stable
- Documentation:
  - FAISS Wiki: https://github.com/facebookresearch/faiss/wiki
  - LangChain Integration: https://python.langchain.com/docs/integrations/vectorstores/faiss
  - API Reference: https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html
- Dependencies:
  - faiss-cpu        # CPU version for development
  - faiss-gpu        # GPU version for production (optional)
- Features utilized:
  - Approximate Nearest Neighbor (ANN) search
  - IndexFlatL2      # Exact L2 distance computation
  - IndexIVFFlat     # Inverted file with flat storage
  - HNSW             # Hierarchical Navigable Small World graphs
  - Metadata filtering with MongoDB-style operators
  - Document storage via InMemoryDocstore
- Performance optimization:
  - GPU acceleration support
  - Clustering-based search
  - Dimension reduction capabilities
- Storage features:
  - Local persistence with save_local/load_local
  - Merging capability for multiple indexes
  - Full document content preservation

## Performance Monitoring

### Resource Tracking
- psutil            # System and process utilities
  - Use Case: Monitor overall system resource usage
  - Features:
    - Process memory consumption
    - CPU utilization
    - Disk I/O statistics

### Time Profiling
- time             # Built-in Python time module
  - Use Case: Basic phase timing measurement
  - Features:
    - Wall-clock time tracking
    - Simple start/end time calculation

- timeit           # Built-in timing module
  - Use Case: Function execution time measurement
  - Features:
    - Multiple execution averaging
    - Microsecond precision timing

### Performance Logging
- logging          # Built-in logging module
  - Use Case: Record performance metrics
  - Features:
    - Timing data logging
    - Resource usage logging
    - Performance report generation