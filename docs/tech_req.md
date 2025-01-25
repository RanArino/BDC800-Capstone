# Technical Requirements

## System Requirements

### Hardware
- Apple Silicon (M3)
- Memory: 24GB RAM
- Storage: Sufficient space for model weights and vector database

## Core Dependencies

### RAG Framework
- langchain         # Main library for RAG implementation
- langchain-core   # Core components

### Vector Database
- chromadb        # Vector store with metadata filtering support

### LLM Integration
- google-generativeai  # For Gemini integration
- ollama             # For local LLM deployment
- python-ollama      # Python SDK for Ollama

### Document Processing
- langchain-document-loaders  # For PDF, DOCX, text file processing
- python-docx                # For Microsoft Word documents
- pypdf                      # For PDF processing

## LLM Models

### Local Models (via Ollama)
1. **Deepseek-r1**
   - Version: 8B and 14B
   - Paper: https://arxiv.org/abs/2501.12948
   - Use Case: Primary local LLM option
   - Status: Pre-downloaded

2. **Phi4**
   - Version: 14B
   - Paper: https://arxiv.org/abs/2412.08905
   - Use Case: Alternative local LLM option
   - Status: Pre-downloaded

### Cloud Models
1. **Gemini**
   - Version: 8B
   - API Integration: google-generativeai package
   - Reference: https://ai.google.dev/gemini-api/docs#python

## Vector Store Configuration

### ChromaDB
- Persistence: Local storage
- Features utilized:
  - Metadata filtering
  - Vector similarity search
  - Document embedding storage