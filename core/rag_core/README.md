# RAG Core Components

This directory (`core/rag_core`) forms the foundational layer of the RAG system, providing essential building blocks for data processing, indexing, and language model interactions. These components are typically utilized and orchestrated by the higher-level framework implementations found in `core/frameworks`.

## Overview

The modules within `rag_core` handle the lower-level tasks necessary for preparing data for retrieval and interfacing with language models for generation and other tasks like summarization.

## Directory Structure

-   **`indexing/`**: Contains tools for processing and structuring data to make it suitable for efficient retrieval, particularly in vector databases.
    -   `chunker.py`: Implements logic for splitting large documents into smaller, semantically meaningful chunks. This is crucial for embedding and retrieval accuracy.
    -   `clustering.py`: Provides functionalities for grouping similar documents or chunks based on their embeddings or content. This can be used for optimizing retrieval or analyzing the corpus structure.
    -   `dim_reduction.py`: Offers methods for reducing the dimensionality of vector embeddings (e.g., using techniques like PCA or UMAP). This helps in reducing storage requirements and computational cost while potentially improving retrieval performance in some cases.
-   **`llm/`**: Manages interactions with large language models (LLMs).
    -   `controller.py`: Acts as an interface or controller for making calls to various LLMs. It might handle API interactions, manage different model configurations (e.g., temperature, max tokens), and potentially route requests.
    -   `schema.py`: Defines data structures (like Pydantic models or dataclasses) for standardizing inputs and outputs when interacting with LLMs (e.g., prompt formats, response objects).
    -   `summarizer.py`: Provides a specialized function or class using an LLM for text summarization tasks. This could be used, for example, to summarize lengthy retrieved documents before feeding them to the final generation prompt.
-   **`retrieval/`**: This subdirectory currently appears to be empty or reserved for future use. The primary implementations of specific retrieval strategies (e.g., vector search, keyword search) and interfaces to vector databases are expected to reside in the top-level `core/retrievers` directory.
-   **`__init__.py`**: Facilitates the import of classes and functions from the submodules, making them accessible to other parts of the codebase (like `core/frameworks`).

## Integration and Usage

The components within `rag_core` are generally not run standalone but are integrated into the workflows defined by the RAG frameworks in `core/frameworks`.

**Example Workflow:**

1.  **Indexing Phase (Offline)**: Documents are processed using `indexing/chunker.py`. Embeddings are generated (likely using models defined elsewhere) and potentially reduced using `indexing/dim_reduction.py`. These processed chunks and their embeddings are stored in a vector index. `indexing/clustering.py` might be used for analysis or pre-filtering.
2.  **Query Phase (Online - within a `core/frameworks` implementation)**:
    *   A query is received.
    *   A retriever (from `core/retrievers`) fetches relevant chunks from the index.
    *   (Optional) The retrieved context might be summarized or processed using `llm/summarizer.py`.
    *   The query and processed context are formatted into a prompt.
    *   The `llm/controller.py` is used to send the prompt to the configured generator LLM.
    *   The LLM's response is received and returned as part of the framework's output.
