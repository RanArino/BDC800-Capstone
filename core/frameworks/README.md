# RAG Framework Implementations

This directory houses different implementations of the core Retrieval-Augmented Generation (RAG) framework logic. It provides the architectural backbone for retrieving relevant information and generating responses based on that information.

## Overview

The RAG frameworks defined here orchestrate the interaction between the retrieval component (finding relevant documents/chunks) and the generation component (language model producing the final answer). Different implementations might explore various strategies for retrieval, context integration, and generation prompting.

## Directory Structure

-   **`base.py`**: Defines the abstract base class (`BaseFramework` or similar) that all specific framework implementations inherit from. This ensures a consistent interface for interacting with different RAG strategies (e.g., a common `run` or `process_query` method).
-   **`simple.py`**: Provides a straightforward, possibly baseline, implementation of a RAG pipeline. This could serve as a reference or a starting point for more complex designs.
-   **`scaler.py`**: Implements a potentially more advanced or "scalable" RAG framework. The name "Scaler" might refer to specific techniques for handling large-scale retrieval, complex reasoning steps, or specific published architectures. It likely incorporates more sophisticated logic compared to `simple.py`.
-   **`scaler_v1.py`**: Appears to be a previous version or iteration of the `scaler.py` framework, possibly kept for comparison or historical reasons.
-   **`schema.py`**: Contains data classes or Pydantic models defining the structure for configuration parameters, inputs (like queries), intermediate states, and outputs (like final responses with retrieved context) used across the different framework implementations.
-   **`__init__.py`**: Makes the core framework classes (e.g., `SimpleRAG`, `ScalerRAG`) easily importable and might offer factory functions (e.g., `load_framework`) to instantiate specific framework types based on configuration.

## Key Concepts

Frameworks in this module are responsible for:

1.  **Query Processing**: Receiving a user query.
2.  **Retrieval**: Interfacing with a retriever component (defined elsewhere, likely in `core/retrievers`) to fetch relevant documents or text chunks.
3.  **Context Management**: Selecting, ranking, and formatting the retrieved context to be passed to the language model.
4.  **Prompt Engineering**: Constructing the appropriate prompt for the language model, incorporating the query and the managed context.
5.  **Generation**: Interfacing with a generator component (defined elsewhere, likely in `core/generators`) to produce the final answer.
6.  **Output Formatting**: Structuring the final output, potentially including the generated answer, retrieved sources, and other metadata as defined in `schema.py`.

## Usage

To use a framework from this module:

1.  **Instantiate**: Create an instance of a specific framework class (e.g., `SimpleRAG`, `ScalerRAG`) usually by providing configuration parameters (like which retriever and generator models to use). Factory functions in `__init__.py` might simplify this.
2.  **Run**: Call the primary execution method (e.g., `framework.run(query="...")`) with the input query.
3.  **Process Output**: Handle the structured output returned by the framework, which will contain the generated response and associated information.

These frameworks are designed to be used in conjunction with components from `core/datasets`, `core/retrievers`, `core/generators`, and evaluated using `core/evaluation`.
