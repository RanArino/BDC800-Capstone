# Methodology

This research implements a hierarchical retrieval architecture through two main phases: the indexing phase and the searching phase. The proposed methodology aims to create and utilize a multi-level granularity structure that enables efficient and context-aware information retrieval.

## Indexing Phase

The indexing phase establishes a hierarchical knowledge structure through the following steps:

### 1. Document-Level Processing

- Each uploaded document undergoes initial processing where a Large Language Model (LLM) generates a comprehensive summary within 1000-2000 tokens
- This summary length is optimized for compatibility with the embedding model while maintaining semantic richness

### 2. Document Segmentation and Embedding

The system processes documents through parallel operations:

- **Chunking**: 
  - Documents are segmented into smaller units (i.e., 100-300 tokens)
  - Each chunk is created with semantic awareness, ensuring complete sentences and maintaining context
  - An overlap mechanism is implemented to preserve contextual continuity between chunks
- **Document Embedding**:
  - The system generates vector representations from document summaries for high-level similarity matching

### 3. Cluster Formation

The system organizes chunks within each document into semantic clusters:

- Implements machine learning clustering algorithms to group semantically related chunks
- Creates an intermediate layer of organization between document-level and chunk-level information
- Maintains hierarchical relationships while enabling efficient retrieval

## Retrieval Phase

The retrieval process is designed to efficiently locate and extract relevant information through:

### 1. Query Processing

- Accepts and processes user queries
- Implements query expansion techniques to enhance retrieval effectiveness

### 2. Multi-Level Search Strategy

The system employs a top-down search approach:

- Conducts initial similarity search at the document level
- Identifies top-k most relevant documents
- Performs focused search within selected documents at the cluster and chunk levels
- Retrieves the most relevant chunks for final response generation
