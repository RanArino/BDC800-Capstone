# Methodology

Our novel RAG framework, Semantic Clustered Abstractive Layers for Efficient Retrieval (SCALER), comprises two main phases: the indexing phase and the searching phase. The proposed methodology aims to create and utilize a multi-level granularity structure that enables efficient and context-aware information retrieval.

## Indexing Phase

The indexing phase establishes a hierarchical knowledge structure through the following steps:

### 1. Document-Level Processing (Only Inter-Document Search)

- Each uploaded document undergoes initial processing where a Large Language Model (LLM) generates a comprehensive summary within 1000-2000 tokens
- The embedding model could be 'text-embedding-005' which has a larger input token limit of 2048 tokens with 768 dimensions
- This summary length is optimized for compatibility with the embedding model while maintaining semantic richness

### 2. Document Segmentation and Embedding (Both Inter-Document and Intra-Document Search)

The system processes documents:

- **Chunking**: 
  - Documents are segmented into smaller units (i.e., 100-300 tokens)
  - Each chunk is created with semantic awareness, ensuring complete sentences and maintaining context
  - An overlap mechanism is implemented to preserve contextual continuity between chunks
  - make sure that the order of the chunks is preserved from the original document in metadata
- **Document Embedding**:
  - The system generates vector representations from document summaries for high-level similarity matching (only inter-document search)
  - the embedding model for chunked document could be 'multi-qa-mpnet-base-cos-v1' which has a smaller input token limit of 512 tokens with 768 dimensions

### 3. Cluster Formation

The system organizes chunks within each document (for only intra-document search) and each chunk (for both inter-document and intra-document search) into semantic clusters:

- Implements machine learning clustering algorithms to group semantically related chunks (K-means & GMMs after reducing dimensions by PCA & UMAP)
- Creates an intermediate layer of organization between document-level and chunk-level information; calcuate and index the cluster centroids on vector storage.
  - for inter-document search, two levels of cluster centroids are calculated and indexed - document-level and chunk-level clusters.
  - for intra-document search, only chunk-level clusters are calculated and indexed.
- (Considering): LLM summarizes each chunk cluster used for query expansion; the use of embedding and indexing for this summary is still under consideration because the chunk cluster centroid is already a summary of the chunk.

## Retrieval Phase

The retrieval process is designed to efficiently locate and extract relevant information through:

### 1. First-Level Retrieval
- Apply the user query to the top-level cluster centroids (document-level clusters for inter-document search and chunk-level clusters for intra-document search)
- Retrieve the top-k most relevant clusters

### 2. Second-Level Retrieval
- Conduct query expansion using the summary corresponding to the retrieved cluster centroids.
- LLM generates a new query for the second-level retrieval from those summaries and an original query.
- Apply the new query to chunks within the retrieved clusters and retrieve the top-k most relevant chunks.
