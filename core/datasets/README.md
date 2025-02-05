# RAG Framework Datasets

This directory contains dataset implementations for the RAG (Retrieval-Augmented Generation) framework. The datasets are organized into two main categories based on their retrieval scope.

## Dataset Sources

### Single-Document Retrieval

These datasets focus on retrieving information from within a single document context.

#### NarrativeQA (Kocisky et al., 2017)
- **Description**: Question-answering dataset based on Wikipedia articles
- **Structure**:
  - `documents.csv`: Contains Wikipedia article links
  - `qas.csv`: Question-answer pairs with document references
- **Dataset Size**: 46,765 QA pairs
- **Source**: [NarrativeQA Repository](https://github.com/google-deepmind/narrativeqa) and [HuggingFace](https://huggingface.co/datasets/deepmind/narrativeqa)

#### Qasper (Dasigi et al., 2021)
- **Description**: Question answering dataset focused on NLP research papers
- **Key Features**:
  - 5,049 questions over 1,585 NLP papers
  - Questions based on title and abstract, answers from full text
  - Expert-curated by NLP practitioners
- **Source**: [Qasper on HuggingFace](https://huggingface.co/datasets/allenai/qasper)

### Multi-Document Retrieval

These datasets require retrieving and synthesizing information across multiple documents.

#### MultiHop-RAG (Tang et al., 2024)
- **Description**: Multi-hop question answering dataset with metadata integration
- **Key Features**:
  - 2,556 queries requiring 2-4 documents for complete answers
  - Includes document metadata
  - Real-world RAG application scenarios
- **Structure**:
  - `corpus.json`: Web article collection
  - `MultiHopRAG.json`: QA pairs with article references
- **Source**: [MultiHop-RAG Repository](https://github.com/yixuantt/MultiHop-RAG/) and [HuggingFace](https://huggingface.co/datasets/yixuantt/MultiHopRAG)

#### Fast, Fetch, and Reason (Krishna et al., 2025)
- **Description**: End-to-end RAG system evaluation dataset
- **Key Features**:
  - 824 carefully designed questions
  - Evaluates factuality, retrieval, and reasoning
  - Includes prompts, answers, and Wikipedia reference links
- **Source**: [FRAMES Benchmark on HuggingFace](https://huggingface.co/datasets/google/frames-benchmark)

### References
- Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., & Gardner, M. (2021). A dataset of information-seeking questions and answers anchored in research papers. In K. Toutanova, A. Rumshisky, L. Zettlemoyer, D. Hakkani-Tur, I. Beltagy, S. Bethard, R. Cotterell, T. Chakraborty, & Y. Zhou (Eds.), Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 4599–4610). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.naacl-main.365
- Kočiský, T., Schwarz, J., Blunsom, P., Dyer, C., Hermann, K. M., Melis, G., & Grefenstette, E. (2018). The narrativeqa reading comprehension challenge. Transactions of the Association for Computational Linguistics, 6, 317-328.
- Krishna, S., Krishna, K., Mohananey, A., Schwarcz, S., Stambler, A., Upadhyay, S., & Faruqui, M. (2024). Fact, fetch, and reason: A unified evaluation of retrieval-augmented generation. arXiv preprint arXiv:2409.12941.
- Tang, Y., & Yang, Y. (2024). Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries. arXiv preprint arXiv:2401.15391.
