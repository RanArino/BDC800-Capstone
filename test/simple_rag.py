# test/simple_rag.py

"""
Simple RAG test
Run the script to debug;
```
python -m pdb test/simple_rag.py
```
"""

import sys
import os
import gc
import warnings
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.frameworks import SimpleRAG
from core.datasets import Qasper
from core.logger.logger import get_logger

# Set up logging
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    try:
        # Initialize RAG
        logger.info("Initializing SimpleRAG")
        simple_rag = SimpleRAG("simple_rag_01")
        
        # Load dataset
        dataset = Qasper()
        dataset.load("test")
        documents = list(dataset.get_documents())[0:1]  # Get first document
        logger.info(f"Loaded {len(documents)} documents")
        
        # Index documents
        logger.info("Indexing documents")
        simple_rag.index(documents)
        logger.info("Indexing completed")
        
        # Test retrieval and generation
        query = "What is this document about?"
        logger.info(f"Testing RAG with query: {query}")
        result = simple_rag.run(query)
        logger.info("RAG test completed")
        logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()

"""
Example output:

{
    'query': 'What is this document about?',
    'answer': (
        'This document appears to be about a research project that involves natural language processing (NLP) '
        'and sentiment analysis using machine learning models. The specific tasks mentioned include:\n\n'
        '1. Event extraction and pair extraction\n'
        '2. Sentiment analysis using the ACP Corpus\n'
        '3. Model training with different combinations of datasets and objective functions\n'
        '4. Hyperparameter optimization using the dev set\n\n'
        'The document seems to be discussing the design and implementation details of a research project, '
        'likely in the field of NLP or computational linguistics.'
    ),
    'context': [
        Document(
            id='d7e36cf3-14b3-4f29-b5f8-3ea5372f90b9',
            metadata={'chunk_id': 29, 'document_id': '1909.00694', 'start_idx': 9585, 'end_idx': 9921},
            page_content=(
                ' last clause in it. The train/dev/test split of the data is shown in Table TABREF19. '
                'The objective function for supervised training is:  where $v_i$ is the $i$-th event, '
                '$R_i$ is the reference score of $v_i$, and $N_{\\rm ACP}$ is the number of the events of the ACP Corpus. '
                'To optimize the hyperparameters, we used the dev set of the AC'
            )
        ),
        Document(
            id='966b5935-9a54-4632-adb4-194b66e5f0ae',
            metadata={'chunk_id': 28, 'document_id': '1909.00694', 'start_idx': 9330, 'end_idx': 9716},
            page_content=(
                ' positive and negative, respectively: . 作業が楽だ。 The work is easy. . 駐車場がない。 There is no parking lot. '
                'Although the ACP corpus was originally constructed in the context of sentiment analysis, '
                'we found that it could roughly be regarded as a collection of affective events. '
                'We parsed each sentence and extracted the last clause in it. The train/dev/test split of the data is shown in Table T'
            )
        ),
        Document(
            id='887812b0-c2dc-48cf-a22d-618f96e016c6',
            metadata={'chunk_id': 32, 'document_id': '1909.00694', 'start_idx': 10558, 'end_idx': 10948},
            page_content=(
                ' output is the final hidden state corresponding to the special classification tag ([CLS]). '
                'For the details of ${\\rm Encoder}$, see Sections SECREF30. We trained the model with the following four combinations '
                'of the datasets: AL, AL+CA+CO (two proposed models), ACP (supervised), and ACP+AL+CA+CO (semi-supervised). '
                'The corresponding objective functions were: $\\mathcal {L}_{\\rm AL}$, $\\math'
            )
        ),
        Document(
            id='2943fdca-e319-4ec5-a396-bc54dfb5dd34',
            metadata={'chunk_id': 23, 'document_id': '1909.00694', 'start_idx': 7912, 'end_idx': 8280},
            page_content=(
                ' into what we conventionally called clauses (mostly consecutive text chunks), each of which contained one main predicate. '
                'KNP also identified the discourse relations of event pairs if explicit discourse connectives BIBREF4 such as “ので” (because) '
                'and “のに” (in spite of) were present. We treated Cause/Reason (原因・理由) and Condition (条件) in the original tagset BIBREF15 as'
            )
        ),
        Document(
            id='c1b3378e-17c2-4029-a4dd-aa928779c1c2',
            metadata={'chunk_id': 24, 'document_id': '1909.00694', 'start_idx': 7912, 'end_idx': 8187},
            page_content=(
                '�) and Condition (条件) in the original tagset BIBREF15 as Cause and Concession (逆接) as Concession, respectively. '
                'Here is an example of event pair extraction. . 重大な失敗を犯したので、仕事をクビになった。 Because [I] made a serious mistake, [I] got fired. '
                'From this sentence, we extracted the event'
            )
        )
    ]
}

"""