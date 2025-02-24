# test/eval-selfchecker.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.evaluation.metrics import check_reasoning

"""
Self-checker evaluation test

- Run the script to debug;
    ```
    python -m pdb test/eval-selfchecker.py
    ```
- Input examples are stored in `test/input_data/qasper_list.json` and `test/input_data/RAGResponse_list.json`
- Output example is generated in `test/output_data/metrics_list.json`
"""

print("\nStart the first test...")
question1 = "What is the seed lexicon?"
answer1 = "The seed lexicon consists of 15 positive words and 15 negative words, which are used to assign polarity scores to extracted events. If the predicate of an event is in the seed lexicon and does not involve complex phenomena like negation, a corresponding polarity score (+1 for positive events and -1 for negative events) is assigned to the event."
reasoning1 = "a vocabulary of positive and negative predicates that helps determine the polarity score of an event."
print(check_reasoning(qa_id="qa1", question=question1, answer=answer1, reasoning=reasoning1))

print("\nStart the second test...")
question2 = "How big are improvements of supervszed learning results trained on smalled labeled data enhanced with proposed approach copared to basic approach?"
answer2 = "3%"
reasoning2 = "Unfortunately, the provided context does not directly answer the question regarding \"improvements\" or specific numbers associated with enhancements. However, it provides insight into a comparison between supervised and semi-supervised models when using small datasets (6,000 events) versus larger ones (0.6 million events). It states that \"Contrary to our expectations, supervised models (ACP) outperformed semi-supervised models (ACP+AL+CA+CO)\" when the dataset was 0.6 million events large.\n\nGiven this information, and interpreting the question as asking for improvements in performance or any metric resulting from applying the proposed discourse relation-based label propagation approach compared to basic methods on smaller labeled data:\n\n- The context suggests that the application of the proposed method is beneficial (\"effective\") when working with small amounts of labeled data. However, it does not provide a numerical measure of improvement.\n\nTo accurately answer the question as phrased regarding \"improvements\" specifically and quantify them, one would need more detailed performance metrics (e.g., accuracy scores, F1-scores) before and after applying the proposed approach on smaller datasets."
print(check_reasoning(qa_id="qa2", question=question2, answer=answer2, reasoning=reasoning2))
