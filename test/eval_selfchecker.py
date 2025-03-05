# test/eval-selfchecker.py

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.evaluation.metrics import check_llm_answer

"""
Self-checker evaluation test

- Run the script to debug;
    ```
    python -m pdb test/eval-selfchecker.py
    ```
- Input examples are stored in `test/input_data/qasper_list.json` and `test/input_data/RAGResponse_qasper_test.json`
- Output example is generated in `test/output_data/metrics_list.json`
"""

# Test dataset
test_cases = [
    {
        "id": "qa1",
        "question": "What is the seed lexicon?",
        "answer": "The seed lexicon consists of 15 positive words and 15 negative words, which are used to assign polarity scores to extracted events. If the predicate of an event is in the seed lexicon and does not involve complex phenomena like negation, a corresponding polarity score (+1 for positive events and -1 for negative events) is assigned to the event.",
        "reasoning": "a vocabulary of positive and negative predicates that helps determine the polarity score of an event."
    },
    {
        "id": "qa2",
        "question": "How big are improvements of supervszed learning results trained on smalled labeled data enhanced with proposed approach copared to basic approach?",
        "answer": "3%",
        "reasoning": "Unfortunately, the provided context does not directly answer the question regarding \"improvements\" or specific numbers associated with enhancements. However, it provides insight into a comparison between supervised and semi-supervised models when using small datasets (6,000 events) versus larger ones (0.6 million events). It states that \"Contrary to our expectations, supervised models (ACP) outperformed semi-supervised models (ACP+AL+CA+CO)\" when the dataset was 0.6 million events large.\n\nGiven this information, and interpreting the question as asking for improvements in performance or any metric resulting from applying the proposed discourse relation-based label propagation approach compared to basic methods on smaller labeled data:\n\n- The context suggests that the application of the proposed method is beneficial (\"effective\") when working with small amounts of labeled data. However, it does not provide a numerical measure of improvement.\n\nTo accurately answer the question as phrased regarding \"improvements\" specifically and quantify them, one would need more detailed performance metrics (e.g., accuracy scores, F1-scores) before and after applying the proposed approach on smaller datasets."
    },
    {
        "id": "qa3",
        "question": "What is the main goal of the paper?",
        "answer": "The main goal of the paper is to develop and evaluate a method for predicting polarity and intensity for sentiment analysis of text.",
        "reasoning": "The paper aims to create a new approach for sentiment analysis by predicting both polarity (positive or negative) and intensity of sentiments in textual content."
    },
    {
        "id": "qa4",
        "question": "How many datasets were used in the evaluation?",
        "answer": "Three datasets were used: SemEval-2016, SemEval-2017, and a custom Twitter dataset.",
        "reasoning": "The researchers evaluated their method using data from two SemEval competitions (2016 and 2017) plus an additional dataset they collected from Twitter."
    },
    {
        "id": "qa5",
        "question": "What programming language was used to implement the model?",
        "answer": "Python",
        "reasoning": "The model was implemented using PyTorch, which is a deep learning framework for Python."
    }
]

# Available models
models = ["phi4:14b", "deepseek-r1:14b", "gemini-2.0-flash"]

def test_model(model_name, test_case_idx=0, test_case=None):
    """Run a test with a specific model and test case"""
    if test_case is None:
        test_case = test_cases[test_case_idx % len(test_cases)]
    
    print(f"\n{'='*80}")
    print(f"Testing model: {model_name} (Test #{test_case_idx+1})")
    print(f"QA ID: {test_case['id']}")
    print(f"Question: {test_case['question']}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = check_llm_answer(
        qa_id=test_case['id'], 
        question=test_case['question'], 
        ground_truth_answer=test_case['answer'], 
        llm_answer=test_case['reasoning'],
        model=model_name
    )
    elapsed = time.time() - start_time
    
    print(f"Result: {result}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    return result

# Test each model 3 times
print("\n\nSTARTING MODEL EVALUATION TESTS")
print("-------------------------------")

# Test with specific model: phi4:14b
for i in range(3):
    test_model("phi4:14b", i)

# Test with specific model: deepseek-r1:14b
for i in range(3):
    test_model("deepseek-r1:14b", i)

# Test with specific model: gemini-2.0-flash
for i in range(3):
    test_model("gemini-2.0-flash", i)

# Test with default model (should use the one from environment variable or fallback to default)
print("\n\nTesting with default model (no model specified)")
test_case = test_cases[0]
print(f"\nQA ID: {test_case['id']}")
print(f"Question: {test_case['question']}")
result = check_llm_answer(
    qa_id=test_case['id'], 
    question=test_case['question'], 
    ground_truth_answer=test_case['answer'], 
    llm_answer=test_case['reasoning']
)
print(f"Result (using default model): {result}")

print("\nALL TESTS COMPLETED")
