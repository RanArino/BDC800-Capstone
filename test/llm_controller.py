# test/llm_controller.py

"""
LLM test
Run the script to debug;
```
python -m pdb test/llm_controller.py
```
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.rag_core.llm.controller import LLMController

if __name__ == "__main__":
    # Sample configuration for testing
    llm_model = "deepseek-r1-8b"
    embedding_model = "huggingface-multi-qa-mpnet"

    # Initialize LLMController with the sample config
    llm_controller = LLMController(llm_model, embedding_model)

    # Test get_embedding method
    embedding_model = llm_controller.get_embedding
    print(f"Embedding Model: {embedding_model}")

    # Test get_llm method
    llm_model = llm_controller.get_llm
    print(f"LLM Model: {llm_model}")

    # Test get_embedding_dim method
    embedding_dim = llm_controller.get_embedding_dim
    print(f"Embedding Dimension: {embedding_dim}")

    # Test generate_text method
    prompt = "What are the benefits of using renewable energy? Tell me one line answer."
    generated_text = llm_controller.generate_text(prompt)
    print(f"Generated Text: {generated_text}")
    
    # Test <think></think> tag removal directly
    print("\nTesting <think></think> tag removal:")
    test_text = "<think>Let me think about renewable energy benefits...</think>\nRenewable energy reduces carbon emissions, provides energy independence, and creates sustainable jobs.\n<think>I should also mention cost savings.</think>"
    print(f"Original text with <think> tags:\n{test_text}")
    
    # Use the regex directly to test tag removal
    import re
    cleaned_text = re.sub(r'<think>.*?</think>', '', test_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    print(f"\nCleaned text:\n{cleaned_text.strip()}")
