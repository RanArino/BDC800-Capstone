
# test/llm.py

"""
LLM test
Run the script to debug;
```
python -m pdb test/llm.py
```
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.utils import LLMController

if __name__ == "__main__":
    # Sample configuration for testing
    llm_model = "llama3.1"
    embedding_model = "huggingface-multi-qa-mpnet"

    # Initialize LLMController with the sample config
    llm_controller = LLMController(llm_model, embedding_model)

    # Test get_embedding method
    embedding_model = llm_controller.get_embedding
    print(f"Embedding Model: {embedding_model}")

    # Test get_llm method
    llm_model = llm_controller.get_llm
    print(f"LLM Model: {llm_model}")

    # Test generate_text method
    prompt = "What are the benefits of using renewable energy?"
    generated_text = llm_controller.generate_text(prompt)
    print(f"Generated Text: {generated_text}")
