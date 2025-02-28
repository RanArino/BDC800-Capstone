# core/rag_core/llm/summarizer.py

from typing import Literal
from langchain_ollama.llms import OllamaLLM


def run_doc_summary(
        document_content: str, 
        model: Literal["llama3.2:1b", "llama3.2:3b"] = "llama3.2:1b",
        max_tokens: int = 512, 
        temperature: float = 0.1
    ) -> str:
    """
    Summarize document content using Ollama LLM with llama3.2 3B model.
    
    Args:
        document_content (str): The content of the document to summarize
        max_tokens (int, optional): Maximum number of tokens in the summary. Defaults to 512.
        temperature (float, optional): Temperature for generation. Lower values make output more deterministic. Defaults to 0.1.
    
    Returns:
        str: Summarized content
    """
    # Initialize Ollama LLM with llama3.2 3B model
    llm = OllamaLLM(
        model=model,
        temperature=temperature,
        num_predict=max_tokens,
    )
    
    # Create prompt for summarization
    prompt = f"""
    Provide a concise summary of the following document within {max_tokens} tokens. 
    Focus on the main points, key findings, and important details.
    
    DOCUMENT:
    {document_content}
    
    SUMMARY:
    """
    
    # Generate summary
    summary = llm.invoke(prompt)
    
    return summary.strip()
