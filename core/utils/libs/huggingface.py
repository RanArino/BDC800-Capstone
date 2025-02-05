# core/utils/libs/huggingface.py

from datasets import load_dataset

def load_hf_dataset(*args, **kwargs):
    return load_dataset(*args, **kwargs)