# core/evaluation/schema.py

"""
Schema for evaluation metrics
"""

from pydantic import BaseModel, model_validator
from typing import Optional, Literal

RougeType = Literal['rouge1', 'rouge2', 'rougeL']
RougeMetricType = Literal['precision', 'recall', 'fmeasure']

class RougeMetrics(BaseModel):
    precision: Optional[float] = None
    recall: Optional[float] = None
    fmeasure: Optional[float] = None

    @model_validator(mode='after')
    def check_at_least_one_metric(self) -> 'RougeMetrics':
        if not any([self.precision, self.recall, self.fmeasure]):
            raise ValueError("At least one metric (precision, recall, or fmeasure) must be provided")
        return self


class GenerationEval(BaseModel):
    rouge1: RougeMetrics
    rouge2: Optional[RougeMetrics] = None
    rougeL: Optional[RougeMetrics] = None
    bleu: float
    cosine_sim: float

