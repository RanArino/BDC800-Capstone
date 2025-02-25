# core/evaluation/schema.py

"""
Schema for evaluation metrics
"""

from pydantic import BaseModel, model_validator
from typing import Optional, Literal, Dict, TypeAlias, Union

RougeType = Literal['rouge1', 'rouge2', 'rougeL']
RougeMetricType = Literal['precision', 'recall', 'fmeasure']

SelfCheckerAnswer = Literal["Yes", "No", "Undetermined"]

ProfilerTimingKey: TypeAlias = str
RankCutOff: TypeAlias = str

class StatValue(BaseModel):
    """Statistical values for a metric."""
    mean: float
    std: float
    median: float
    q1: float
    q3: float
    min: float
    max: float

class RougeMetrics(BaseModel):
    precision: Optional[Union[float, StatValue]] = None
    recall: Optional[Union[float, StatValue]] = None
    fmeasure: Optional[Union[float, StatValue]] = None

    @model_validator(mode='after')
    def check_at_least_one_metric(self) -> 'RougeMetrics':
        if all(metric is None for metric in [self.precision, self.recall, self.fmeasure]):
            raise ValueError("At least one metric (precision, recall, or fmeasure) must be provided")
        return self


class GenerationEval(BaseModel):
    rouge1: RougeMetrics
    rouge2: Optional[RougeMetrics] = None
    rougeL: Optional[RougeMetrics] = None
    bleu: Union[float, StatValue]
    cosine_sim: Union[float, StatValue]
    self_checker: Optional[SelfCheckerAnswer] = None
    self_checker_accuracy: Optional[float] = None  # For summary stats only
    
class RetrievalEval(BaseModel):
    map: Dict[RankCutOff, Union[float, StatValue]]
    mrr: Dict[RankCutOff, Union[float, StatValue]]
    hit: Dict[RankCutOff, Union[float, StatValue]]

class MetricsSummary(BaseModel):
    qa_id: str
    query: str
    ground_truth: str
    llm_answer: str
    generation: Optional[GenerationEval] = None
    retrieval: Optional[RetrievalEval] = None

class MetricsSummaryStats(BaseModel):
    """Statistical summary of metrics across multiple QA pairs."""
    total_queries: int
    processing_time: Dict[ProfilerTimingKey, float]
    memory_usage: Dict[ProfilerTimingKey, Dict[str, float]]
    generation: Optional[GenerationEval] = None
    retrieval: Optional[RetrievalEval] = None
