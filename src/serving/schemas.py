"""
Pydantic request/response schemas for the FastAPI serving layer.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class AnalyzeRequest(BaseModel):
    ticker: str = "TSLA"
    benchmark: str = "SPY"
    start_date: str = "2024-10-01"
    end_date: str = "2024-12-31"
    z_threshold: float = 2.5
    use_rag: bool = True


class AnalyzeResponse(BaseModel):
    event_summary: Dict[str, Any] = Field(default_factory=dict)
    similarity_analysis: Optional[Dict[str, Any]] = None
    latest_explanation: Optional[str] = None
    multiple_explanations: Dict[str, str] = Field(default_factory=dict)


class EventPayload(BaseModel):
    """Shape matches the event rows the pipeline itself produces
    (`processors.anomaly_detector` + `News_Headlines` from
    `processors.news_retriever`) so a `/analyze` result's events can be
    fed straight into `/explain`."""

    Date: str
    Ticker: str
    Event_Type: str
    Z_score: float
    Return: Optional[float] = None
    News_Headlines: str = ""


class ExplainRequest(BaseModel):
    event: EventPayload
    historical_context: List[EventPayload] = Field(default_factory=list)


class CitationPayload(BaseModel):
    marker: str
    source_text: str
    event_index: Optional[Any] = None


class ExplainResponse(BaseModel):
    explanation: str
    citations: List[CitationPayload]
    unverified_citation_markers: List[str]
    is_fully_grounded: bool
