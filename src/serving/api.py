"""
FastAPI wrapper around the existing pipeline/Strategy interfaces -- a new
consumer of `PipelineBuilder` and the grounded explainer, not a parallel
implementation (todo.md section 3).
"""

from fastapi import Depends, FastAPI, HTTPException

from core.exceptions import AnalysisError
from pipeline.pipeline_factory import PipelineBuilder
from processors.grounded_explainer import GroundedGroqExplanationStrategy

from .dependencies import get_grounded_strategy, get_settings_dep
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CitationPayload,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
)

app = FastAPI(
    title="Financial Anomaly Detection RAG API",
    description="Grounded RAG pipeline for explaining financial market anomalies.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, settings=Depends(get_settings_dep)) -> AnalyzeResponse:
    """Run the full pipeline (data load -> anomaly detection -> news ->
    RAG retrieval -> grounded explanation) for a ticker/date range."""
    try:
        builder = (
            PipelineBuilder()
            .with_ticker(request.ticker)
            .with_benchmark(request.benchmark)
            .with_date_range(request.start_date, request.end_date)
            .with_anomaly_detection(z_threshold=request.z_threshold)
            .with_rag(use_rag=request.use_rag)
        )
        if settings.finnhub_api_key:
            builder.with_news_service('finnhub', settings.finnhub_api_key)
        else:
            builder.with_news_service('yahoo')
        if settings.groq_api_key:
            builder.with_ai_service('groq', settings.groq_api_key)

        pipeline = builder.build()
        results = pipeline.run()
    except AnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))

    multiple_explanations = results.get('multiple_explanations') or {}
    return AnalyzeResponse(
        event_summary=results.get('event_summary', {}),
        similarity_analysis=results.get('similarity_analysis'),
        latest_explanation=results.get('latest_explanation'),
        multiple_explanations={str(k): v for k, v in multiple_explanations.items()},
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    strategy: GroundedGroqExplanationStrategy = Depends(get_grounded_strategy),
) -> ExplainResponse:
    """Ad-hoc grounded explanation for a single anomaly, given its own
    event data plus a pool of historical events to retrieve from --
    doesn't require running the full pipeline first."""
    event_data = request.event.model_dump()
    history = [h.model_dump() for h in request.historical_context]

    try:
        grounded = strategy.explain_event_grounded(event_data, history)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Explanation generation failed: {e}")

    return ExplainResponse(
        explanation=grounded.explanation,
        citations=[CitationPayload(**c.model_dump()) for c in grounded.citations],
        unverified_citation_markers=grounded.unverified_citation_markers,
        is_fully_grounded=grounded.is_fully_grounded,
    )
