#!/usr/bin/env python3
"""
MCP tool server exposing `detect_anomalies` and `explain_anomaly` so an
external agent (e.g. Claude Desktop) can call this project's pipeline over
stdio, per todo.md section 4.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path when run directly

from mcp.server.fastmcp import FastMCP

from config.settings import get_settings
from core.exceptions import AnalysisError
from processors.anomaly_detector import AnomalyDetector, ZScoreStrategy
from processors.data_loader import DataLoader, YFinanceDataStrategy
from serving.dependencies import _build_grounded_strategy

mcp = FastMCP("financial-anomaly-detection-rag")


@mcp.tool()
def detect_anomalies(ticker: str, benchmark: str = "SPY", start_date: str = "2024-10-01",
                      end_date: str = "2024-12-31", z_threshold: float = 2.5) -> List[Dict[str, Any]]:
    """Detect statistically anomalous return days for `ticker` against
    `benchmark` over [start_date, end_date] using Z-score + volatility-spike
    detection. Returns a list of event dicts (empty list if none found)."""
    try:
        loader = DataLoader(YFinanceDataStrategy())
        data = loader.load_market_data(ticker, benchmark, start_date, end_date)
        detector = AnomalyDetector(ZScoreStrategy(z_threshold=z_threshold))
        events = detector.detect(data['processed_data'])
    except AnalysisError as e:
        return [{"error": str(e)}]

    if events.empty:
        return []
    return events.assign(Date=events['Date'].astype(str)).to_dict('records')


@mcp.tool()
def explain_anomaly(event: Dict[str, Any], historical_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Generate a citation-grounded explanation for a single market anomaly
    event. `event` and each item of `historical_context` should have
    Date/Ticker/Event_Type/Z_score/News_Headlines (as produced by
    `detect_anomalies` plus news text)."""
    settings = get_settings()
    if not settings.groq_api_key:
        return {"error": "GROQ_API_KEY is not configured"}

    strategy = _build_grounded_strategy()
    grounded = strategy.explain_event_grounded(event, historical_context or [])

    return {
        "explanation": grounded.explanation,
        "citations": [c.model_dump() for c in grounded.citations],
        "unverified_citation_markers": grounded.unverified_citation_markers,
        "is_fully_grounded": grounded.is_fully_grounded,
    }


if __name__ == "__main__":
    mcp.run()
