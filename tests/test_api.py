import pytest
from fastapi.testclient import TestClient

from core.exceptions import AnalysisError
from processors.grounded_explainer import Citation, GroundedExplanation, GroundedGroqExplanationStrategy
from serving import api
from serving.dependencies import get_grounded_strategy


class FakeGroundedStrategy(GroundedGroqExplanationStrategy):
    def __init__(self, response: GroundedExplanation):
        self._response = response

    def explain_event_grounded(self, event_data, similar_events):
        return self._response

    def is_available(self):
        return True


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_explain_returns_grounded_explanation(client):
    fake_response = GroundedExplanation(
        explanation="Driven by strong earnings [S1]",
        citations=[Citation(marker="S1", source_text="beats EPS estimates")],
        unverified_citation_markers=[],
    )
    api.app.dependency_overrides[get_grounded_strategy] = lambda: FakeGroundedStrategy(fake_response)
    try:
        response = client.post("/explain", json={
            "event": {
                "Date": "2025-01-15", "Ticker": "TEST", "Event_Type": "Positive Outlier",
                "Z_score": 3.4, "Return": 0.09, "News_Headlines": "Beats EPS estimates by 18%",
            },
            "historical_context": [],
        })
    finally:
        api.app.dependency_overrides.pop(get_grounded_strategy, None)

    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] == "Driven by strong earnings [S1]"
    assert body["is_fully_grounded"] is True
    assert body["citations"][0]["marker"] == "S1"


def test_explain_propagates_generation_failure_as_502(client):
    class FailingStrategy(FakeGroundedStrategy):
        def explain_event_grounded(self, event_data, similar_events):
            raise RuntimeError("groq call failed")

    api.app.dependency_overrides[get_grounded_strategy] = lambda: FailingStrategy(None)
    try:
        response = client.post("/explain", json={
            "event": {"Date": "2025-01-15", "Ticker": "TEST", "Event_Type": "Positive Outlier", "Z_score": 3.4},
            "historical_context": [],
        })
    finally:
        api.app.dependency_overrides.pop(get_grounded_strategy, None)

    assert response.status_code == 502


def test_analyze_returns_400_on_analysis_error(client, monkeypatch):
    def raise_analysis_error(*args, **kwargs):
        raise AnalysisError("no data for ticker")

    monkeypatch.setattr(api.PipelineBuilder, "build", raise_analysis_error)

    response = client.post("/analyze", json={"ticker": "NOPE"})

    assert response.status_code == 400
    assert "no data for ticker" in response.json()["detail"]


def test_analyze_maps_pipeline_results_to_response(client, monkeypatch):
    class FakePipeline:
        def run(self, **kwargs):
            return {
                "event_summary": {"total_events": 2},
                "similarity_analysis": None,
                "latest_explanation": "Because X happened",
                "multiple_explanations": {0: "Explanation A", 1: "Explanation B"},
            }

    monkeypatch.setattr(api.PipelineBuilder, "build", lambda self: FakePipeline())

    response = client.post("/analyze", json={"ticker": "TEST"})

    assert response.status_code == 200
    body = response.json()
    assert body["event_summary"] == {"total_events": 2}
    assert body["latest_explanation"] == "Because X happened"
    assert body["multiple_explanations"] == {"0": "Explanation A", "1": "Explanation B"}
