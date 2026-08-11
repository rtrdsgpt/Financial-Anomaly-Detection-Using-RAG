import json
import types

import pandas as pd
import pytest

from processors.chunking import ChunkerFactory
from processors.grounded_explainer import Citation, CitationVerifier, GroundedExplanation, GroundedGroqExplanationStrategy
from processors.rag_retriever import RAGRetriever
from processors.reranker import NoOpReranker
from processors.vector_store import NumpyVectorStoreStrategy


class TestCitationVerifier:
    def test_verified_citation_passes(self):
        from processors.chunking import Chunk
        from processors.reranker import RankedChunk

        sources = [RankedChunk(chunk=Chunk(text="Company beats Q3 EPS estimates by 18%"), score=1.0, original_rank=0)]
        explanation = GroundedExplanation(
            explanation="Beat driven by strong demand [S1]",
            citations=[Citation(marker="S1", source_text="beats Q3 EPS estimates by 18%")],
        )

        result = CitationVerifier().verify(explanation, sources)

        assert result.unverified_citation_markers == []
        assert result.is_fully_grounded is True

    def test_unverified_citation_is_flagged(self):
        from processors.chunking import Chunk
        from processors.reranker import RankedChunk

        sources = [RankedChunk(chunk=Chunk(text="Company beats Q3 EPS estimates by 18%"), score=1.0, original_rank=0)]
        explanation = GroundedExplanation(
            explanation="Beat driven by a fabricated claim [S1]",
            citations=[Citation(marker="S1", source_text="this text does not appear anywhere in the sources")],
        )

        result = CitationVerifier().verify(explanation, sources)

        assert result.unverified_citation_markers == ["S1"]
        assert result.is_fully_grounded is False

    def test_no_citations_is_trivially_grounded(self):
        explanation = GroundedExplanation(explanation="No citations here", citations=[])
        result = CitationVerifier().verify(explanation, [])
        assert result.is_fully_grounded is True


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        message = types.SimpleNamespace(content=self._content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


class _FakeGroqClient:
    def __init__(self, content: str):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(content))


@pytest.fixture
def retriever(fake_embedding_generator):
    return RAGRetriever(
        chunker=ChunkerFactory.create_chunker("headline"),
        embedding_generator=fake_embedding_generator,
        vector_store=NumpyVectorStoreStrategy(),
        reranker=NoOpReranker(),
    )


def make_strategy(retriever, fake_response_json: dict) -> GroundedGroqExplanationStrategy:
    strategy = GroundedGroqExplanationStrategy.__new__(GroundedGroqExplanationStrategy)
    strategy.api_key = "test-key"
    strategy.retriever = retriever
    strategy.model = "test-model"
    strategy.top_k_sources = 5
    strategy.client = _FakeGroqClient(json.dumps(fake_response_json))
    strategy._is_available = True
    return strategy


class TestGroundedGroqExplanationStrategy:
    def test_explain_event_grounded_parses_and_verifies_citations(self, retriever):
        event = {"Date": "2025-01-15", "Event_Type": "Positive Outlier", "Z_score": 3.4,
                  "Return": 0.09, "News_Headlines": "Beats EPS estimates by 18%"}
        history = pd.DataFrame([
            {"Date": "2024-10-16", "Ticker": "NSTR", "Event_Type": "Positive Outlier", "Z_score": 2.9,
             "News_Headlines": "Cloud bookings accelerate sharply this quarter"},
        ]).to_dict('records')

        strategy = make_strategy(retriever, {
            "explanation": "Driven by cloud momentum [S1]",
            "citations": [{"marker": "S1", "source_text": "Cloud bookings accelerate sharply this quarter"}],
        })

        grounded = strategy.explain_event_grounded(event, history)

        assert grounded.explanation == "Driven by cloud momentum [S1]"
        assert grounded.unverified_citation_markers == []

    def test_explain_event_returns_rendered_text_with_unverified_flag(self, retriever):
        event = {"Date": "2025-01-15", "Event_Type": "Positive Outlier", "Z_score": 3.4,
                  "Return": 0.09, "News_Headlines": "Beats EPS estimates"}
        history = pd.DataFrame([
            {"Date": "2024-10-16", "Ticker": "NSTR", "Event_Type": "Positive Outlier", "Z_score": 2.9,
             "News_Headlines": "Some unrelated headline text here"},
        ]).to_dict('records')

        strategy = make_strategy(retriever, {
            "explanation": "Fabricated reasoning [S1]",
            "citations": [{"marker": "S1", "source_text": "this quote is not in any source"}],
        })

        rendered = strategy.explain_event(event, history)

        assert "Fabricated reasoning [S1]" in rendered
        assert "unverified citations: S1" in rendered

    def test_no_history_still_produces_explanation(self, retriever):
        event = {"Date": "2025-01-15", "Event_Type": "Positive Outlier", "Z_score": 3.4,
                  "Return": 0.09, "News_Headlines": "Beats EPS estimates"}

        strategy = make_strategy(retriever, {"explanation": "No sources available", "citations": []})

        grounded = strategy.explain_event_grounded(event, [])

        assert grounded.explanation == "No sources available"
        assert grounded.is_fully_grounded is True
