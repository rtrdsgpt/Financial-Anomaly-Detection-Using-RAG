import pandas as pd
import pytest

from processors.chunking import ChunkerFactory
from processors.rag_retriever import RAGRetriever
from processors.reranker import NoOpReranker, RankedChunk, RerankerStrategy
from processors.vector_store import NumpyVectorStoreStrategy


def make_events(n=3):
    return pd.DataFrame([
        {
            "Date": f"2025-0{i + 1}-01",
            "Ticker": "TEST",
            "Event_Type": "Positive Outlier",
            "Z_score": 2.5 + i,
            "News_Headlines": f"Headline A for event {i}; Headline B for event {i}",
        }
        for i in range(n)
    ])


class ScriptedReranker(RerankerStrategy):
    """Returns a pre-scripted score on each call, one per retry attempt,
    so `retrieve_with_adaptive_retry`'s widening logic can be tested
    deterministically instead of depending on real embedding relevance."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def rerank(self, query, candidates, top_k=5):
        if not candidates:
            return []
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return [RankedChunk(chunk=candidates[0], score=score, original_rank=0)]

    def is_available(self):
        return True


@pytest.fixture
def retriever(fake_embedding_generator):
    return RAGRetriever(
        chunker=ChunkerFactory.create_chunker("headline"),
        embedding_generator=fake_embedding_generator,
        vector_store=NumpyVectorStoreStrategy(),
        reranker=NoOpReranker(),
    )


class TestBuildCorpus:
    def test_chunks_every_event_and_indexes_them(self, retriever):
        count = retriever.build_corpus(make_events(3))
        # 2 headline chunks per event x 3 events
        assert count == 6
        assert len(retriever._corpus) == 6

    def test_empty_events_yields_empty_corpus(self, retriever):
        count = retriever.build_corpus(pd.DataFrame(columns=["Date", "Ticker", "Event_Type", "Z_score", "News_Headlines"]))
        assert count == 0
        assert retriever._index is None


class TestRetrieve:
    def test_retrieve_returns_ranked_chunks_up_to_top_k(self, retriever):
        result = retriever.retrieve("query about event 1", make_events(3), top_n=6, top_k=2)
        assert len(result.candidates) <= 2
        assert result.attempts == 1

    def test_retrieve_on_empty_events_returns_no_candidates(self, retriever):
        empty = pd.DataFrame(columns=["Date", "Ticker", "Event_Type", "Z_score", "News_Headlines"])
        result = retriever.retrieve("query", empty, top_n=5, top_k=5)
        assert result.candidates == []
        assert result.max_score == 0.0


class TestAdaptiveRetry:
    def test_stops_as_soon_as_relevance_floor_met(self, fake_embedding_generator):
        reranker = ScriptedReranker(scores=[0.9])
        retriever = RAGRetriever(
            chunker=ChunkerFactory.create_chunker("headline"),
            embedding_generator=fake_embedding_generator,
            vector_store=NumpyVectorStoreStrategy(),
            reranker=reranker,
            min_relevance_score=0.5,
            max_retries=2,
        )

        result = retriever.retrieve_with_adaptive_retry(
            "query", event_window_provider=lambda window: make_events(3), initial_window=5,
        )

        assert result.attempts == 1
        assert result.max_score == 0.9
        assert reranker.calls == 1

    def test_widens_window_until_floor_met(self, fake_embedding_generator):
        reranker = ScriptedReranker(scores=[0.1, 0.1, 0.9])
        retriever = RAGRetriever(
            chunker=ChunkerFactory.create_chunker("headline"),
            embedding_generator=fake_embedding_generator,
            vector_store=NumpyVectorStoreStrategy(),
            reranker=reranker,
            min_relevance_score=0.5,
            max_retries=2,
            window_growth=2.0,
        )

        result = retriever.retrieve_with_adaptive_retry(
            "query", event_window_provider=lambda window: make_events(3), initial_window=5,
        )

        assert result.attempts == 3
        assert result.max_score == 0.9
        assert result.window_used == 5 * 2 * 2  # widened twice before the third (successful) attempt

    def test_gives_up_after_max_retries_without_looping_forever(self, fake_embedding_generator):
        reranker = ScriptedReranker(scores=[0.1, 0.1, 0.1])
        retriever = RAGRetriever(
            chunker=ChunkerFactory.create_chunker("headline"),
            embedding_generator=fake_embedding_generator,
            vector_store=NumpyVectorStoreStrategy(),
            reranker=reranker,
            min_relevance_score=0.9,
            max_retries=2,
        )

        result = retriever.retrieve_with_adaptive_retry(
            "query", event_window_provider=lambda window: make_events(3), initial_window=5,
        )

        assert result.attempts == 3  # 1 initial + 2 retries, then stop
        assert reranker.calls == 3

    def test_empty_event_window_short_circuits(self, fake_embedding_generator):
        retriever = RAGRetriever(
            chunker=ChunkerFactory.create_chunker("headline"),
            embedding_generator=fake_embedding_generator,
            vector_store=NumpyVectorStoreStrategy(),
            reranker=NoOpReranker(),
        )

        empty = pd.DataFrame(columns=["Date", "Ticker", "Event_Type", "Z_score", "News_Headlines"])
        result = retriever.retrieve_with_adaptive_retry(
            "query", event_window_provider=lambda window: empty, initial_window=5,
        )

        assert result.candidates == []
        assert result.attempts == 1
