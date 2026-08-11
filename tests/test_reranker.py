import pytest

from processors.chunking import Chunk
from processors.reranker import NoOpReranker, RerankerFactory


class TestNoOpReranker:
    def test_preserves_order_and_truncates_to_top_k(self):
        reranker = NoOpReranker()
        candidates = [Chunk(text=f"chunk {i}") for i in range(5)]

        ranked = reranker.rerank("query", candidates, top_k=3)

        assert len(ranked) == 3
        assert [r.chunk.text for r in ranked] == ["chunk 0", "chunk 1", "chunk 2"]

    def test_scores_are_descending(self):
        reranker = NoOpReranker()
        candidates = [Chunk(text=f"chunk {i}") for i in range(4)]

        ranked = reranker.rerank("query", candidates, top_k=4)

        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self):
        reranker = NoOpReranker()
        assert reranker.rerank("query", [], top_k=5) == []

    def test_is_available(self):
        assert NoOpReranker().is_available() is True


class TestRerankerFactory:
    def test_creates_noop_reranker(self):
        assert isinstance(RerankerFactory.create_reranker("none"), NoOpReranker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            RerankerFactory.create_reranker("nonexistent")
