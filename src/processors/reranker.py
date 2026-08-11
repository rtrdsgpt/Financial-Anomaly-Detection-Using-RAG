"""
Reranking processor implementing Strategy pattern and Factory pattern.

Reorders vector-search candidates by query relevance using a cross-encoder,
which scores each (query, candidate) pair jointly rather than via cosine
distance between independently-embedded vectors -- the standard second
stage of a real RAG pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from core.exceptions import AnalysisError
from processors.chunking import Chunk


class RerankingError(AnalysisError):
    """Exception raised when reranking fails"""
    pass


@dataclass
class RankedChunk:
    """A chunk plus its relevance score and where it started before rerank"""

    chunk: Chunk
    score: float
    original_rank: int


class RerankerStrategy(ABC):
    """Strategy interface for reordering retrieved candidates by relevance"""

    @abstractmethod
    def rerank(self, query: str, candidates: List[Chunk], top_k: int = 5) -> List[RankedChunk]:
        """Score and reorder candidates, returning at most top_k"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the reranking model/service is available"""
        pass


class CrossEncoderReranker(RerankerStrategy):
    """Reranks using a `sentence-transformers` cross-encoder that jointly
    scores each (query, chunk) pair -- more accurate than reusing the
    bi-encoder cosine distance from retrieval, at the cost of one forward
    pass per candidate."""

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = None
        self._is_available = False
        try:
            self._load_model()
        except Exception:
            pass

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self._is_available = True
        except Exception as e:
            self._is_available = False
            raise RerankingError(f"Failed to load cross-encoder model: {e}")

    def rerank(self, query: str, candidates: List[Chunk], top_k: int = 5) -> List[RankedChunk]:
        try:
            if not candidates:
                return []

            if not self._is_available or not self.model:
                self._load_model()

            pairs = [(query, candidate.text) for candidate in candidates]
            scores = self.model.predict(pairs)

            ranked = [
                RankedChunk(chunk=candidate, score=float(score), original_rank=rank)
                for rank, (candidate, score) in enumerate(zip(candidates, scores))
            ]
            ranked.sort(key=lambda r: r.score, reverse=True)

            return ranked[:top_k]

        except Exception as e:
            raise RerankingError(f"Cross-encoder reranking failed: {e}")

    def is_available(self) -> bool:
        return self._is_available


class NoOpReranker(RerankerStrategy):
    """Fallback reranker: keeps retrieval order, using the (already
    relevance-sorted) candidate position as a synthetic descending score so
    downstream relevance-floor checks still behave sensibly."""

    def rerank(self, query: str, candidates: List[Chunk], top_k: int = 5) -> List[RankedChunk]:
        ranked = [
            RankedChunk(chunk=candidate, score=1.0 / (1.0 + rank), original_rank=rank)
            for rank, candidate in enumerate(candidates)
        ]
        return ranked[:top_k]

    def is_available(self) -> bool:
        return True


class RerankerFactory:
    """Factory for creating reranking strategies (Factory pattern)"""

    @staticmethod
    def create_cross_encoder_reranker(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> CrossEncoderReranker:
        return CrossEncoderReranker(model_name=model_name)

    @staticmethod
    def create_noop_reranker() -> NoOpReranker:
        return NoOpReranker()

    @staticmethod
    def create_reranker(strategy: str, **kwargs) -> RerankerStrategy:
        if strategy.lower() in ('cross_encoder', 'cross-encoder'):
            return RerankerFactory.create_cross_encoder_reranker(**kwargs)
        elif strategy.lower() in ('none', 'noop', 'no_op'):
            return RerankerFactory.create_noop_reranker()
        else:
            raise ValueError(f"Unknown reranker strategy: {strategy}")
