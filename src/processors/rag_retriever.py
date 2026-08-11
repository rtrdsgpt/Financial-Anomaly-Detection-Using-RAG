"""
RAG retriever composing chunking, embedding, vector-store, and reranking
strategies into one retrieval call.

This moves retrieval granularity from whole-event (the original
FAISS-over-event-embeddings pipeline) to source-chunk, which is what makes
grounded citations possible: a retrieved candidate now carries the exact
source text a claim can be checked against, not just a similar event row.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional
import pandas as pd

from core.base import BaseProcessor
from core.exceptions import AnalysisError
from processors.chunking import Chunk, ChunkingStrategy
from processors.embedding_generator import EmbeddingGenerator
from processors.reranker import NoOpReranker, RankedChunk, RerankerStrategy
from processors.similarity_analyzer import SimilarityStrategy


class RAGRetrievalError(AnalysisError):
    """Exception raised when RAG retrieval fails"""
    pass


@dataclass
class RetrievalResult:
    """Result of a (possibly retried) retrieval call"""

    query: str
    candidates: List[RankedChunk]
    attempts: int = 1
    window_used: Optional[int] = None

    @property
    def max_score(self) -> float:
        return max((c.score for c in self.candidates), default=0.0)


class RAGRetriever(BaseProcessor):
    """Composes `ChunkingStrategy` + `EmbeddingGenerator` +
    `SimilarityStrategy` (FAISS/Chroma/numpy) + `RerankerStrategy` into a
    single retrieval call: build a chunk corpus from historical events'
    news text, embed it, index it, retrieve top-N candidates for the
    anomaly being explained, then rerank to top-k."""

    def __init__(
        self,
        chunker: ChunkingStrategy,
        embedding_generator: EmbeddingGenerator,
        vector_store: SimilarityStrategy,
        reranker: Optional[RerankerStrategy] = None,
        min_relevance_score: float = 0.0,
        max_retries: int = 2,
        window_growth: float = 2.0,
        text_column: str = 'News_Headlines',
    ):
        super().__init__("RAGRetriever")
        self.chunker = chunker
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.reranker = reranker or NoOpReranker()
        self.min_relevance_score = min_relevance_score
        self.max_retries = max_retries
        self.window_growth = window_growth
        self.text_column = text_column

        self._corpus: List[Chunk] = []
        self._index: Any = None

    def build_corpus(self, events: pd.DataFrame) -> int:
        """Chunk every event's news text, embed the chunks, and index them.
        Returns the number of chunks indexed."""
        try:
            self.log_info(f"Building RAG corpus from {len(events)} events")

            corpus: List[Chunk] = []
            for event_index, event in events.iterrows():
                text = event.get(self.text_column, '') if hasattr(event, 'get') else event[self.text_column]
                metadata = {
                    'event_index': event_index,
                    'Date': str(event.get('Date', event_index)),
                    'Ticker': event.get('Ticker'),
                    'Event_Type': event.get('Event_Type'),
                    'Z_score': event.get('Z_score'),
                }
                corpus.extend(self.chunker.chunk(text, metadata))

            self._corpus = corpus

            if not corpus:
                self._index = None
                self.log_info("No chunks produced from events; corpus is empty")
                return 0

            texts = [chunk.text for chunk in corpus]
            embeddings = self.embedding_generator.generate_embeddings(texts)

            if hasattr(self.vector_store, 'set_documents'):
                metadatas = [chunk.metadata for chunk in corpus]
                self.vector_store.set_documents(texts, metadatas)

            self._index = self.vector_store.create_index(embeddings)

            self.log_info(f"Indexed {len(corpus)} chunks from {len(events)} events")
            return len(corpus)

        except Exception as e:
            self.log_error(f"Failed to build RAG corpus: {e}", e)
            raise RAGRetrievalError(f"Failed to build RAG corpus: {e}")

    def retrieve(self, query_text: str, events: pd.DataFrame, top_n: int = 10, top_k: int = 5) -> RetrievalResult:
        """Build the corpus from `events` and retrieve the top_k most
        relevant chunks for `query_text` in a single (non-retrying) pass."""
        try:
            self.build_corpus(events)

            if not self._corpus or self._index is None:
                return RetrievalResult(query=query_text, candidates=[], attempts=1)

            query_embedding = self.embedding_generator.generate_embeddings([query_text])
            if query_embedding.size == 0:
                return RetrievalResult(query=query_text, candidates=[], attempts=1)

            top_n = min(top_n, len(self._corpus))
            _, indices = self.vector_store.find_similar(query_embedding[0], self._index, k=top_n)

            candidates = [self._corpus[i] for i in indices if 0 <= i < len(self._corpus)]
            ranked = self.reranker.rerank(query_text, candidates, top_k=top_k)

            return RetrievalResult(query=query_text, candidates=ranked, attempts=1)

        except RAGRetrievalError:
            raise
        except Exception as e:
            self.log_error(f"Retrieval failed: {e}", e)
            raise RAGRetrievalError(f"Retrieval failed: {e}")

    def retrieve_with_adaptive_retry(
        self,
        query_text: str,
        event_window_provider: Callable[[int], pd.DataFrame],
        initial_window: int,
        top_n: int = 10,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve, and if the best reranked score falls below
        `min_relevance_score`, widen the historical lookback window and
        retry -- capped at `max_retries` extra attempts so this stays a
        bounded, deterministic loop rather than open-ended agentic retry."""
        window = initial_window
        attempt = 0

        while True:
            attempt += 1
            events = event_window_provider(window)

            if events is None or events.empty:
                return RetrievalResult(query=query_text, candidates=[], attempts=attempt, window_used=window)

            result = self.retrieve(query_text, events, top_n=top_n, top_k=top_k)
            result.attempts = attempt
            result.window_used = window

            if result.max_score >= self.min_relevance_score or attempt > self.max_retries:
                if attempt > 1:
                    self.log_info(
                        f"Adaptive retry settled after {attempt} attempt(s), "
                        f"window={window}, max_score={result.max_score:.4f}"
                    )
                return result

            self.log_info(
                f"Relevance floor not met (max_score={result.max_score:.4f} < "
                f"{self.min_relevance_score}); widening window {window} -> "
                f"{int(window * self.window_growth)} and retrying"
            )
            window = int(window * self.window_growth)
