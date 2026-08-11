"""
FastAPI dependency providers.

The grounded explainer strategy is expensive to construct -- it loads the
sentence-transformer embedding model and (if enabled) the cross-encoder
reranker model -- so it's built once per process and cached, not
reconstructed per request.
"""

from functools import lru_cache

from fastapi import HTTPException

from config.settings import Settings, get_settings
from processors.chunking import ChunkerFactory
from processors.embedding_generator import EmbeddingGeneratorFactory
from processors.grounded_explainer import GroundedGroqExplanationStrategy
from processors.rag_retriever import RAGRetriever
from processors.reranker import RerankerFactory
from processors.vector_store import VectorStoreFactory


def get_settings_dep() -> Settings:
    return get_settings()


@lru_cache
def _build_grounded_strategy() -> GroundedGroqExplanationStrategy:
    settings = get_settings()
    if not settings.groq_api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")

    chunker = ChunkerFactory.create_chunker(settings.chunking_strategy)
    embedding_generator = EmbeddingGeneratorFactory.create_generator('sentence_transformer')

    if settings.vector_store_backend == 'chroma':
        try:
            import chromadb  # noqa: F401
            vector_store = VectorStoreFactory.create('chroma', persist_directory=settings.chroma_persist_directory)
        except ImportError:
            vector_store = VectorStoreFactory.create('numpy')
    else:
        vector_store = VectorStoreFactory.create(settings.vector_store_backend)

    if settings.use_reranker:
        reranker = RerankerFactory.create_reranker('cross_encoder')
        if not reranker.is_available():
            reranker = RerankerFactory.create_reranker('none')
    else:
        reranker = RerankerFactory.create_reranker('none')

    retriever = RAGRetriever(
        chunker=chunker,
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        reranker=reranker,
        min_relevance_score=settings.rag_min_relevance_score,
        max_retries=settings.rag_max_retries,
    )
    return GroundedGroqExplanationStrategy(settings.groq_api_key, retriever)


def get_grounded_strategy() -> GroundedGroqExplanationStrategy:
    """Not cached itself (so a missing-key HTTPException isn't cached
    forever), but delegates to the lru_cache'd builder above."""
    return _build_grounded_strategy()
