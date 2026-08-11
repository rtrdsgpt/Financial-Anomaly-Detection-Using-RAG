import numpy as np
import pytest

from pipeline.pipeline_factory import PipelineBuilder, PipelineDirector
from processors.embedding_generator import EmbeddingGenerator, EmbeddingGeneratorFactory, EmbeddingStrategy
from processors.reranker import NoOpReranker, RerankerFactory


class FakeEmbeddingStrategy(EmbeddingStrategy):
    """Keeps pipeline construction offline/fast -- no real model download."""

    def generate_embeddings(self, texts):
        return np.zeros((len(texts), 4))

    def get_dimension(self):
        return 4

    def is_available(self):
        return True


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    """Pipeline construction otherwise tries to load a real
    sentence-transformers embedding model and (if use_reranker defaults to
    True) a real cross-encoder -- both real HuggingFace downloads. Keep
    these tests offline/fast; embedding and reranker behavior have their
    own dedicated test modules."""
    monkeypatch.setattr(
        EmbeddingGeneratorFactory, "create_generator",
        staticmethod(lambda service, **kwargs: EmbeddingGenerator(FakeEmbeddingStrategy())),
    )
    monkeypatch.setattr(
        RerankerFactory, "create_reranker",
        staticmethod(lambda strategy, **kwargs: NoOpReranker()),
    )


class TestPipelineBuilder:
    def test_fluent_chain_builds_expected_config(self):
        builder = (
            PipelineBuilder()
            .with_ticker("NVDA")
            .with_benchmark("QQQ")
            .with_date_range("2024-01-01", "2024-06-01")
            .with_anomaly_detection(z_threshold=3.0)
        )
        assert builder.config == {
            'ticker': 'NVDA',
            'benchmark': 'QQQ',
            'start_date': '2024-01-01',
            'end_date': '2024-06-01',
            'z_threshold': 3.0,
            'vol_window': 10,
            'vol_multiplier': 2.0,
        }

    def test_with_rag_sets_all_knobs(self):
        builder = PipelineBuilder().with_rag(
            use_rag=True, vector_store_backend='numpy', chunking_strategy='headline',
            use_reranker=False, use_grounded_citations=False, min_relevance_score=0.3, max_retries=1,
        )
        assert builder.config['use_rag'] is True
        assert builder.config['vector_store_backend'] == 'numpy'
        assert builder.config['use_reranker'] is False
        assert builder.config['rag_max_retries'] == 1

    def test_build_applies_defaults_including_rag(self):
        builder = PipelineBuilder().with_ticker("TEST").with_rag(vector_store_backend='numpy', use_reranker=False)
        pipeline = builder.build()

        assert pipeline.config['ticker'] == 'TEST'
        assert pipeline.config['use_rag'] is True
        assert pipeline.use_rag is True
        assert pipeline.rag_retriever is not None

    def test_build_respects_use_rag_false(self):
        builder = PipelineBuilder().with_ticker("TEST").with_rag(use_rag=False)
        pipeline = builder.build()

        assert pipeline.use_rag is False
        assert pipeline.rag_retriever is None


class TestPipelineDirector:
    def test_create_basic_pipeline(self):
        pipeline = PipelineDirector.create_basic_pipeline("AAPL")
        assert pipeline.config['ticker'] == 'AAPL'
        assert pipeline.config['benchmark'] == 'SPY'
