import numpy as np
import pandas as pd
import pytest

from core.exceptions import EmbeddingError
from processors.embedding_generator import EmbeddingGenerator, EmbeddingGeneratorFactory, EmbeddingStrategy


class FakeEmbeddingStrategy(EmbeddingStrategy):
    """Avoids loading a real sentence-transformers model."""

    def __init__(self, dimension=4, available=True):
        self.dimension = dimension
        self._available = available

    def generate_embeddings(self, texts):
        return np.array([[float(len(t))] * self.dimension for t in texts])

    def get_dimension(self):
        return self.dimension

    def is_available(self):
        return self._available


class TestEmbeddingGenerator:
    def test_generate_embeddings_delegates_to_strategy(self):
        generator = EmbeddingGenerator(FakeEmbeddingStrategy())
        embeddings = generator.generate_embeddings(["hello", "hi"])
        assert embeddings.shape == (2, 4)

    def test_raises_if_no_strategy(self):
        generator = EmbeddingGenerator(strategy=None)
        with pytest.raises(EmbeddingError):
            generator.generate_embeddings(["hello"])

    def test_raises_if_strategy_unavailable(self):
        generator = EmbeddingGenerator(FakeEmbeddingStrategy(available=False))
        with pytest.raises(EmbeddingError):
            generator.generate_embeddings(["hello"])

    def test_add_embeddings_to_events_combines_event_type_and_headlines(self):
        generator = EmbeddingGenerator(FakeEmbeddingStrategy(dimension=2))
        events = pd.DataFrame({
            "Event_Type": ["Positive Outlier"],
            "News_Headlines": ["Some headline"],
        })

        result = generator.add_embeddings_to_events(events)

        assert 'Embedding' in result.columns
        assert len(result.iloc[0]['Embedding']) == 2

    def test_get_dimension_returns_zero_without_strategy(self):
        assert EmbeddingGenerator(strategy=None).get_dimension() == 0


class TestEmbeddingGeneratorFactory:
    def test_openai_requires_api_key(self):
        with pytest.raises(ValueError):
            EmbeddingGeneratorFactory.create_generator('openai', api_key=None)

    def test_unknown_service_raises(self):
        with pytest.raises(ValueError):
            EmbeddingGeneratorFactory.create_generator('nonexistent')
