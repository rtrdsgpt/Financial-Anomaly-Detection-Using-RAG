import numpy as np
import pandas as pd
import pytest

from core.exceptions import SimilarityAnalysisError
from processors.similarity_analyzer import CosineSimilarityStrategy, FAISSStrategy, SimilarityAnalyzer


class TestFAISSStrategy:
    def test_create_index_and_find_similar(self):
        strategy = FAISSStrategy(index_type='flat')
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]], dtype='float32')
        index = strategy.create_index(embeddings)

        distances, indices = strategy.find_similar(np.array([0.1, 0.0]), index, k=2)

        assert indices[0] == 0
        assert len(indices) == 2

    def test_create_index_rejects_empty(self):
        strategy = FAISSStrategy()
        with pytest.raises(SimilarityAnalysisError):
            strategy.create_index(np.array([]))


class TestCosineSimilarityStrategy:
    def test_find_similar_ranks_by_cosine_similarity(self):
        strategy = CosineSimilarityStrategy()
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.01]])
        index = strategy.create_index(embeddings)

        distances, indices = strategy.find_similar(np.array([1.0, 0.0]), index, k=3)

        assert indices[0] == 0  # identical direction => most similar

    def test_get_parameters_before_index_created(self):
        strategy = CosineSimilarityStrategy()
        params = strategy.get_parameters()
        assert params['dimension'] is None


class TestSimilarityAnalyzer:
    def test_defaults_to_faiss_strategy(self):
        analyzer = SimilarityAnalyzer()
        assert isinstance(analyzer.strategy, FAISSStrategy)

    def test_analyze_similarity_patterns_handles_empty(self):
        analyzer = SimilarityAnalyzer()
        result = analyzer.analyze_similarity_patterns(pd.DataFrame(), index=None)
        assert result['total_events'] == 0

    def test_set_strategy_swaps_strategy(self):
        analyzer = SimilarityAnalyzer()
        cosine = CosineSimilarityStrategy()
        analyzer.set_strategy(cosine)
        assert analyzer.strategy is cosine
