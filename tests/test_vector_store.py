import numpy as np
import pytest

from core.exceptions import SimilarityAnalysisError
from processors.similarity_analyzer import FAISSStrategy
from processors.vector_store import NumpyVectorStoreStrategy, VectorStoreFactory


class TestNumpyVectorStoreStrategy:
    def test_create_index_and_find_similar_returns_nearest_first(self):
        store = NumpyVectorStoreStrategy()
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
        index = store.create_index(embeddings)

        distances, indices = store.find_similar(np.array([0.1, 0.0]), index, k=2)

        assert list(indices[:2]) == [0, 1]
        assert distances[0] < distances[1]

    def test_create_index_rejects_empty_embeddings(self):
        store = NumpyVectorStoreStrategy()
        with pytest.raises(SimilarityAnalysisError):
            store.create_index(np.array([]).reshape(0, 4))

    def test_set_documents_length_mismatch_raises(self):
        store = NumpyVectorStoreStrategy()
        with pytest.raises(SimilarityAnalysisError):
            store.set_documents(["a", "b"], [{}])

    def test_get_document_roundtrips(self):
        store = NumpyVectorStoreStrategy()
        store.set_documents(["headline one", "headline two"], [{"i": 0}, {"i": 1}])
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        store.create_index(embeddings)

        text, meta = store.get_document(1)
        assert text == "headline two"
        assert meta == {"i": 1}

    def test_get_document_out_of_range_returns_empty(self):
        store = NumpyVectorStoreStrategy()
        assert store.get_document(0) == ('', {})

    def test_get_parameters(self):
        store = NumpyVectorStoreStrategy()
        store.create_index(np.array([[1.0, 2.0, 3.0]]))
        params = store.get_parameters()
        assert params == {'index_type': 'numpy', 'dimension': 3}


class TestVectorStoreFactory:
    def test_creates_numpy_backend(self):
        assert isinstance(VectorStoreFactory.create('numpy'), NumpyVectorStoreStrategy)

    def test_creates_faiss_backend(self):
        assert isinstance(VectorStoreFactory.create('faiss'), FAISSStrategy)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            VectorStoreFactory.create('nonexistent')
