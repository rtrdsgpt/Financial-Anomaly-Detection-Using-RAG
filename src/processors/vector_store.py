"""
Vector store processor implementing Strategy pattern and Factory pattern.

Provides document-carrying alternatives to the existing FAISSStrategy
(`processors/similarity_analyzer.py`) so retrieved candidates carry their
source chunk text and metadata -- required for grounded citations. Both
strategies satisfy the existing `SimilarityStrategy` interface, so they are
drop-in replacements anywhere a `SimilarityAnalyzer` is used.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.exceptions import SimilarityAnalysisError
from processors.similarity_analyzer import SimilarityStrategy


class VectorStoreStrategy(SimilarityStrategy):
    """Strategy interface for similarity backends that also carry document
    text/metadata alongside vectors, needed for citation-grounded retrieval."""

    @abstractmethod
    def set_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Attach document text/metadata, aligned by position with the
        embeddings that will be passed to `create_index`."""
        pass

    @abstractmethod
    def get_document(self, position: int) -> Tuple[str, Dict[str, Any]]:
        """Return the (text, metadata) stored at the given position"""
        pass


class ChromaVectorStoreStrategy(VectorStoreStrategy):
    """Vector store strategy backed by a local Chroma `PersistentClient` --
    no server/DB infra needed, matching the project's deliberately-bounded
    MLOps footprint."""

    def __init__(self, collection_name: str = 'rag_chunks', persist_directory: str = 'chroma_db'):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.dimension: Optional[int] = None
        self._client = None
        self._collection = None
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

    def set_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if len(documents) != len(metadatas):
            raise SimilarityAnalysisError("documents and metadatas must be the same length")
        self._documents = documents
        self._metadatas = metadatas

    def create_index(self, embeddings: np.ndarray) -> Any:
        try:
            import chromadb

            if embeddings.size == 0:
                raise SimilarityAnalysisError("No embeddings provided")

            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
            self.dimension = embeddings.shape[1]

            self._client = chromadb.PersistentClient(path=self.persist_directory)
            # Recreate on every index build: this vector store models one
            # in-memory retrieval corpus per pipeline run, not a durable
            # cross-run store, so stale rows from a previous run must not
            # leak into the current search space.
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(self.collection_name)

            ids = [str(i) for i in range(len(embeddings))]
            documents = self._documents if len(self._documents) == len(embeddings) else ['' for _ in ids]
            metadatas = self._metadatas if len(self._metadatas) == len(embeddings) else [{} for _ in ids]
            # Chroma rejects empty metadata dicts; keep a placeholder key.
            metadatas = [m if m else {'_empty': True} for m in metadatas]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )

            return self._collection

        except SimilarityAnalysisError:
            raise
        except Exception as e:
            raise SimilarityAnalysisError(f"Failed to create Chroma index: {e}")

    def find_similar(self, query_embedding: np.ndarray, index: Any, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        try:
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=1e6, neginf=-1e6)
            results = index.query(query_embeddings=[query_embedding.tolist()], n_results=k)

            distances = np.array(results['distances'][0], dtype=float) if results['distances'] else np.array([])
            indices = np.array([int(i) for i in results['ids'][0]]) if results['ids'] else np.array([], dtype=int)

            return distances, indices

        except Exception as e:
            raise SimilarityAnalysisError(f"Failed to find similar items in Chroma: {e}")

    def get_document(self, position: int) -> Tuple[str, Dict[str, Any]]:
        if 0 <= position < len(self._documents):
            return self._documents[position], self._metadatas[position]
        return '', {}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'index_type': 'chroma',
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'dimension': self.dimension,
        }


class NumpyVectorStoreStrategy(VectorStoreStrategy):
    """Pure-numpy document-carrying vector store. Used as the offline/test
    default and as a fallback if `chromadb` is unavailable in a given
    environment -- implements the same `VectorStoreStrategy` interface so
    it's a transparent substitute, not a silent behavior change."""

    def __init__(self):
        self.dimension: Optional[int] = None
        self._embeddings: Optional[np.ndarray] = None
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

    def set_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if len(documents) != len(metadatas):
            raise SimilarityAnalysisError("documents and metadatas must be the same length")
        self._documents = documents
        self._metadatas = metadatas

    def create_index(self, embeddings: np.ndarray) -> Any:
        if embeddings.size == 0:
            raise SimilarityAnalysisError("No embeddings provided")

        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        self.dimension = embeddings.shape[1]
        self._embeddings = embeddings.astype('float32')
        return self._embeddings

    def find_similar(self, query_embedding: np.ndarray, index: Any, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        try:
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=1e6, neginf=-1e6).astype('float32')
            distances = np.linalg.norm(index - query_embedding, axis=1)
            k = min(k, len(distances))
            top_k_indices = np.argsort(distances)[:k]
            return distances[top_k_indices], top_k_indices
        except Exception as e:
            raise SimilarityAnalysisError(f"Failed to find similar items: {e}")

    def get_document(self, position: int) -> Tuple[str, Dict[str, Any]]:
        if 0 <= position < len(self._documents):
            return self._documents[position], self._metadatas[position]
        return '', {}

    def get_parameters(self) -> Dict[str, Any]:
        return {'index_type': 'numpy', 'dimension': self.dimension}


class VectorStoreFactory:
    """Factory for creating vector store / similarity strategies, giving
    parity with `NewsRetrieverFactory`/`EmbeddingGeneratorFactory`."""

    @staticmethod
    def create(backend: str, **kwargs) -> SimilarityStrategy:
        backend = backend.lower()
        if backend == 'chroma':
            return ChromaVectorStoreStrategy(**kwargs)
        elif backend == 'numpy':
            return NumpyVectorStoreStrategy()
        elif backend == 'faiss':
            from processors.similarity_analyzer import FAISSStrategy
            return FAISSStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown vector store backend: {backend}")
