"""
Shared pytest fixtures. Pure-function/offline logic only -- no model
loading, no GPU, no network -- so this runs anywhere (see CI).
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Makes `import core.*` / `processors.*` / etc. work even without
# `pip install -e .` first, since the packages live under src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class FakeEmbeddingGenerator:
    """Deterministic, hash-based stand-in for `EmbeddingGenerator` that
    never loads a real model -- keeps tests offline and fast while still
    exercising the actual retrieval/reranking code paths."""

    DIMENSION = 16

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.random(self.DIMENSION))
        return np.array(vectors) if vectors else np.zeros((0, self.DIMENSION))

    def get_dimension(self) -> int:
        return self.DIMENSION

    def is_available(self) -> bool:
        return True


@pytest.fixture
def fake_embedding_generator() -> FakeEmbeddingGenerator:
    return FakeEmbeddingGenerator()
