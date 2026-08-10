"""
Chunking processor implementing Strategy pattern and Factory pattern.

Splits event news text into individually-citable chunks so retrieval and
grounded explanation operate over source text, not whole events.
"""

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.exceptions import AnalysisError


class ChunkingError(AnalysisError):
    """Exception raised when chunking fails"""
    pass


@dataclass
class Chunk:
    """A single citable unit of source text plus provenance metadata"""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class ChunkingStrategy(ABC):
    """Strategy interface for splitting event text into citable chunks"""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks, attaching the given metadata to each"""
        pass


class HeadlineChunker(ChunkingStrategy):
    """Splits the existing `'; '`-joined News_Headlines string into
    individually-citable headline chunks -- the natural boundary already
    present in the pipeline's news retrieval output."""

    def __init__(self, min_length: int = 8):
        self.min_length = min_length

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        try:
            if not text or not str(text).strip():
                return []

            metadata = metadata or {}
            headlines = [h.strip() for h in str(text).split(';')]
            headlines = [h for h in headlines if len(h) >= self.min_length]

            chunks = []
            for position, headline in enumerate(headlines):
                chunk_metadata = dict(metadata)
                chunk_metadata['chunk_position'] = position
                chunks.append(Chunk(text=headline, metadata=chunk_metadata))

            return chunks

        except Exception as e:
            raise ChunkingError(f"Headline chunking failed: {e}")


class RecursiveCharacterChunker(ChunkingStrategy):
    """Recursively splits longer free text (e.g. filings, articles) on a
    priority list of separators, falling back to a hard character split,
    with a small overlap between consecutive chunks to preserve context
    across a boundary."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", " "]

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50,
                 separators: Optional[List[str]] = None):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        try:
            if not text or not str(text).strip():
                return []

            metadata = metadata or {}
            pieces = self._split(str(text).strip(), self.separators)
            merged = self._merge_with_overlap(pieces)

            chunks = []
            for position, piece in enumerate(merged):
                chunk_metadata = dict(metadata)
                chunk_metadata['chunk_position'] = position
                chunks.append(Chunk(text=piece, metadata=chunk_metadata))

            return chunks

        except Exception as e:
            raise ChunkingError(f"Recursive character chunking failed: {e}")

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text on the first separator that keeps pieces
        under chunk_size, falling back to a hard character split."""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator, remaining_separators = separators[0], separators[1:]
        raw_parts = [p for p in re.split(re.escape(separator), text) if p]

        if len(raw_parts) <= 1:
            return self._split(text, remaining_separators)

        result: List[str] = []
        for part in raw_parts:
            if len(part) > self.chunk_size:
                result.extend(self._split(part, remaining_separators))
            else:
                result.append(part)
        return result

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Greedily pack small pieces up to chunk_size, carrying a
        character overlap forward into the next chunk."""
        if not pieces:
            return []

        merged: List[str] = []
        current = ""

        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                    overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                    current = f"{overlap} {piece}".strip() if overlap else piece
                else:
                    current = piece

        if current:
            merged.append(current)

        return merged


class ChunkerFactory:
    """Factory for creating chunking strategies (Factory pattern)"""

    @staticmethod
    def create_headline_chunker(min_length: int = 8) -> HeadlineChunker:
        return HeadlineChunker(min_length=min_length)

    @staticmethod
    def create_recursive_chunker(chunk_size: int = 400, chunk_overlap: int = 50) -> RecursiveCharacterChunker:
        return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @staticmethod
    def create_chunker(strategy: str, **kwargs) -> ChunkingStrategy:
        if strategy.lower() == 'headline':
            return ChunkerFactory.create_headline_chunker(**kwargs)
        elif strategy.lower() == 'recursive':
            return ChunkerFactory.create_recursive_chunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
