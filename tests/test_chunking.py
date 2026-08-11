import pytest

from processors.chunking import ChunkerFactory, HeadlineChunker, RecursiveCharacterChunker


class TestHeadlineChunker:
    def test_splits_on_semicolon(self):
        chunker = HeadlineChunker()
        chunks = chunker.chunk("Company beats EPS estimates; Guidance raised for FY25; Shares jump")
        assert [c.text for c in chunks] == [
            "Company beats EPS estimates",
            "Guidance raised for FY25",
            "Shares jump",
        ]

    def test_attaches_metadata_and_position(self):
        chunker = HeadlineChunker()
        chunks = chunker.chunk("First headline here; Second headline here", metadata={"Ticker": "ABC"})
        assert chunks[0].metadata == {"Ticker": "ABC", "chunk_position": 0}
        assert chunks[1].metadata == {"Ticker": "ABC", "chunk_position": 1}

    def test_filters_short_fragments(self):
        chunker = HeadlineChunker(min_length=10)
        chunks = chunker.chunk("ok; A genuinely long headline about markets")
        assert len(chunks) == 1
        assert chunks[0].text == "A genuinely long headline about markets"

    def test_empty_text_returns_no_chunks(self):
        chunker = HeadlineChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk(None) == []

    def test_chunk_ids_are_unique(self):
        chunker = HeadlineChunker()
        chunks = chunker.chunk("Headline number one here; Headline number two here")
        assert chunks[0].chunk_id != chunks[1].chunk_id


class TestRecursiveCharacterChunker:
    def test_short_text_is_single_chunk(self):
        chunker = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=50)
        chunks = chunker.chunk("A short piece of text.")
        assert len(chunks) == 1
        assert chunks[0].text == "A short piece of text."

    def test_long_text_is_split_within_chunk_size(self):
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
        text = ". ".join([f"Sentence number {i} in a longer document" for i in range(10)])
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 50 + 10  # allow small overlap slack

    def test_rejects_overlap_not_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(chunk_size=50, chunk_overlap=50)

    def test_empty_text_returns_no_chunks(self):
        chunker = RecursiveCharacterChunker()
        assert chunker.chunk("   ") == []


class TestChunkerFactory:
    def test_creates_headline_chunker(self):
        chunker = ChunkerFactory.create_chunker("headline")
        assert isinstance(chunker, HeadlineChunker)

    def test_creates_recursive_chunker(self):
        chunker = ChunkerFactory.create_chunker("recursive")
        assert isinstance(chunker, RecursiveCharacterChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            ChunkerFactory.create_chunker("nonexistent")
