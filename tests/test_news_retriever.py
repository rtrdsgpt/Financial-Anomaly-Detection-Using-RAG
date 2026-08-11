import pandas as pd
import pytest

from core.exceptions import NewsRetrievalError
from processors.news_retriever import NewsRetrievalStrategy, NewsRetriever, NewsRetrieverFactory


class FakeNewsStrategy(NewsRetrievalStrategy):
    def __init__(self, headlines_by_index=None, available=True):
        self.headlines_by_index = headlines_by_index or {}
        self._available = available

    def retrieve_news(self, ticker, date, window_days=1):
        return self.headlines_by_index.get(date, '')

    def is_available(self):
        return self._available


def make_events():
    dates = pd.date_range("2025-01-01", periods=3)
    return pd.DataFrame({"Date": dates, "Event_Type": ["Positive Outlier"] * 3}, index=dates)


class TestNewsRetriever:
    def test_add_news_to_events_filters_out_events_without_news(self):
        events = make_events()
        headlines = {events.index[0]: "Some real headline", events.index[1]: "", events.index[2]: "Another headline"}
        retriever = NewsRetriever(FakeNewsStrategy(headlines))

        result = retriever.add_news_to_events(events, "TEST")

        assert len(result) == 2
        assert set(result['News_Headlines']) == {"Some real headline", "Another headline"}

    def test_retrieve_news_raises_if_no_strategy_set(self):
        retriever = NewsRetriever(strategy=None)
        with pytest.raises(NewsRetrievalError):
            retriever.retrieve_news("TEST", pd.Timestamp("2025-01-01"))

    def test_retrieve_news_raises_if_service_unavailable(self):
        retriever = NewsRetriever(FakeNewsStrategy(available=False))
        with pytest.raises(NewsRetrievalError):
            retriever.retrieve_news("TEST", pd.Timestamp("2025-01-01"))

    def test_is_available_reflects_strategy(self):
        assert NewsRetriever(FakeNewsStrategy(available=True)).is_available() is True
        assert NewsRetriever(FakeNewsStrategy(available=False)).is_available() is False
        assert NewsRetriever(strategy=None).is_available() is False


class TestNewsRetrieverFactory:
    def test_finnhub_requires_api_key(self):
        with pytest.raises(ValueError):
            NewsRetrieverFactory.create_retriever('finnhub', api_key=None)

    def test_yahoo_does_not_require_api_key(self):
        retriever = NewsRetrieverFactory.create_retriever('yahoo')
        assert retriever.strategy is not None

    def test_unknown_service_raises(self):
        with pytest.raises(ValueError):
            NewsRetrieverFactory.create_retriever('nonexistent')
