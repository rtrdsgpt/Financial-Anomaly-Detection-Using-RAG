import pandas as pd
import pytest

from core.exceptions import DataLoadError
from processors.data_loader import DataLoader, DataLoadStrategy


class FakeDataLoadStrategy(DataLoadStrategy):
    """Avoids any real yfinance/network call -- returns fixed OHLC data."""

    def __init__(self, close_prices):
        self.close_prices = close_prices

    def load_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.date_range(start_date, periods=len(self.close_prices))
        return pd.DataFrame({"Close": self.close_prices}, index=dates)


class TestDataLoader:
    def test_load_market_data_computes_returns_and_zscore(self):
        loader = DataLoader(FakeDataLoadStrategy([100, 101, 99, 102, 105, 103, 104, 108, 110, 109, 111, 115]))
        loader.strategy = FakeDataLoadStrategy([100, 101, 99, 102, 105, 103, 104, 108, 110, 109, 111, 115])

        result = loader.load_market_data("TEST", "BENCH", "2025-01-01", "2025-01-12")

        processed = result['processed_data']
        assert 'Return' in processed.columns
        assert 'Z_score' in processed.columns
        assert 'Rolling_STD' in processed.columns
        assert not processed.empty

    def test_find_close_column_prefers_close(self):
        loader = DataLoader(FakeDataLoadStrategy([1, 2, 3]))
        df = pd.DataFrame({"Adj Close": [1, 2], "Close": [3, 4]})
        assert loader._find_close_column(df) == "Close"

    def test_find_close_column_raises_if_missing(self):
        loader = DataLoader(FakeDataLoadStrategy([1, 2, 3]))
        with pytest.raises(DataLoadError):
            loader._find_close_column(pd.DataFrame({"Open": [1, 2]}))

    def test_set_strategy_swaps_loader(self):
        loader = DataLoader(FakeDataLoadStrategy([1, 2]))
        new_strategy = FakeDataLoadStrategy([5, 6])
        loader.set_strategy(new_strategy)
        assert loader.strategy is new_strategy
