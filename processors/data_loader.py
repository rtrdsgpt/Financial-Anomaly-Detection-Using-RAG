"""
Data loading processor implementing Strategy pattern.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import warnings
from abc import ABC, abstractmethod

from core.base import BaseProcessor
from core.interfaces import IDataProcessor
from core.exceptions import DataLoadError

class DataLoadStrategy(ABC):
    """Strategy interface for different data loading approaches"""
    
    @abstractmethod
    def load_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data using the specific strategy"""
        pass

class YFinanceDataStrategy(DataLoadStrategy):
    """Strategy for loading data from Yahoo Finance"""
    
    def load_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from Yahoo Finance"""
        try:
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if data.empty:
                raise DataLoadError(f"No data found for {ticker}")
            return data
        except Exception as e:
            raise DataLoadError(f"Failed to load data for {ticker}: {e}")

class DataLoader(BaseProcessor, IDataProcessor):
    """Data loader with Strategy pattern implementation"""
    
    def __init__(self, strategy: Optional[DataLoadStrategy] = None):
        super().__init__("DataLoader")
        self.strategy = strategy or YFinanceDataStrategy()
        self.logger.info("DataLoader initialized")
    
    def set_strategy(self, strategy: DataLoadStrategy) -> None:
        """Set the data loading strategy (Strategy pattern)"""
        self.strategy = strategy
        self.logger.info(f"Data loading strategy changed to {strategy.__class__.__name__}")
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the input data"""
        return self._preprocess_data(data)
    
    def load_market_data(self, ticker: str, benchmark: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load market data for ticker and benchmark"""
        try:
            self.log_info(f"Loading data for {ticker} and {benchmark}")
            self.notify_progress("data_loading", 10.0, f"Loading data for {ticker}")
            
            # Load ticker data
            ticker_data = self.strategy.load_data(ticker, start_date, end_date)
            self.notify_progress("data_loading", 30.0, f"Loaded {ticker} data")
            
            # Load benchmark data
            benchmark_data = self.strategy.load_data(benchmark, start_date, end_date)
            self.notify_progress("data_loading", 50.0, f"Loaded {benchmark} data")
            
            # Preprocess data
            processed_data = self._preprocess_data(ticker_data, benchmark_data, ticker, benchmark)
            
            self.notify_progress("data_loading", 100.0, "Data loading completed")
            self.log_info(f"Successfully loaded {len(processed_data)} days of data")
            
            return {
                'ticker_data': ticker_data,
                'benchmark_data': benchmark_data,
                'processed_data': processed_data
            }
            
        except Exception as e:
            self.log_error(f"Data loading failed: {e}", e)
            self.notify_error(e, "data_loading")
            raise DataLoadError(f"Data loading failed: {e}")
    
    def _preprocess_data(self, ticker_data: pd.DataFrame, benchmark_data: pd.DataFrame, 
                        ticker: str, benchmark: str) -> pd.DataFrame:
        """Preprocess the loaded data"""
        try:
            # Suppress warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
            np.seterr(over='ignore', under='ignore')
            
            # Find the correct close column name
            ticker_close_col = self._find_close_column(ticker_data)
            benchmark_close_col = self._find_close_column(benchmark_data)
            
            # Calculate returns
            ticker_data['Return'] = ticker_data[ticker_close_col].pct_change()
            benchmark_data['Benchmark_Return'] = benchmark_data[benchmark_close_col].pct_change()
            
            # Combine data
            combined_data = pd.DataFrame(index=ticker_data.index)
            combined_data['Ticker'] = ticker
            combined_data['Benchmark'] = benchmark
            combined_data['Return'] = ticker_data['Return']
            combined_data['Benchmark_Return'] = benchmark_data['Benchmark_Return']
            combined_data['Close'] = ticker_data[ticker_close_col]
            combined_data['Benchmark_Close'] = benchmark_data[benchmark_close_col]
            
            # Calculate additional metrics
            combined_data['Rolling_STD'] = combined_data['Return'].rolling(window=10).std()
            combined_data['Z_score'] = (combined_data['Return'] - combined_data['Return'].mean()) / combined_data['Return'].std()
            
            # Remove NaN values
            combined_data = combined_data.dropna()
            
            self.log_info(f"Data preprocessing completed. Shape: {combined_data.shape}")
            return combined_data
            
        except Exception as e:
            self.log_error(f"Data preprocessing failed: {e}", e)
            raise DataLoadError(f"Data preprocessing failed: {e}")
    
    def _find_close_column(self, data: pd.DataFrame) -> str:
        """Find the correct close column name"""
        for col in ['Close', 'Adj Close', 'close', 'adj_close']:
            if col in data.columns:
                return col
        raise DataLoadError(f"Could not find Close column. Available columns: {data.columns.tolist()}")
