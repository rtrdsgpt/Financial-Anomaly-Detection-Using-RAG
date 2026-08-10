"""
News retrieval processor implementing Strategy pattern and Factory pattern.
"""

import pandas as pd
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import timedelta, datetime
import time

from core.base import BaseProcessor, BaseAPIClient
from core.interfaces import INewsRetriever
from core.exceptions import NewsRetrievalError, APIClientError

class NewsRetrievalStrategy(ABC):
    """Strategy interface for different news retrieval services"""
    
    @abstractmethod
    def retrieve_news(self, ticker: str, date: datetime, window_days: int = 1) -> str:
        """Retrieve news using the specific strategy"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available"""
        pass

class FinnhubNewsStrategy(NewsRetrievalStrategy):
    """Strategy for retrieving news from Finnhub API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self._is_available = True
        self.max_articles = 5
    
    def retrieve_news(self, ticker: str, date: datetime, window_days: int = 1) -> str:
        """Retrieve news from Finnhub API"""
        try:
            start = (date - timedelta(days=window_days)).strftime('%Y-%m-%d')
            end = (date + timedelta(days=window_days)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': ticker,
                'from': start,
                'to': end,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                raise APIClientError(f"API error {response.status_code} for {ticker} on {date}")
            
            articles = response.json()
            
            if isinstance(articles, list) and articles:
                headlines = [article.get('headline', '') for article in articles[:self.max_articles]]
                headlines = [h for h in headlines if h.strip()]
                if headlines:
                    return '; '.join(headlines)
            
            # Try wider date range if no news found
            if window_days < 7:
                return self.retrieve_news(ticker, date, window_days=7)
            
            return ''
            
        except Exception as e:
            raise APIClientError(f"Failed to retrieve news from Finnhub: {e}")
    
    def is_available(self) -> bool:
        """Check if Finnhub API is available"""
        try:
            # Test API with a simple request
            url = f"{self.base_url}/quote"
            params = {'symbol': 'AAPL', 'token': self.api_key}
            response = requests.get(url, params=params, timeout=5)
            return response.status_code == 200
        except:
            return False

class YahooNewsStrategy(NewsRetrievalStrategy):
    """Strategy for retrieving news from Yahoo Finance"""
    
    def __init__(self):
        self._is_available = True
    
    def retrieve_news(self, ticker: str, date: datetime, window_days: int = 1) -> str:
        """Retrieve news from Yahoo Finance"""
        try:
            from yahoo_fin import news
            
            # Get recent news
            all_news = news.get_yf_rss(ticker)
            news_df = pd.DataFrame(all_news)
            
            if not news_df.empty and 'published_parsed' in news_df.columns:
                news_df['date'] = news_df['published_parsed'].apply(lambda x: datetime(*x[:6]))
                
                # Filter news around the event date
                start_date = date - timedelta(days=window_days)
                end_date = date + timedelta(days=window_days)
                
                filtered_news = news_df[
                    (news_df['date'] >= start_date) & 
                    (news_df['date'] <= end_date)
                ]
                
                if not filtered_news.empty:
                    headlines = filtered_news['title'].head(5).tolist()
                    return '; '.join(headlines)
            
            return ''
            
        except Exception as e:
            raise APIClientError(f"Failed to retrieve news from Yahoo: {e}")
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available"""
        try:
            from yahoo_fin import news
            # Test with a simple request
            news.get_yf_rss('AAPL')
            return True
        except:
            return False

class NewsRetriever(BaseProcessor, INewsRetriever):
    """News retriever with Strategy pattern implementation"""
    
    def __init__(self, strategy: Optional[NewsRetrievalStrategy] = None):
        super().__init__("NewsRetriever")
        self.strategy = strategy
        self.logger.info("NewsRetriever initialized")
    
    def set_strategy(self, strategy: NewsRetrievalStrategy) -> None:
        """Set the news retrieval strategy (Strategy pattern)"""
        self.strategy = strategy
        self.logger.info(f"News retrieval strategy changed to {strategy.__class__.__name__}")
    
    def retrieve_news(self, ticker: str, date: pd.Timestamp, window_days: int = 1) -> str:
        """Retrieve news for a specific ticker and date"""
        try:
            if not self.strategy:
                raise NewsRetrievalError("No news retrieval strategy set")
            
            if not self.strategy.is_available():
                raise NewsRetrievalError("News service not available")
            
            self.log_info(f"Retrieving news for {ticker} on {date}")
            news = self.strategy.retrieve_news(ticker, date, window_days)
            
            if news:
                self.log_info(f"Retrieved news for {ticker}: {len(news)} characters")
            else:
                self.log_info(f"No news found for {ticker} on {date}")
            
            return news
            
        except Exception as e:
            self.log_error(f"News retrieval failed for {ticker}: {e}", e)
            self.notify_error(e, "news_retrieval")
            raise NewsRetrievalError(f"News retrieval failed: {e}")
    
    def add_news_to_events(self, events: pd.DataFrame, ticker: str, window_days: int = 1) -> pd.DataFrame:
        """Add news headlines to market events"""
        try:
            self.log_info(f"Adding news to {len(events)} events")
            self.notify_progress("news_retrieval", 10.0, "Fetching news headlines")
            
            # Add news headlines for each event
            events['News_Headlines'] = events['Date'].apply(
                lambda d: self.retrieve_news(ticker, d, window_days)
            )
            
            # Filter out events without news
            events_with_news = events[
                events['News_Headlines'].notna() & 
                (events['News_Headlines'].str.strip() != '')
            ].copy()
            
            self.notify_progress("news_retrieval", 100.0, f"Found news for {len(events_with_news)} events")
            self.log_info(f"News retrieval completed: {len(events_with_news)} events with news")
            
            return events_with_news
            
        except Exception as e:
            self.log_error(f"Failed to add news to events: {e}", e)
            self.notify_error(e, "news_retrieval")
            raise NewsRetrievalError(f"Failed to add news to events: {e}")
    
    def is_available(self) -> bool:
        """Check if the news service is available"""
        if not self.strategy:
            return False
        return self.strategy.is_available()

class NewsRetrieverFactory:
    """Factory for creating news retrieval strategies (Factory pattern)"""
    
    @staticmethod
    def create_finnhub_retriever(api_key: str) -> NewsRetriever:
        """Create a Finnhub-based news retriever"""
        strategy = FinnhubNewsStrategy(api_key)
        return NewsRetriever(strategy)
    
    @staticmethod
    def create_yahoo_retriever() -> NewsRetriever:
        """Create a Yahoo Finance-based news retriever"""
        strategy = YahooNewsStrategy()
        return NewsRetriever(strategy)
    
    @staticmethod
    def create_retriever(service: str, api_key: Optional[str] = None) -> NewsRetriever:
        """Create a news retriever based on the service type"""
        if service.lower() == 'finnhub':
            if not api_key:
                raise ValueError("API key required for Finnhub service")
            return NewsRetrieverFactory.create_finnhub_retriever(api_key)
        elif service.lower() == 'yahoo':
            return NewsRetrieverFactory.create_yahoo_retriever()
        else:
            raise ValueError(f"Unknown news service: {service}")
