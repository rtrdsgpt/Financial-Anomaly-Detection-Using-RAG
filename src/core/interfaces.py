"""
Interfaces and abstract base classes following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class IDataProcessor(ABC):
    """Interface for data processing components"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the input data and return processed data"""
        pass

class IAnomalyDetector(ABC):
    """Interface for anomaly detection algorithms"""
    
    @abstractmethod
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the data"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters"""
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """Set the parameters"""
        pass

class INewsRetriever(ABC):
    """Interface for news retrieval services"""
    
    @abstractmethod
    def retrieve_news(self, ticker: str, date: pd.Timestamp, window_days: int = 1) -> str:
        """Retrieve news for a specific ticker and date"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the news service is available"""
        pass

class IEmbeddingGenerator(ABC):
    """Interface for embedding generation"""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        pass

class ISimilarityAnalyzer(ABC):
    """Interface for similarity analysis"""
    
    @abstractmethod
    def create_index(self, embeddings: np.ndarray) -> Any:
        """Create a similarity index from embeddings"""
        pass
    
    @abstractmethod
    def find_similar(self, query_embedding: np.ndarray, index: Any, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar items to the query"""
        pass

class IAIExplainer(ABC):
    """Interface for AI explanation generation"""
    
    @abstractmethod
    def explain_event(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> str:
        """Generate explanation for an event"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI service is available"""
        pass

class IProgressObserver(ABC):
    """Interface for progress tracking (Observer pattern)"""
    
    @abstractmethod
    def update_progress(self, step: str, progress: float, message: str) -> None:
        """Update progress information"""
        pass
    
    @abstractmethod
    def on_error(self, error: Exception, step: str) -> None:
        """Handle errors during processing"""
        pass

class IDataRepository(ABC):
    """Interface for data access (Repository pattern)"""
    
    @abstractmethod
    def save_events(self, events: pd.DataFrame, filename: str) -> None:
        """Save events to storage"""
        pass
    
    @abstractmethod
    def load_events(self, filename: str) -> pd.DataFrame:
        """Load events from storage"""
        pass
    
    @abstractmethod
    def save_explanations(self, explanations: Dict[int, str], filename: str) -> None:
        """Save explanations to storage"""
        pass
