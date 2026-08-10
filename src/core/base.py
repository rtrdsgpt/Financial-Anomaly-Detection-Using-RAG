"""
Base classes implementing common functionality and design patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .interfaces import IProgressObserver, IDataRepository
from .exceptions import AnalysisError

class BaseProcessor(ABC):
    """Base class for all data processors"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._observers: List[IProgressObserver] = []
    
    def add_observer(self, observer: IProgressObserver) -> None:
        """Add a progress observer (Observer pattern)"""
        self._observers.append(observer)
    
    def remove_observer(self, observer: IProgressObserver) -> None:
        """Remove a progress observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify_progress(self, step: str, progress: float, message: str) -> None:
        """Notify all observers of progress"""
        for observer in self._observers:
            observer.update_progress(step, progress, message)
    
    def notify_error(self, error: Exception, step: str) -> None:
        """Notify all observers of errors"""
        for observer in self._observers:
            observer.on_error(error, step)
    
    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log error message"""
        if exception:
            self.logger.error(f"[{self.name}] {message}: {exception}")
        else:
            self.logger.error(f"[{self.name}] {message}")

class BaseDataRepository(IDataRepository):
    """Base implementation of data repository"""
    
    def __init__(self, base_path: str = "results"):
        self.base_path = base_path
        self.logger = logging.getLogger(f"{__name__}.DataRepository")
    
    def save_events(self, events: pd.DataFrame, filename: str) -> None:
        """Save events to storage"""
        try:
            import os
            os.makedirs(self.base_path, exist_ok=True)
            filepath = f"{self.base_path}/{filename}"
            events.to_parquet(filepath)
            self.logger.info(f"Events saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save events: {e}")
            raise AnalysisError(f"Failed to save events: {e}")
    
    def load_events(self, filename: str) -> pd.DataFrame:
        """Load events from storage"""
        try:
            filepath = f"{self.base_path}/{filename}"
            return pd.read_parquet(filepath)
        except Exception as e:
            self.logger.error(f"Failed to load events: {e}")
            raise AnalysisError(f"Failed to load events: {e}")
    
    def save_explanations(self, explanations: Dict[int, str], filename: str) -> None:
        """Save explanations to storage"""
        try:
            import os
            os.makedirs(self.base_path, exist_ok=True)
            filepath = f"{self.base_path}/{filename}"
            with open(filepath, 'w') as f:
                for event_idx, explanation in explanations.items():
                    f.write(f"=== Event {event_idx} ===\n")
                    f.write(explanation)
                    f.write("\n\n" + "="*50 + "\n\n")
            self.logger.info(f"Explanations saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save explanations: {e}")
            raise AnalysisError(f"Failed to save explanations: {e}")

class BaseAPIClient(ABC):
    """Base class for API clients with common functionality"""
    
    def __init__(self, api_key: str, name: str):
        self.api_key = api_key
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._is_available = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the API client"""
        pass
    
    def is_available(self) -> bool:
        """Check if the API client is available"""
        return self._is_available
    
    def log_api_call(self, endpoint: str, status_code: int) -> None:
        """Log API call information"""
        self.logger.info(f"API call to {endpoint}: {status_code}")

class BaseAnalysisPipeline:
    """Base class for analysis pipelines with template method pattern"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._observers: List[IProgressObserver] = []
        self._results: Dict[str, Any] = {}
    
    def add_observer(self, observer: IProgressObserver) -> None:
        """Add a progress observer"""
        self._observers.append(observer)
    
    def notify_progress(self, step: str, progress: float, message: str) -> None:
        """Notify all observers of progress"""
        for observer in self._observers:
            observer.update_progress(step, progress, message)
    
    def notify_error(self, error: Exception, step: str) -> None:
        """Notify all observers of errors"""
        for observer in self._observers:
            observer.on_error(error, step)
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Template method for running the pipeline"""
        try:
            self.logger.info(f"Starting {self.name} pipeline")
            self.notify_progress("start", 0.0, f"Starting {self.name}")
            
            # Template method - subclasses implement these steps
            self._preprocess(**kwargs)
            self._analyze(**kwargs)
            self._postprocess(**kwargs)
            
            self.notify_progress("complete", 100.0, f"{self.name} completed successfully")
            self.logger.info(f"{self.name} pipeline completed successfully")
            return self._results
            
        except Exception as e:
            self.logger.error(f"{self.name} pipeline failed: {e}")
            self.notify_error(e, "pipeline")
            raise
    
    @abstractmethod
    def _preprocess(self, **kwargs) -> None:
        """Preprocessing step - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _analyze(self, **kwargs) -> None:
        """Analysis step - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _postprocess(self, **kwargs) -> None:
        """Postprocessing step - to be implemented by subclasses"""
        pass
