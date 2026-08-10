"""
Anomaly detection processor implementing Strategy pattern.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from core.base import BaseProcessor
from core.interfaces import IAnomalyDetector
from core.exceptions import AnomalyDetectionError

class AnomalyDetectionStrategy(ABC):
    """Strategy interface for different anomaly detection algorithms"""
    
    @abstractmethod
    def detect(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Detect anomalies using the specific strategy"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters"""
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """Set the parameters"""
        pass

class ZScoreStrategy(AnomalyDetectionStrategy):
    """Strategy for Z-score based anomaly detection"""
    
    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold
        self.vol_window = 10
        self.vol_multiplier = 2.0
    
    def detect(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Detect anomalies using Z-score method"""
        events = []
        
        for idx, row in data.iterrows():
            event_type = None
            
            # Z-score based detection
            if abs(row['Z_score']) > self.z_threshold:
                if row['Z_score'] > 0:
                    event_type = 'Positive Outlier'
                else:
                    event_type = 'Negative Outlier'
            
            # Volatility spike detection
            elif row['Rolling_STD'] > 0 and abs(row['Return']) > self.vol_multiplier * row['Rolling_STD']:
                event_type = 'Volatility Spike'
            
            if event_type:
                events.append({
                    'Date': idx,
                    'Ticker': row['Ticker'],
                    'Return': row['Return'],
                    'Z_score': row['Z_score'],
                    'Rolling_STD': row['Rolling_STD'],
                    'Event_Type': event_type,
                    'Close': row['Close']
                })
        
        return pd.DataFrame(events)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'z_threshold': self.z_threshold,
            'vol_window': self.vol_window,
            'vol_multiplier': self.vol_multiplier
        }
    
    def set_parameters(self, **kwargs) -> None:
        if 'z_threshold' in kwargs:
            self.z_threshold = kwargs['z_threshold']
        if 'vol_window' in kwargs:
            self.vol_window = kwargs['vol_window']
        if 'vol_multiplier' in kwargs:
            self.vol_multiplier = kwargs['vol_multiplier']

class IsolationForestStrategy(AnomalyDetectionStrategy):
    """Strategy for Isolation Forest based anomaly detection"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = None
    
    def detect(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Prepare features
            features = data[['Return', 'Benchmark_Return', 'Rolling_STD']].fillna(0)
            
            # Fit the model
            self.model = IsolationForest(contamination=self.contamination, random_state=42)
            predictions = self.model.fit_predict(features)
            
            # Create events for anomalies
            events = []
            for idx, (_, row) in enumerate(data.iterrows()):
                if predictions[idx] == -1:  # Anomaly detected
                    event_type = 'Isolation Forest Anomaly'
                    events.append({
                        'Date': data.index[idx],
                        'Ticker': row['Ticker'],
                        'Return': row['Return'],
                        'Z_score': row.get('Z_score', 0),
                        'Rolling_STD': row.get('Rolling_STD', 0),
                        'Event_Type': event_type,
                        'Close': row['Close']
                    })
            
            return pd.DataFrame(events)
            
        except ImportError:
            raise AnomalyDetectionError("scikit-learn not available for Isolation Forest")
        except Exception as e:
            raise AnomalyDetectionError(f"Isolation Forest detection failed: {e}")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'contamination': self.contamination}
    
    def set_parameters(self, **kwargs) -> None:
        if 'contamination' in kwargs:
            self.contamination = kwargs['contamination']

class AnomalyDetector(BaseProcessor, IAnomalyDetector):
    """Anomaly detector with Strategy pattern implementation"""
    
    def __init__(self, strategy: Optional[AnomalyDetectionStrategy] = None):
        super().__init__("AnomalyDetector")
        self.strategy = strategy or ZScoreStrategy()
        self.logger.info("AnomalyDetector initialized")
    
    def set_strategy(self, strategy: AnomalyDetectionStrategy) -> None:
        """Set the anomaly detection strategy (Strategy pattern)"""
        self.strategy = strategy
        self.logger.info(f"Anomaly detection strategy changed to {strategy.__class__.__name__}")
    
    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the data"""
        try:
            self.log_info("Starting anomaly detection")
            self.notify_progress("anomaly_detection", 10.0, "Detecting anomalies")
            
            events = self.strategy.detect(data)
            
            self.notify_progress("anomaly_detection", 100.0, f"Detected {len(events)} anomalies")
            self.log_info(f"Anomaly detection completed: {len(events)} events found")
            
            return events
            
        except Exception as e:
            self.log_error(f"Anomaly detection failed: {e}", e)
            self.notify_error(e, "anomaly_detection")
            raise AnomalyDetectionError(f"Anomaly detection failed: {e}")
    
    def analyze_events(self, events: pd.DataFrame) -> Dict[str, int]:
        """Analyze the detected events"""
        try:
            if events.empty:
                return {
                    'total_events': 0,
                    'positive_outliers': 0,
                    'negative_outliers': 0,
                    'volatility_spikes': 0
                }
            
            analysis = {
                'total_events': len(events),
                'positive_outliers': len(events[events['Event_Type'] == 'Positive Outlier']),
                'negative_outliers': len(events[events['Event_Type'] == 'Negative Outlier']),
                'volatility_spikes': len(events[events['Event_Type'] == 'Volatility Spike'])
            }
            
            self.log_info(f"Event analysis completed: {analysis}")
            return analysis
            
        except Exception as e:
            self.log_error(f"Event analysis failed: {e}", e)
            raise AnomalyDetectionError(f"Event analysis failed: {e}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters"""
        return self.strategy.get_parameters()
    
    def set_parameters(self, **kwargs) -> None:
        """Set the parameters"""
        self.strategy.set_parameters(**kwargs)
        self.log_info(f"Parameters updated: {kwargs}")
