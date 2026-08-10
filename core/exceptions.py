"""
Custom exceptions for the quantitative analysis framework.
"""

class AnalysisError(Exception):
    """Base exception for analysis errors"""
    pass

class DataLoadError(AnalysisError):
    """Exception raised when data loading fails"""
    pass

class AnomalyDetectionError(AnalysisError):
    """Exception raised when anomaly detection fails"""
    pass

class NewsRetrievalError(AnalysisError):
    """Exception raised when news retrieval fails"""
    pass

class EmbeddingError(AnalysisError):
    """Exception raised when embedding generation fails"""
    pass

class SimilarityAnalysisError(AnalysisError):
    """Exception raised when similarity analysis fails"""
    pass

class AIExplanationError(AnalysisError):
    """Exception raised when AI explanation generation fails"""
    pass

class APIClientError(AnalysisError):
    """Exception raised when API client operations fail"""
    pass
