"""
Similarity analysis processor implementing Strategy pattern.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import faiss

from core.base import BaseProcessor
from core.interfaces import ISimilarityAnalyzer
from core.exceptions import SimilarityAnalysisError

class SimilarityStrategy(ABC):
    """Strategy interface for different similarity analysis methods"""
    
    @abstractmethod
    def create_index(self, embeddings: np.ndarray) -> Any:
        """Create a similarity index from embeddings"""
        pass
    
    @abstractmethod
    def find_similar(self, query_embedding: np.ndarray, index: Any, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar items to the query"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters"""
        pass

class FAISSStrategy(SimilarityStrategy):
    """Strategy for similarity analysis using FAISS"""
    
    def __init__(self, index_type: str = 'flat'):
        self.index_type = index_type
        self.dimension = None
        self.index = None
    
    def create_index(self, embeddings: np.ndarray) -> Any:
        """Create a FAISS index from embeddings"""
        try:
            if embeddings.size == 0:
                raise SimilarityAnalysisError("No embeddings provided")
            
            # Clean embeddings
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
            
            self.dimension = embeddings.shape[1]
            
            if self.index_type == 'flat':
                # L2 distance index
                index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == 'ivf':
                # IVF index for larger datasets
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
            
            # Add embeddings to index
            index.add(embeddings.astype('float32'))
            
            if self.index_type == 'ivf':
                index.train(embeddings.astype('float32'))
            
            self.index = index
            return index
            
        except Exception as e:
            raise SimilarityAnalysisError(f"Failed to create FAISS index: {e}")
    
    def find_similar(self, query_embedding: np.ndarray, index: Any, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar items using FAISS"""
        try:
            # Clean query embedding
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=1e6, neginf=-1e6)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search for similar items
            distances, indices = index.search(query_embedding, k)
            
            # Clean distances
            distances = np.nan_to_num(distances, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return distances[0], indices[0]
            
        except Exception as e:
            raise SimilarityAnalysisError(f"Failed to find similar items: {e}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters"""
        return {
            'index_type': self.index_type,
            'dimension': self.dimension
        }

class CosineSimilarityStrategy(SimilarityStrategy):
    """Strategy for similarity analysis using cosine similarity"""
    
    def __init__(self):
        self.embeddings = None
        self.normalized_embeddings = None
    
    def create_index(self, embeddings: np.ndarray) -> Any:
        """Create a cosine similarity index"""
        try:
            if embeddings.size == 0:
                raise SimilarityAnalysisError("No embeddings provided")
            
            # Clean embeddings
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            self.normalized_embeddings = embeddings / norms
            
            self.embeddings = embeddings
            return self.normalized_embeddings
            
        except Exception as e:
            raise SimilarityAnalysisError(f"Failed to create cosine similarity index: {e}")
    
    def find_similar(self, query_embedding: np.ndarray, index: Any, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar items using cosine similarity"""
        try:
            # Clean query embedding
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                query_norm = 1
            query_embedding = query_embedding / query_norm
            
            # Calculate cosine similarities
            similarities = np.dot(index, query_embedding)
            
            # Get top k similar items
            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_k_similarities = similarities[top_k_indices]
            
            # Convert similarities to distances (1 - similarity)
            distances = 1 - top_k_similarities
            
            return distances, top_k_indices
            
        except Exception as e:
            raise SimilarityAnalysisError(f"Failed to find similar items: {e}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters"""
        return {
            'index_type': 'cosine',
            'dimension': self.embeddings.shape[1] if self.embeddings is not None else None
        }

class SimilarityAnalyzer(BaseProcessor, ISimilarityAnalyzer):
    """Similarity analyzer with Strategy pattern implementation"""
    
    def __init__(self, strategy: Optional[SimilarityStrategy] = None):
        super().__init__("SimilarityAnalyzer")
        self.strategy = strategy or FAISSStrategy()
        self.logger.info("SimilarityAnalyzer initialized")
    
    def set_strategy(self, strategy: SimilarityStrategy) -> None:
        """Set the similarity analysis strategy (Strategy pattern)"""
        self.strategy = strategy
        self.logger.info(f"Similarity analysis strategy changed to {strategy.__class__.__name__}")
    
    def create_index(self, embeddings: np.ndarray) -> Any:
        """Create a similarity index from embeddings"""
        try:
            self.log_info(f"Creating similarity index for {len(embeddings)} embeddings")
            self.notify_progress("similarity_analysis", 10.0, "Creating similarity index")
            
            index = self.strategy.create_index(embeddings)
            
            self.notify_progress("similarity_analysis", 50.0, "Similarity index created")
            self.log_info(f"Similarity index created successfully")
            
            return index
            
        except Exception as e:
            self.log_error(f"Failed to create similarity index: {e}", e)
            self.notify_error(e, "similarity_analysis")
            raise SimilarityAnalysisError(f"Failed to create similarity index: {e}")
    
    def find_similar(self, query_embedding: np.ndarray, index: Any, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find similar items to the query"""
        try:
            self.log_info(f"Finding {k} similar items")
            
            distances, indices = self.strategy.find_similar(query_embedding, index, k)
            
            self.log_info(f"Found {len(indices)} similar items")
            return distances, indices
            
        except Exception as e:
            self.log_error(f"Failed to find similar items: {e}", e)
            raise SimilarityAnalysisError(f"Failed to find similar items: {e}")
    
    def analyze_similarity_patterns(self, events: pd.DataFrame, index: Any) -> Dict[str, Any]:
        """Analyze similarity patterns in the events"""
        try:
            self.log_info("Analyzing similarity patterns")
            self.notify_progress("similarity_analysis", 70.0, "Analyzing similarity patterns")
            
            if events.empty:
                return {
                    'total_events': 0,
                    'avg_similarity_distance': 0.0,
                    'min_similarity_distance': 0.0,
                    'max_similarity_distance': 0.0
                }
            
            # Extract embeddings
            embeddings = np.array([event['Embedding'] for _, event in events.iterrows()])
            
            # Calculate pairwise similarities
            all_distances = []
            for i, embedding in enumerate(embeddings):
                distances, _ = self.find_similar(embedding, index, k=min(len(embeddings), 5))
                # Filter finite distances
                finite_distances = distances[np.isfinite(distances)]
                all_distances.extend(finite_distances)
            
            # Calculate statistics
            if all_distances:
                analysis = {
                    'total_events': len(events),
                    'avg_similarity_distance': np.mean(all_distances),
                    'min_similarity_distance': np.min(all_distances),
                    'max_similarity_distance': np.max(all_distances)
                }
            else:
                analysis = {
                    'total_events': len(events),
                    'avg_similarity_distance': 0.0,
                    'min_similarity_distance': 0.0,
                    'max_similarity_distance': 0.0
                }
            
            self.notify_progress("similarity_analysis", 100.0, "Similarity analysis completed")
            self.log_info(f"Similarity analysis completed: {analysis}")
            
            return analysis
            
        except Exception as e:
            self.log_error(f"Failed to analyze similarity patterns: {e}", e)
            self.notify_error(e, "similarity_analysis")
            raise SimilarityAnalysisError(f"Failed to analyze similarity patterns: {e}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameters"""
        return self.strategy.get_parameters()
