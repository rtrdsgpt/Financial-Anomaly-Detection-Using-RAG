"""
Embedding generation processor implementing Strategy pattern.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
import warnings

from core.base import BaseProcessor
from core.interfaces import IEmbeddingGenerator
from core.exceptions import EmbeddingError

class EmbeddingStrategy(ABC):
    """Strategy interface for different embedding generation methods"""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the specific strategy"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the embedding service is available"""
        pass

class SentenceTransformerStrategy(EmbeddingStrategy):
    """Strategy for generating embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self._is_available = False
        # Try to load the model immediately
        try:
            self._load_model()
        except Exception:
            # If loading fails, we'll try again later
            pass
    
    def _load_model(self) -> None:
        """Load the embedding model"""
        try:
            # Suppress warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
            np.seterr(over='ignore', under='ignore')
            
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self._is_available = True
            
        except Exception as e:
            self._is_available = False
            raise EmbeddingError(f"Failed to load embedding model: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers"""
        try:
            if not self._is_available or not self.model:
                self._load_model()
            
            if not self.model:
                raise EmbeddingError("Embedding model not loaded")
            
            # Clean texts
            texts = [str(text).strip() for text in texts if str(text).strip()]
            if not texts:
                return np.array([]).reshape(0, self.dimension)
            
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Clip extreme values to prevent overflow
            embeddings = np.clip(embeddings, -1e6, 1e6)
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension
    
    def is_available(self) -> bool:
        """Check if the embedding service is available"""
        return self._is_available

class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    """Strategy for generating embeddings using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = 'text-embedding-ada-002'):
        self.api_key = api_key
        self.model = model
        self.dimension = 1536  # OpenAI ada-002 dimension
        self._is_available = False
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # Clean texts
            texts = [str(text).strip() for text in texts if str(text).strip()]
            if not texts:
                return np.array([]).reshape(0, self.dimension)
            
            # Generate embeddings
            response = client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = np.array([item.embedding for item in response.data])
            self._is_available = True
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate OpenAI embeddings: {e}")
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension
    
    def is_available(self) -> bool:
        """Check if the embedding service is available"""
        return self._is_available

class EmbeddingGenerator(BaseProcessor, IEmbeddingGenerator):
    """Embedding generator with Strategy pattern implementation"""
    
    def __init__(self, strategy: Optional[EmbeddingStrategy] = None):
        super().__init__("EmbeddingGenerator")
        self.strategy = strategy
        self.logger.info("EmbeddingGenerator initialized")
    
    def set_strategy(self, strategy: EmbeddingStrategy) -> None:
        """Set the embedding generation strategy (Strategy pattern)"""
        self.strategy = strategy
        self.logger.info(f"Embedding strategy changed to {strategy.__class__.__name__}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            if not self.strategy:
                raise EmbeddingError("No embedding strategy set")
            
            if not self.strategy.is_available():
                raise EmbeddingError("Embedding service not available")
            
            self.log_info(f"Generating embeddings for {len(texts)} texts")
            self.notify_progress("embedding_generation", 10.0, "Generating embeddings")
            
            embeddings = self.strategy.generate_embeddings(texts)
            
            self.notify_progress("embedding_generation", 100.0, f"Generated {len(embeddings)} embeddings")
            self.log_info(f"Embedding generation completed: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            self.log_error(f"Embedding generation failed: {e}", e)
            self.notify_error(e, "embedding_generation")
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    def add_embeddings_to_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Add embeddings to events DataFrame"""
        try:
            self.log_info(f"Adding embeddings to {len(events)} events")
            
            # Prepare texts for embedding
            texts = []
            for _, event in events.iterrows():
                # Combine event information for embedding
                text = f"{event['Event_Type']} {event['News_Headlines']}"
                texts.append(text)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to events
            events_with_embeddings = events.copy()
            events_with_embeddings['Embedding'] = [emb.tolist() for emb in embeddings]
            
            self.log_info(f"Embeddings added to {len(events_with_embeddings)} events")
            return events_with_embeddings
            
        except Exception as e:
            self.log_error(f"Failed to add embeddings to events: {e}", e)
            self.notify_error(e, "embedding_generation")
            raise EmbeddingError(f"Failed to add embeddings to events: {e}")
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        if not self.strategy:
            return 0
        return self.strategy.get_dimension()
    
    def is_available(self) -> bool:
        """Check if the embedding service is available"""
        if not self.strategy:
            return False
        return self.strategy.is_available()

class EmbeddingGeneratorFactory:
    """Factory for creating embedding generators (Factory pattern)"""
    
    @staticmethod
    def create_sentence_transformer_generator(model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingGenerator:
        """Create a sentence-transformer-based embedding generator"""
        strategy = SentenceTransformerStrategy(model_name)
        return EmbeddingGenerator(strategy)
    
    @staticmethod
    def create_openai_generator(api_key: str, model: str = 'text-embedding-ada-002') -> EmbeddingGenerator:
        """Create an OpenAI-based embedding generator"""
        strategy = OpenAIEmbeddingStrategy(api_key, model)
        return EmbeddingGenerator(strategy)
    
    @staticmethod
    def create_generator(service: str, **kwargs) -> EmbeddingGenerator:
        """Create an embedding generator based on the service type"""
        if service.lower() == 'sentence_transformer':
            model_name = kwargs.get('model_name', 'all-MiniLM-L6-v2')
            return EmbeddingGeneratorFactory.create_sentence_transformer_generator(model_name)
        elif service.lower() == 'openai':
            api_key = kwargs.get('api_key')
            model = kwargs.get('model', 'text-embedding-ada-002')
            if not api_key:
                raise ValueError("API key required for OpenAI service")
            return EmbeddingGeneratorFactory.create_openai_generator(api_key, model)
        else:
            raise ValueError(f"Unknown embedding service: {service}")
