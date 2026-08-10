"""
AI explanation processor implementing Strategy pattern and Factory pattern.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.base import BaseProcessor, BaseAPIClient
from core.interfaces import IAIExplainer
from core.exceptions import AIExplanationError, APIClientError
from .similarity_analyzer import SimilarityAnalyzer

class AIExplanationStrategy(ABC):
    """Strategy interface for different AI explanation services"""
    
    @abstractmethod
    def explain_event(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> str:
        """Generate explanation for an event"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the AI service is available"""
        pass

class GroqExplanationStrategy(AIExplanationStrategy):
    """Strategy for generating explanations using Groq API"""
    
    def __init__(self, api_key: str, model: str = 'meta-llama/llama-4-scout-17b-16e-instruct'):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._is_available = False
        # Try to initialize the client immediately
        try:
            self._initialize_client()
        except Exception:
            # If initialization fails, we'll try again later
            pass
    
    def _initialize_client(self) -> None:
        """Initialize the Groq client"""
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self._is_available = True
        except Exception as e:
            self._is_available = False
            raise APIClientError(f"Failed to initialize Groq client: {e}")
    
    def explain_event(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> str:
        """Generate explanation using Groq API"""
        try:
            if not self._is_available:
                self._initialize_client()
            
            if not self.client:
                raise APIClientError("Groq client not initialized")
            
            # Create explanation prompt
            prompt = self._create_explanation_prompt(event_data, similar_events)
            
            # Get AI explanation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise APIClientError(f"Failed to generate Groq explanation: {e}")
    
    def _create_explanation_prompt(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> str:
        """Create a prompt for explaining a market event"""
        # Format similar events context
        context_parts = []
        for event in similar_events:
            context_parts.append(
                f"Date: {event['Date']}, Event: {event['Event_Type']}, "
                f"Z-score: {event['Z_score']:.2f}, News: {event['News_Headlines'][:200]}..."
            )
        
        context = "\n\n".join(context_parts)
        
        # Create the main prompt
        prompt = f"""You are a financial analyst assistant.
Based on the following historical news and events similar to today's anomaly, explain what might be causing this market movement.

Similar Historical Events:
{context}

Today's Event Details:
- Date: {event_data['Date']}
- Event Type: {event_data['Event_Type']}
- Z-score: {event_data['Z_score']:.2f}
- Return: {event_data['Return']:.4f}
- News: {event_data['News_Headlines']}

Please provide a comprehensive explanation of what might be causing this market movement, considering:
1. The historical patterns from similar events
2. The specific news and context
3. Potential market drivers and sentiment factors
4. Risk factors and implications

Explanation:"""
        
        return prompt
    
    def is_available(self) -> bool:
        """Check if the Groq service is available"""
        return self._is_available

class OpenAIExplanationStrategy(AIExplanationStrategy):
    """Strategy for generating explanations using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo'):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._is_available = False
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self._is_available = True
        except Exception as e:
            self._is_available = False
            raise APIClientError(f"Failed to initialize OpenAI client: {e}")
    
    def explain_event(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> str:
        """Generate explanation using OpenAI API"""
        try:
            if not self._is_available:
                self._initialize_client()
            
            if not self.client:
                raise APIClientError("OpenAI client not initialized")
            
            # Create explanation prompt
            prompt = self._create_explanation_prompt(event_data, similar_events)
            
            # Get AI explanation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise APIClientError(f"Failed to generate OpenAI explanation: {e}")
    
    def _create_explanation_prompt(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> str:
        """Create a prompt for explaining a market event"""
        # Format similar events context
        context_parts = []
        for event in similar_events:
            context_parts.append(
                f"Date: {event['Date']}, Event: {event['Event_Type']}, "
                f"Z-score: {event['Z_score']:.2f}, News: {event['News_Headlines'][:200]}..."
            )
        
        context = "\n\n".join(context_parts)
        
        # Create the main prompt
        prompt = f"""You are a financial analyst assistant.
Based on the following historical news and events similar to today's anomaly, explain what might be causing this market movement.

Similar Historical Events:
{context}

Today's Event Details:
- Date: {event_data['Date']}
- Event Type: {event_data['Event_Type']}
- Z-score: {event_data['Z_score']:.2f}
- Return: {event_data['Return']:.4f}
- News: {event_data['News_Headlines']}

Please provide a comprehensive explanation of what might be causing this market movement, considering:
1. The historical patterns from similar events
2. The specific news and context
3. Potential market drivers and sentiment factors
4. Risk factors and implications

Explanation:"""
        
        return prompt
    
    def is_available(self) -> bool:
        """Check if the OpenAI service is available"""
        return self._is_available

class AIExplainer(BaseProcessor, IAIExplainer):
    """AI explainer with Strategy pattern implementation"""
    
    def __init__(self, strategy: Optional[AIExplanationStrategy] = None):
        super().__init__("AIExplainer")
        self.strategy = strategy
        self.logger.info("AIExplainer initialized")
    
    def set_strategy(self, strategy: AIExplanationStrategy) -> None:
        """Set the AI explanation strategy (Strategy pattern)"""
        self.strategy = strategy
        self.logger.info(f"AI explanation strategy changed to {strategy.__class__.__name__}")
    
    def explain_event(self, event_data: Dict[str, Any], similar_events: List[Dict[str, Any]]) -> str:
        """Generate explanation for an event"""
        try:
            if not self.strategy:
                raise AIExplanationError("No AI explanation strategy set")
            
            if not self.strategy.is_available():
                raise AIExplanationError("AI service not available")
            
            self.log_info(f"Generating explanation for event on {event_data['Date']}")
            self.notify_progress("ai_explanation", 10.0, "Generating AI explanation")
            
            explanation = self.strategy.explain_event(event_data, similar_events)
            
            self.notify_progress("ai_explanation", 100.0, "AI explanation generated")
            self.log_info(f"AI explanation generated: {len(explanation)} characters")
            
            return explanation
            
        except Exception as e:
            self.log_error(f"AI explanation failed: {e}", e)
            self.notify_error(e, "ai_explanation")
            raise AIExplanationError(f"AI explanation failed: {e}")
    
    def explain_latest_event(self, events: pd.DataFrame, similarity_analyzer, index: Any) -> str:
        """Explain the latest market event"""
        try:
            if events.empty:
                return "No events available to explain"
            
            # Get the latest event
            latest_event = events.iloc[-1].to_dict()
            
            # Find similar events
            from processors.similarity_analyzer import SimilarityAnalyzer
            if isinstance(similarity_analyzer, SimilarityAnalyzer):
                distances, indices = similarity_analyzer.find_similar(
                    np.array(latest_event['Embedding']), index, k=5
                )
                
                # Get similar events data
                similar_events = []
                for idx in indices:
                    if idx < len(events):
                        similar_events.append(events.iloc[idx].to_dict())
                
                return self.explain_event(latest_event, similar_events)
            else:
                return "Similarity analyzer not available"
                
        except Exception as e:
            self.log_error(f"Failed to explain latest event: {e}", e)
            raise AIExplanationError(f"Failed to explain latest event: {e}")
    
    def explain_multiple_events(self, events: pd.DataFrame, similarity_analyzer, index: Any, 
                               event_indices: Optional[List[int]] = None) -> Dict[int, str]:
        """Explain multiple market events"""
        try:
            if event_indices is None:
                # Default to last 3 events
                event_indices = list(range(max(0, len(events) - 3), len(events)))
            
            explanations = {}
            
            for event_idx in event_indices:
                if 0 <= event_idx < len(events):
                    event_data = events.iloc[event_idx].to_dict()
                    
                    # Find similar events
                    if isinstance(similarity_analyzer, SimilarityAnalyzer):
                        distances, indices = similarity_analyzer.find_similar(
                            np.array(event_data['Embedding']), index, k=5
                        )
                        
                        # Get similar events data
                        similar_events = []
                        for idx in indices:
                            if idx < len(events):
                                similar_events.append(events.iloc[idx].to_dict())
                        
                        explanation = self.explain_event(event_data, similar_events)
                        explanations[event_idx] = explanation
                    else:
                        explanations[event_idx] = "Similarity analyzer not available"
                else:
                    explanations[event_idx] = f"Invalid event index: {event_idx}"
            
            self.log_info(f"Generated explanations for {len(explanations)} events")
            return explanations
            
        except Exception as e:
            self.log_error(f"Failed to explain multiple events: {e}", e)
            raise AIExplanationError(f"Failed to explain multiple events: {e}")
    
    def is_available(self) -> bool:
        """Check if the AI service is available"""
        if not self.strategy:
            return False
        return self.strategy.is_available()

class AIExplainerFactory:
    """Factory for creating AI explainers (Factory pattern)"""
    
    @staticmethod
    def create_groq_explainer(api_key: str, model: str = 'meta-llama/llama-4-scout-17b-16e-instruct') -> AIExplainer:
        """Create a Groq-based AI explainer"""
        strategy = GroqExplanationStrategy(api_key, model)
        return AIExplainer(strategy)
    
    @staticmethod
    def create_openai_explainer(api_key: str, model: str = 'gpt-3.5-turbo') -> AIExplainer:
        """Create an OpenAI-based AI explainer"""
        strategy = OpenAIExplanationStrategy(api_key, model)
        return AIExplainer(strategy)
    
    @staticmethod
    def create_explainer(service: str, api_key: str, **kwargs) -> AIExplainer:
        """Create an AI explainer based on the service type"""
        if service.lower() == 'groq':
            model = kwargs.get('model', 'meta-llama/llama-4-scout-17b-16e-instruct')
            return AIExplainerFactory.create_groq_explainer(api_key, model)
        elif service.lower() == 'openai':
            model = kwargs.get('model', 'gpt-3.5-turbo')
            return AIExplainerFactory.create_openai_explainer(api_key, model)
        else:
            raise ValueError(f"Unknown AI service: {service}")
