"""
Pipeline factory implementing Factory pattern for creating analysis pipelines.
"""

from typing import Dict, Any, Optional, List
import logging

from .quantitative_pipeline import QuantitativeAnalysisPipeline
from .progress_observer import StreamlitProgressObserver, ConsoleProgressObserver, MultiProgressObserver
from core.exceptions import AnalysisError

class PipelineFactory:
    """Factory for creating analysis pipelines (Factory pattern)"""
    
    @staticmethod
    def create_quantitative_pipeline(config: Dict[str, Any], 
                                   progress_observers: Optional[List] = None) -> QuantitativeAnalysisPipeline:
        """Create a quantitative analysis pipeline"""
        try:
            # Create pipeline
            pipeline = QuantitativeAnalysisPipeline(config)
            
            # Add progress observers
            if progress_observers:
                for observer in progress_observers:
                    pipeline.add_observer(observer)
            
            return pipeline
            
        except Exception as e:
            raise AnalysisError(f"Failed to create quantitative pipeline: {e}")
    
    @staticmethod
    def create_streamlit_pipeline(config: Dict[str, Any], 
                                progress_bar=None, status_text=None, results_container=None) -> QuantitativeAnalysisPipeline:
        """Create a pipeline with Streamlit progress observer"""
        try:
            # Create Streamlit observer
            streamlit_observer = StreamlitProgressObserver(progress_bar, status_text, results_container)
            
            # Create pipeline with Streamlit observer
            pipeline = PipelineFactory.create_quantitative_pipeline(config, [streamlit_observer])
            
            return pipeline
            
        except Exception as e:
            raise AnalysisError(f"Failed to create Streamlit pipeline: {e}")
    
    @staticmethod
    def create_console_pipeline(config: Dict[str, Any]) -> QuantitativeAnalysisPipeline:
        """Create a pipeline with console progress observer"""
        try:
            # Create console observer
            console_observer = ConsoleProgressObserver()
            
            # Create pipeline with console observer
            pipeline = PipelineFactory.create_quantitative_pipeline(config, [console_observer])
            
            return pipeline
            
        except Exception as e:
            raise AnalysisError(f"Failed to create console pipeline: {e}")
    
    @staticmethod
    def create_multi_observer_pipeline(config: Dict[str, Any], 
                                     streamlit_components=None) -> QuantitativeAnalysisPipeline:
        """Create a pipeline with multiple progress observers"""
        try:
            observers = []
            
            # Add console observer
            observers.append(ConsoleProgressObserver())
            
            # Add Streamlit observer if components provided
            if streamlit_components:
                progress_bar = streamlit_components.get('progress_bar')
                status_text = streamlit_components.get('status_text')
                results_container = streamlit_components.get('results_container')
                observers.append(StreamlitProgressObserver(progress_bar, status_text, results_container))
            
            # Create multi-observer
            multi_observer = MultiProgressObserver(observers)
            
            # Create pipeline with multi-observer
            pipeline = PipelineFactory.create_quantitative_pipeline(config, [multi_observer])
            
            return pipeline
            
        except Exception as e:
            raise AnalysisError(f"Failed to create multi-observer pipeline: {e}")

class PipelineBuilder:
    """Builder pattern for creating complex pipeline configurations"""
    
    def __init__(self):
        self.config = {}
        self.observers = []
        self.logger = logging.getLogger(f"{__name__}.PipelineBuilder")
    
    def with_ticker(self, ticker: str) -> 'PipelineBuilder':
        """Set the ticker symbol"""
        self.config['ticker'] = ticker
        return self
    
    def with_benchmark(self, benchmark: str) -> 'PipelineBuilder':
        """Set the benchmark symbol"""
        self.config['benchmark'] = benchmark
        return self
    
    def with_date_range(self, start_date: str, end_date: str) -> 'PipelineBuilder':
        """Set the date range"""
        self.config['start_date'] = start_date
        self.config['end_date'] = end_date
        return self
    
    def with_anomaly_detection(self, z_threshold: float = 2.5, 
                             vol_window: int = 10, vol_multiplier: float = 2.0) -> 'PipelineBuilder':
        """Configure anomaly detection parameters"""
        self.config['z_threshold'] = z_threshold
        self.config['vol_window'] = vol_window
        self.config['vol_multiplier'] = vol_multiplier
        return self
    
    def with_news_service(self, service: str, api_key: Optional[str] = None) -> 'PipelineBuilder':
        """Configure news service"""
        self.config['news_service'] = service
        if api_key:
            self.config['news_api_key'] = api_key
        return self
    
    def with_embedding_service(self, service: str, **kwargs) -> 'PipelineBuilder':
        """Configure embedding service"""
        self.config['embedding_service'] = service
        self.config['embedding_config'] = kwargs
        return self
    
    def with_ai_service(self, service: str, api_key: str, **kwargs) -> 'PipelineBuilder':
        """Configure AI service"""
        self.config['ai_service'] = service
        self.config['ai_api_key'] = api_key
        self.config['ai_config'] = kwargs
        return self
    
    def with_similarity_analysis(self, index_type: str = 'flat') -> 'PipelineBuilder':
        """Configure similarity analysis"""
        self.config['similarity_index_type'] = index_type
        return self
    
    def with_save_results(self, save: bool = True) -> 'PipelineBuilder':
        """Configure result saving"""
        self.config['save_results'] = save
        return self
    
    def with_streamlit_observer(self, progress_bar=None, status_text=None, results_container=None) -> 'PipelineBuilder':
        """Add Streamlit progress observer"""
        observer = StreamlitProgressObserver(progress_bar, status_text, results_container)
        self.observers.append(observer)
        return self
    
    def with_console_observer(self) -> 'PipelineBuilder':
        """Add console progress observer"""
        observer = ConsoleProgressObserver()
        self.observers.append(observer)
        return self
    
    def build(self) -> QuantitativeAnalysisPipeline:
        """Build the pipeline with the configured parameters"""
        try:
            # Set default values
            self.config.setdefault('ticker', 'TSLA')
            self.config.setdefault('benchmark', 'SPY')
            self.config.setdefault('start_date', '2024-10-01')
            self.config.setdefault('end_date', '2024-12-31')
            self.config.setdefault('news_service', 'yahoo')
            self.config.setdefault('embedding_service', 'sentence_transformer')
            self.config.setdefault('similarity_index_type', 'flat')
            self.config.setdefault('save_results', True)
            
            # Create pipeline
            pipeline = PipelineFactory.create_quantitative_pipeline(self.config, self.observers)
            
            self.logger.info(f"Pipeline built successfully with config: {self.config}")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to build pipeline: {e}")
            raise AnalysisError(f"Pipeline build failed: {e}")

class PipelineDirector:
    """Director for creating pre-configured pipelines (Director pattern)"""
    
    @staticmethod
    def create_basic_pipeline(ticker: str, benchmark: str = 'SPY') -> QuantitativeAnalysisPipeline:
        """Create a basic pipeline with minimal configuration"""
        builder = PipelineBuilder()
        return (builder
                .with_ticker(ticker)
                .with_benchmark(benchmark)
                .with_console_observer()
                .build())
    
    @staticmethod
    def create_advanced_pipeline(ticker: str, benchmark: str = 'SPY', 
                               news_api_key: Optional[str] = None,
                               ai_api_key: Optional[str] = None) -> QuantitativeAnalysisPipeline:
        """Create an advanced pipeline with all services"""
        builder = PipelineBuilder()
        
        # Basic configuration
        builder.with_ticker(ticker).with_benchmark(benchmark)
        
        # News service
        if news_api_key:
            builder.with_news_service('finnhub', news_api_key)
        else:
            builder.with_news_service('yahoo')
        
        # AI service
        if ai_api_key:
            builder.with_ai_service('groq', ai_api_key)
        
        # Console observer
        builder.with_console_observer()
        
        return builder.build()
    
    @staticmethod
    def create_streamlit_pipeline(ticker: str, benchmark: str = 'SPY',
                                news_api_key: Optional[str] = None,
                                ai_api_key: Optional[str] = None,
                                progress_bar=None, status_text=None, results_container=None) -> QuantitativeAnalysisPipeline:
        """Create a pipeline optimized for Streamlit UI"""
        builder = PipelineBuilder()
        
        # Basic configuration
        builder.with_ticker(ticker).with_benchmark(benchmark)
        
        # News service
        if news_api_key:
            builder.with_news_service('finnhub', news_api_key)
        else:
            builder.with_news_service('yahoo')
        
        # AI service
        if ai_api_key:
            builder.with_ai_service('groq', ai_api_key)
        
        # Streamlit observer
        builder.with_streamlit_observer(progress_bar, status_text, results_container)
        
        return builder.build()
