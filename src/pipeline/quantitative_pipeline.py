"""
Main quantitative analysis pipeline implementing Template Method pattern.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from core.base import BaseAnalysisPipeline
from core.interfaces import IProgressObserver, IDataRepository
from core.exceptions import AnalysisError

from processors.data_loader import DataLoader, YFinanceDataStrategy
from processors.anomaly_detector import AnomalyDetector, ZScoreStrategy, IsolationForestStrategy
from processors.news_retriever import NewsRetriever, NewsRetrieverFactory
from processors.embedding_generator import EmbeddingGenerator, EmbeddingGeneratorFactory
from processors.similarity_analyzer import SimilarityAnalyzer, FAISSStrategy, CosineSimilarityStrategy
from processors.ai_explainer import AIExplainer, AIExplainerFactory

class QuantitativeAnalysisPipeline(BaseAnalysisPipeline):
    """Main quantitative analysis pipeline with Template Method pattern"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("QuantitativeAnalysisPipeline")
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.QuantitativeAnalysisPipeline")
        
        # Initialize components
        self._initialize_components()
        
        # Results storage
        self._results: Dict[str, Any] = {}
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components"""
        try:
            # Data loader
            self.data_loader = DataLoader(YFinanceDataStrategy())
            
            # Anomaly detector
            anomaly_strategy = ZScoreStrategy(
                z_threshold=self.config.get('z_threshold', 2.5)
            )
            # Set additional parameters
            anomaly_strategy.vol_window = self.config.get('vol_window', 10)
            anomaly_strategy.vol_multiplier = self.config.get('vol_multiplier', 2.0)
            self.anomaly_detector = AnomalyDetector(anomaly_strategy)
            
            # News retriever
            news_service = self.config.get('news_service', 'finnhub')
            news_api_key = self.config.get('news_api_key')
            if news_service == 'finnhub' and news_api_key:
                self.news_retriever = NewsRetrieverFactory.create_retriever('finnhub', news_api_key)
            else:
                self.news_retriever = NewsRetrieverFactory.create_retriever('yahoo')
            
            # Embedding generator
            embedding_service = self.config.get('embedding_service', 'sentence_transformer')
            embedding_config = self.config.get('embedding_config', {})
            self.embedding_generator = EmbeddingGeneratorFactory.create_generator(
                embedding_service, **embedding_config
            )
            
            # Similarity analyzer
            similarity_strategy = FAISSStrategy(
                index_type=self.config.get('similarity_index_type', 'flat')
            )
            self.similarity_analyzer = SimilarityAnalyzer(similarity_strategy)
            
            # AI explainer
            ai_service = self.config.get('ai_service', 'groq')
            ai_api_key = self.config.get('ai_api_key')
            ai_config = self.config.get('ai_config', {})
            if ai_api_key:
                self.ai_explainer = AIExplainerFactory.create_explainer(ai_service, ai_api_key, **ai_config)
            else:
                self.ai_explainer = None
            
            # Add observers to components
            for component in [self.data_loader, self.anomaly_detector, self.news_retriever, 
                            self.embedding_generator, self.similarity_analyzer]:
                for observer in self._observers:
                    component.add_observer(observer)
            
            if self.ai_explainer:
                for observer in self._observers:
                    self.ai_explainer.add_observer(observer)
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise AnalysisError(f"Pipeline initialization failed: {e}")
    
    def _preprocess(self, **kwargs) -> None:
        """Preprocessing step - load and prepare data"""
        try:
            self.logger.info("Starting preprocessing phase")
            self.notify_progress("preprocessing", 10.0, "Loading market data")
            
            # Extract parameters
            ticker = kwargs.get('ticker', self.config.get('ticker', 'TSLA'))
            benchmark = kwargs.get('benchmark', self.config.get('benchmark', 'SPY'))
            start_date = kwargs.get('start_date', self.config.get('start_date', '2024-10-01'))
            end_date = kwargs.get('end_date', self.config.get('end_date', '2024-12-31'))
            
            # Load market data
            data_results = self.data_loader.load_market_data(ticker, benchmark, start_date, end_date)
            
            # Store results
            self._results.update(data_results)
            
            self.notify_progress("preprocessing", 100.0, "Data loading completed")
            self.logger.info("Preprocessing phase completed successfully")
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            self.notify_error(e, "preprocessing")
            raise AnalysisError(f"Preprocessing failed: {e}")
    
    def _analyze(self, **kwargs) -> None:
        """Analysis step - detect anomalies and generate insights"""
        try:
            self.logger.info("Starting analysis phase")
            self.notify_progress("analysis", 10.0, "Detecting market anomalies")
            
            # Get processed data
            processed_data = self._results.get('processed_data')
            if processed_data is None or processed_data.empty:
                raise AnalysisError("No processed data available for analysis")
            
            # Anomaly detection
            events = self.anomaly_detector.detect(processed_data)
            event_summary = self.anomaly_detector.analyze_events(events)
            
            self._results['events'] = events
            self._results['event_summary'] = event_summary
            
            if events.empty:
                self.logger.warning("No market events detected")
                self.notify_progress("analysis", 100.0, "No events detected")
                return
            
            self.notify_progress("analysis", 30.0, "Fetching news headlines")
            
            # News retrieval
            ticker = kwargs.get('ticker', self.config.get('ticker', 'TSLA'))
            news_window = kwargs.get('news_window', self.config.get('news_window', 1))
            events_with_news = self.news_retriever.add_news_to_events(events, ticker, news_window)
            
            self._results['events_with_news'] = events_with_news
            
            if events_with_news.empty:
                self.logger.warning("No events with news found, proceeding with events only")
                # Use original events if no news found
                events_with_news = events.copy()
                events_with_news['News_Headlines'] = ''
            
            self.notify_progress("analysis", 50.0, "Generating embeddings")
            
            # Embedding generation
            events_with_embeddings = self.embedding_generator.add_embeddings_to_events(events_with_news)
            
            self._results['events_with_embeddings'] = events_with_embeddings
            
            self.notify_progress("analysis", 70.0, "Creating similarity index")
            
            # Similarity analysis
            embeddings = np.array([event['Embedding'] for _, event in events_with_embeddings.iterrows()])
            index = self.similarity_analyzer.create_index(embeddings)
            similarity_analysis = self.similarity_analyzer.analyze_similarity_patterns(events_with_embeddings, index)
            
            self._results['faiss_index'] = index
            self._results['similarity_analysis'] = similarity_analysis
            
            self.notify_progress("analysis", 90.0, "Generating AI explanations")
            
            # AI explanations
            if self.ai_explainer and self.ai_explainer.is_available():
                latest_explanation = self.ai_explainer.explain_latest_event(
                    events_with_embeddings, self.similarity_analyzer, index
                )
                
                recent_indices = list(range(max(0, len(events_with_embeddings) - 3), len(events_with_embeddings)))
                multiple_explanations = self.ai_explainer.explain_multiple_events(
                    events_with_embeddings, self.similarity_analyzer, index, recent_indices
                )
                
                self._results['latest_explanation'] = latest_explanation
                self._results['multiple_explanations'] = multiple_explanations
            else:
                self.logger.warning("AI explainer not available, skipping explanations")
            
            self.notify_progress("analysis", 100.0, "Analysis completed")
            self.logger.info("Analysis phase completed successfully")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            self.notify_error(e, "analysis")
            raise AnalysisError(f"Analysis failed: {e}")
    
    def _postprocess(self, **kwargs) -> None:
        """Postprocessing step - save results and generate reports"""
        try:
            self.logger.info("Starting postprocessing phase")
            self.notify_progress("postprocessing", 10.0, "Saving results")
            
            # Save results if requested
            save_results = kwargs.get('save_results', self.config.get('save_results', True))
            if save_results:
                self._save_results()
            
            self.notify_progress("postprocessing", 100.0, "Postprocessing completed")
            self.logger.info("Postprocessing phase completed successfully")
            
        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            self.notify_error(e, "postprocessing")
            raise AnalysisError(f"Postprocessing failed: {e}")
    
    def _save_results(self) -> None:
        """Save pipeline results to files"""
        try:
            from core.base import BaseDataRepository
            repository = BaseDataRepository()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save events with embeddings
            if 'events_with_embeddings' in self._results:
                events_file = f"events_with_embeddings_{timestamp}.parquet"
                repository.save_events(self._results['events_with_embeddings'], events_file)
                self.logger.info(f"Events saved to {events_file}")
            
            # Save explanations
            if 'multiple_explanations' in self._results:
                explanations_file = f"explanations_{timestamp}.txt"
                repository.save_explanations(self._results['multiple_explanations'], explanations_file)
                self.logger.info(f"Explanations saved to {explanations_file}")
            
            # Save summary
            summary_file = f"pipeline_summary_{timestamp}.txt"
            self._save_summary(summary_file)
            self.logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise AnalysisError(f"Failed to save results: {e}")
    
    def _save_summary(self, filename: str) -> None:
        """Save pipeline summary to file"""
        try:
            with open(f"results/{filename}", 'w') as f:
                f.write("QUANTITATIVE ANALYSIS PIPELINE SUMMARY\n")
                f.write("="*50 + "\n\n")
                
                # Configuration
                f.write("CONFIGURATION:\n")
                for key, value in self.config.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # Results
                f.write("RESULTS:\n")
                if 'event_summary' in self._results:
                    f.write("Event Summary:\n")
                    for key, value in self._results['event_summary'].items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                if 'similarity_analysis' in self._results:
                    f.write("Similarity Analysis:\n")
                    for key, value in self._results['similarity_analysis'].items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # Timestamp
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        except Exception as e:
            self.logger.error(f"Failed to save summary: {e}")
            raise AnalysisError(f"Failed to save summary: {e}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get pipeline results"""
        return self._results.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline results"""
        summary = {
            'pipeline_name': self.name,
            'config': self.config,
            'results_available': list(self._results.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        if 'event_summary' in self._results:
            summary['event_summary'] = self._results['event_summary']
        
        if 'similarity_analysis' in self._results:
            summary['similarity_analysis'] = self._results['similarity_analysis']
        
        return summary
