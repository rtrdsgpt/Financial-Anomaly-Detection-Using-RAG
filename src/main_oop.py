"""
OOP-based main application implementing design patterns and SOLID principles.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import sys
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(over='ignore', under='ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.exceptions import AnalysisError
from pipeline.quantitative_pipeline import QuantitativeAnalysisPipeline
from pipeline.pipeline_factory import PipelineDirector, PipelineFactory
from pipeline.progress_observer import ConsoleProgressObserver

class QuantitativeAnalysisApp:
    """Main application class implementing OOP principles"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.QuantitativeAnalysisApp")
        self.pipeline: Optional[QuantitativeAnalysisPipeline] = None
        self.results: Dict[str, Any] = {}
        
        # Setup logging
        self._setup_logging()
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from api_keys.txt file"""
        api_keys = {}
        try:
            with open('api_keys.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        api_keys[key.strip()] = value.strip().strip('"')
        except FileNotFoundError:
            self.logger.warning("api_keys.txt file not found")
        return api_keys
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        api_keys = self._load_api_keys()
        
        return {
            'ticker': 'TSLA',
            'benchmark': 'SPY',
            'start_date': '2024-10-01',
            'end_date': '2024-12-31',
            'z_threshold': 2.5,
            'vol_window': 10,
            'vol_multiplier': 2.0,
            'news_service': 'yahoo',
            'news_api_key': api_keys.get('FIN_HUB'),
            'embedding_service': 'sentence_transformer',
            'similarity_index_type': 'flat',
            'ai_service': 'groq',
            'ai_api_key': api_keys.get('GROQ_API'),
            'save_results': True
        }
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('quantitative_analysis.log')
            ]
        )
    
    def _initialize_pipeline(self) -> None:
        """Initialize the analysis pipeline"""
        try:
            self.pipeline = PipelineFactory.create_console_pipeline(self.config)
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise AnalysisError(f"Pipeline initialization failed: {e}")
    
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        try:
            self.logger.info("Starting quantitative analysis")
            
            # Update config with provided parameters
            updated_config = {**self.config, **kwargs}
            
            # Run pipeline
            self.results = self.pipeline.run(**updated_config)
            
            self.logger.info("Analysis completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise AnalysisError(f"Analysis failed: {e}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results"""
        return self.results.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary"""
        if not self.results:
            return {"error": "No analysis results available"}
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results_available': list(self.results.keys())
        }
        
        # Add event summary if available
        if 'event_summary' in self.results:
            summary['event_summary'] = self.results['event_summary']
        
        # Add similarity analysis if available
        if 'similarity_analysis' in self.results:
            summary['similarity_analysis'] = self.results['similarity_analysis']
        
        return summary
    
    def print_summary(self) -> None:
        """Print analysis summary to console"""
        try:
            print("\n" + "="*60)
            print("QUANTITATIVE ANALYSIS SUMMARY")
            print("="*60)
            
            if not self.results:
                print("No analysis results available")
                return
            
            # Configuration
            print(f"\nConfiguration:")
            print(f"  Ticker: {self.config.get('ticker', 'N/A')}")
            print(f"  Benchmark: {self.config.get('benchmark', 'N/A')}")
            print(f"  Date Range: {self.config.get('start_date', 'N/A')} to {self.config.get('end_date', 'N/A')}")
            
            # Event summary
            if 'event_summary' in self.results:
                print(f"\nEvent Detection:")
                for key, value in self.results['event_summary'].items():
                    print(f"  {key}: {value}")
            
            # Similarity analysis
            if 'similarity_analysis' in self.results:
                print(f"\nSimilarity Analysis:")
                for key, value in self.results['similarity_analysis'].items():
                    print(f"  {key}: {value}")
            
            # Latest explanation
            if 'latest_explanation' in self.results:
                print(f"\nLatest Event Explanation:")
                print("-" * 40)
                print(self.results['latest_explanation'])
                print("-" * 40)
            
            print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Failed to print summary: {e}")
            print(f"Error printing summary: {e}")
    
    def show_recent_events(self, limit: int = 5) -> None:
        """Show recent events"""
        try:
            if 'events_with_embeddings' not in self.results:
                print("No events with embeddings available")
                return
            
            events = self.results['events_with_embeddings']
            print(f"\nRecent Events ({len(events)} total):")
            
            recent_events = events.tail(limit)
            for _, event in recent_events.iterrows():
                print(f"  {event['Date']}: {event['Event_Type']} (Z-score: {event['Z_score']:.2f})")
                if 'News_Headlines' in event and pd.notna(event['News_Headlines']):
                    news_preview = str(event['News_Headlines'])[:100]
                    print(f"    News: {news_preview}...")
                print()
                
        except Exception as e:
            self.logger.error(f"Failed to show recent events: {e}")
            print(f"Error showing recent events: {e}")

def main():
    """Main function"""
    try:
        print("="*60)
        print("QUANTITATIVE ANALYSIS PIPELINE (OOP VERSION)")
        print("="*60)
        
        # Create application
        app = QuantitativeAnalysisApp()
        
        # Run analysis
        results = app.run_analysis()
        
        # Print summary
        app.print_summary()
        
        # Show recent events
        app.show_recent_events()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
