"""
OOP-based Streamlit application implementing design patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from core.exceptions import AnalysisError
from pipeline.quantitative_pipeline import QuantitativeAnalysisPipeline
from pipeline.pipeline_factory import PipelineDirector
from pipeline.progress_observer import StreamlitProgressObserver
from ui.ui_components import (
    UIComponentManager, UIComponentFactory,
    DataDisplayComponent, EventsDisplayComponent, ChartComponent,
    AIExplanationComponent, SimilarityAnalysisComponent, DownloadComponent
)

class StreamlitApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StreamlitApp")
        self.component_manager = UIComponentManager()
        self._initialize_components()
        self._setup_page_config()
        self._setup_css()
    
    def _initialize_components(self) -> None:
        """Initialize UI components"""
        try:
            # Register components
            self.component_manager.register_component("data_display", UIComponentFactory.create_data_display())
            self.component_manager.register_component("events_display", UIComponentFactory.create_events_display())
            self.component_manager.register_component("chart", UIComponentFactory.create_chart())
            self.component_manager.register_component("ai_explanation", UIComponentFactory.create_ai_explanation())
            self.component_manager.register_component("similarity_analysis", UIComponentFactory.create_similarity_analysis())
            self.component_manager.register_component("download", UIComponentFactory.create_download())
            
            self.logger.info("UI components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise AnalysisError(f"Component initialization failed: {e}")
    
    def _setup_page_config(self) -> None:
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Quantitative Market Analysis",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _setup_css(self) -> None:
        """Setup custom CSS"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .ai-explanation {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 0.5rem;
                border-left: 4px solid #007bff;
                margin: 1rem 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #212529;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .error-box {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 0.5rem;
                padding: 1rem;
                margin: 1rem 0;
            }
            .success-box {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 0.5rem;
                padding: 1rem;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render the sidebar with configuration options"""
        st.sidebar.header("⚙️ Configuration")
        
        # Analysis parameters
        ticker = st.sidebar.text_input("Stock Ticker", value="TSLA", help="Enter the stock symbol (e.g., TSLA, AAPL, MSFT)")
        benchmark = st.sidebar.text_input("Benchmark", value="SPY", help="Enter the benchmark symbol (e.g., SPY, QQQ)")
        
        # Date range
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2024, 10, 1), help="Analysis start date")
        with col2:
            end_date = st.date_input("End Date", value=datetime(2024, 12, 31), help="Analysis end date")
        
        # Analysis parameters
        st.sidebar.subheader("📈 Analysis Parameters")
        z_threshold = st.sidebar.slider("Z-Score Threshold", min_value=1.0, max_value=5.0, value=2.5, step=0.1, help="Threshold for detecting outliers")
        vol_window = st.sidebar.slider("Volatility Window", min_value=5, max_value=30, value=10, help="Rolling window for volatility calculation")
        vol_multiplier = st.sidebar.slider("Volatility Multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.1, help="Multiplier for volatility spike detection")
        
        # News parameters
        st.sidebar.subheader("📰 News Parameters")
        news_window = st.sidebar.slider("News Window (days)", min_value=1, max_value=7, value=1, help="Days around event to fetch news")
        max_news = st.sidebar.slider("Max News Articles", min_value=1, max_value=20, value=5, help="Maximum news articles per event")
        
        # API Keys
        st.sidebar.subheader("🔑 API Keys")
        groq_key = st.sidebar.text_input("Groq API Key", type="password", help="Required for AI explanations")
        finnhub_key = st.sidebar.text_input("Finnhub API Key", type="password", help="Required for news retrieval")
        
        return {
            'ticker': ticker,
            'benchmark': benchmark,
            'start_date': start_date,
            'end_date': end_date,
            'z_threshold': z_threshold,
            'vol_window': vol_window,
            'vol_multiplier': vol_multiplier,
            'news_window': news_window,
            'max_news': max_news,
            'groq_key': groq_key,
            'finnhub_key': finnhub_key
        }
    
    def render_welcome_page(self) -> None:
        """Render the welcome page"""
        st.markdown('<h1 class="main-header">📊 Financial Anomaly Interpretability using LLMs</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 Welcome to Quantitative Market Analysis
        
        This application provides comprehensive market analysis using:
        - **Statistical Anomaly Detection**: Identify unusual market movements
        - **News Integration**: Connect events to relevant news headlines  
        - **AI-Powered Explanations**: Generate insights using similar historical events
        - **Interactive Charts**: Visualize trends and events
        """)
        
        st.markdown("---")
        
        # Sample data preview
        st.subheader("📈 Sample Analysis Preview")
        
        # Create sample data for preview
        sample_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        sample_returns = np.random.normal(0, 0.02, len(sample_dates))
        sample_data = pd.DataFrame({
            'Date': sample_dates,
            'Return': sample_returns,
            'Cumulative_Return': (1 + sample_returns).cumprod() - 1
        })
        
        # Plot sample data
        import plotly.express as px
        fig = px.line(sample_data, x='Date', y='Cumulative_Return', 
                      title="Sample Cumulative Returns",
                      labels={'Cumulative_Return': 'Cumulative Return', 'Date': 'Date'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def run_analysis(self, config: Dict[str, Any]) -> None:
        """Run the analysis pipeline"""
        try:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            # Create pipeline
            pipeline = PipelineDirector.create_streamlit_pipeline(
                ticker=config['ticker'],
                benchmark=config['benchmark'],
                news_api_key=config['finnhub_key'] if config['finnhub_key'] else None,
                ai_api_key=config['groq_key'] if config['groq_key'] else None,
                progress_bar=progress_bar,
                status_text=status_text,
                results_container=results_container
            )
            
            # Run pipeline
            results = pipeline.run(
                ticker=config['ticker'],
                benchmark=config['benchmark'],
                start_date=config['start_date'].strftime('%Y-%m-%d'),
                end_date=config['end_date'].strftime('%Y-%m-%d'),
                z_threshold=config['z_threshold'],
                vol_window=config['vol_window'],
                vol_multiplier=config['vol_multiplier'],
                news_window=config['news_window'],
                max_news=config['max_news']
            )
            
            # Display results
            self._display_results(results, config)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            self.logger.error(f"Analysis failed: {e}")
    
    def _display_results(self, results: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Display analysis results"""
        try:
            # Display data summary
            if 'processed_data' in results:
                self.component_manager.render_component("data_display", 
                    data=results['processed_data'], 
                    title="📊 Data Summary"
                )
            
            # Display events
            if 'events_with_news' in results:
                self.component_manager.render_component("events_display",
                    events=results['events_with_news'],
                    title="📋 Market Events with News"
                )
            
            # Display chart
            if 'processed_data' in results and 'events_with_news' in results:
                self.component_manager.render_component("chart",
                    data=results['processed_data'],
                    events=results['events_with_news'],
                    title="📈 Price Chart with Events",
                    ticker=config['ticker']
                )
            
            # Display AI explanations
            if 'latest_explanation' in results and 'multiple_explanations' in results:
                self.component_manager.render_component("ai_explanation",
                    latest_explanation=results['latest_explanation'],
                    multiple_explanations=results['multiple_explanations'],
                    events=results.get('events_with_embeddings'),
                    title="🤖 AI-Powered Market Analysis"
                )
            
            # Display similarity analysis
            if 'similarity_analysis' in results:
                self.component_manager.render_component("similarity_analysis",
                    similarity_analysis=results['similarity_analysis'],
                    title="🔍 Similarity Analysis"
                )
            
            # Display download options
            if 'events_with_embeddings' in results:
                self.component_manager.render_component("download",
                    events=results['events_with_embeddings'],
                    explanations=results.get('multiple_explanations'),
                    title="💾 Download Results"
                )
            
        except Exception as e:
            st.error(f"❌ Failed to display results: {e}")
            self.logger.error(f"Failed to display results: {e}")
    
    def run(self) -> None:
        """Run the Streamlit application"""
        try:
            # Render sidebar
            config = self.render_sidebar()
            
            # Run analysis button
            run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", key="run_analysis_btn")
            
            if run_analysis:
                # Validate configuration
                if not config['groq_key'] or not config['finnhub_key']:
                    st.error("❌ Please enter both API keys to run the analysis.")
                else:
                    # Run analysis
                    self.run_analysis(config)
            else:
                # Show welcome page
                self.render_welcome_page()
                
        except Exception as e:
            st.error(f"❌ Application error: {e}")
            self.logger.error(f"Application error: {e}")

def main():
    """Main function to run the Streamlit app"""
    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {e}")
        st.info("💡 Try using the command line interface: `python main.py`")

if __name__ == "__main__":
    main()
