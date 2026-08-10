"""
UI components implementing OOP principles and design patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
import logging

from core.interfaces import IProgressObserver
from core.exceptions import AnalysisError

class UIComponent(ABC):
    """Base class for UI components"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def render(self, **kwargs) -> Any:
        """Render the UI component"""
        pass

class DataDisplayComponent(UIComponent):
    """Component for displaying data"""
    
    def __init__(self):
        super().__init__("DataDisplayComponent")
    
    def render(self, data: pd.DataFrame, title: str = "Data Display") -> None:
        """Render data display"""
        try:
            st.subheader(title)
            
            if data.empty:
                st.warning("No data to display")
                return
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(data))
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                if 'Return' in data.columns:
                    st.metric("Avg Return", f"{data['Return'].mean():.4f}")
            with col4:
                if 'Return' in data.columns:
                    st.metric("Volatility", f"{data['Return'].std():.4f}")
            
            # Display data table
            st.dataframe(data.head(10))
            
        except Exception as e:
            self.logger.error(f"Failed to render data display: {e}")
            st.error(f"Failed to display data: {e}")

class EventsDisplayComponent(UIComponent):
    """Component for displaying market events"""
    
    def __init__(self):
        super().__init__("EventsDisplayComponent")
    
    def render(self, events: pd.DataFrame, title: str = "Market Events") -> None:
        """Render events display"""
        try:
            st.subheader(title)
            
            if events.empty:
                st.warning("No events to display")
                return
            
            # Display events summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Events", len(events))
            with col2:
                positive_count = len(events[events['Event_Type'] == 'Positive Outlier'])
                st.metric("Positive Outliers", positive_count)
            with col3:
                negative_count = len(events[events['Event_Type'] == 'Negative Outlier'])
                st.metric("Negative Outliers", negative_count)
            with col4:
                volatility_count = len(events[events['Event_Type'] == 'Volatility Spike'])
                st.metric("Volatility Spikes", volatility_count)
            
            # Display events with details
            for idx, (_, event) in enumerate(events.iterrows()):
                with st.expander(f"Event {idx+1}: {event['Event_Type']} on {event['Date'].strftime('%Y-%m-%d')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Z-Score", f"{event['Z_score']:.2f}")
                        st.metric("Return", f"{event['Return']:.4f}")
                    with col2:
                        st.metric("Price", f"${event['Close']:.2f}")
                        if 'Rolling_STD' in event:
                            st.metric("Volatility", f"{event['Rolling_STD']:.4f}")
                    
                    # Display news if available
                    if 'News_Headlines' in event and pd.notna(event['News_Headlines']):
                        st.write("**News Headlines:**")
                        headlines = str(event['News_Headlines']).split('; ')
                        for i, headline in enumerate(headlines[:3]):
                            st.write(f"{i+1}. {headline}")
                        if len(headlines) > 3:
                            st.write(f"... and {len(headlines) - 3} more headlines")
            
        except Exception as e:
            self.logger.error(f"Failed to render events display: {e}")
            st.error(f"Failed to display events: {e}")

class ChartComponent(UIComponent):
    """Component for displaying charts"""
    
    def __init__(self):
        super().__init__("ChartComponent")
    
    def render(self, data: pd.DataFrame, events: Optional[pd.DataFrame] = None, 
               title: str = "Price Chart", ticker: str = "Stock") -> None:
        """Render price chart with events"""
        try:
            st.subheader(title)
            
            if data.empty:
                st.warning("No data to display")
                return
            
            # Find the correct close column name
            close_col = None
            for col in ['Close', 'Adj Close', 'close', 'adj_close']:
                if col in data.columns:
                    close_col = col
                    break
            
            if close_col is None:
                st.error(f"Could not find Close column. Available columns: {data.columns.tolist()}")
                return
            
            # Create the chart
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=list(data.index),
                y=list(data[close_col]),
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add events if provided
            if events is not None and not events.empty and 'Close' in events.columns:
                event_dates = pd.to_datetime(events['Date'])
                
                fig.add_trace(go.Scatter(
                    x=event_dates,
                    y=events['Close'],
                    mode='markers',
                    name='Events',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='diamond'
                    ),
                    text=events['Event_Type'],
                    hovertemplate="<b>Event</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Type: %{text}<extra></extra>"
                ))
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} Stock Price with Events",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                showlegend=True
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Failed to render chart: {e}")
            st.error(f"Failed to display chart: {e}")

class AIExplanationComponent(UIComponent):
    """Component for displaying AI explanations"""
    
    def __init__(self):
        super().__init__("AIExplanationComponent")
    
    def render(self, latest_explanation: str, multiple_explanations: Optional[Dict[int, str]] = None,
               events: Optional[pd.DataFrame] = None, title: str = "AI-Powered Market Analysis") -> None:
        """Render AI explanations"""
        try:
            st.subheader(title)
            
            # Latest event explanation
            if latest_explanation:
                st.markdown("#### Latest Event Explanation")
                st.markdown(f'<div class="ai-explanation">{latest_explanation}</div>', unsafe_allow_html=True)
            
            # Multiple events explanations
            if multiple_explanations and events is not None:
                st.markdown("#### Historical Event Analysis")
                
                for event_idx, explanation in multiple_explanations.items():
                    if event_idx < len(events):
                        event = events.iloc[event_idx]
                        with st.expander(f"Event {event_idx+1}: {event['Event_Type']} on {event['Date'].strftime('%Y-%m-%d')}"):
                            st.markdown(f'<div class="ai-explanation">{explanation}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            self.logger.error(f"Failed to render AI explanations: {e}")
            st.error(f"Failed to display AI explanations: {e}")

class SimilarityAnalysisComponent(UIComponent):
    """Component for displaying similarity analysis"""
    
    def __init__(self):
        super().__init__("SimilarityAnalysisComponent")
    
    def render(self, similarity_analysis: Dict[str, Any], title: str = "Similarity Analysis") -> None:
        """Render similarity analysis"""
        try:
            st.subheader(title)
            
            if not similarity_analysis:
                st.warning("No similarity analysis data available")
                return
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Events", similarity_analysis.get('total_events', 0))
            with col2:
                st.metric("Avg Distance", f"{similarity_analysis.get('avg_similarity_distance', 0):.4f}")
            with col3:
                st.metric("Min Distance", f"{similarity_analysis.get('min_similarity_distance', 0):.4f}")
            
        except Exception as e:
            self.logger.error(f"Failed to render similarity analysis: {e}")
            st.error(f"Failed to display similarity analysis: {e}")

class DownloadComponent(UIComponent):
    """Component for download functionality"""
    
    def __init__(self):
        super().__init__("DownloadComponent")
    
    def render(self, events: pd.DataFrame, explanations: Optional[Dict[int, str]] = None,
               title: str = "Download Results") -> None:
        """Render download options"""
        try:
            st.subheader(title)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download events as CSV
                csv = events.to_csv(index=False)
                st.download_button(
                    label="📊 Download Events (CSV)",
                    data=csv,
                    file_name=f"market_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download explanations as text
                if explanations:
                    explanations_text = "\n\n".join([f"Event {idx}:\n{explanation}" for idx, explanation in explanations.items()])
                    st.download_button(
                        label="📝 Download Explanations (TXT)",
                        data=explanations_text,
                        file_name=f"explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to render download options: {e}")
            st.error(f"Failed to display download options: {e}")

class UIComponentFactory:
    """Factory for creating UI components (Factory pattern)"""
    
    @staticmethod
    def create_data_display() -> DataDisplayComponent:
        """Create a data display component"""
        return DataDisplayComponent()
    
    @staticmethod
    def create_events_display() -> EventsDisplayComponent:
        """Create an events display component"""
        return EventsDisplayComponent()
    
    @staticmethod
    def create_chart() -> ChartComponent:
        """Create a chart component"""
        return ChartComponent()
    
    @staticmethod
    def create_ai_explanation() -> AIExplanationComponent:
        """Create an AI explanation component"""
        return AIExplanationComponent()
    
    @staticmethod
    def create_similarity_analysis() -> SimilarityAnalysisComponent:
        """Create a similarity analysis component"""
        return SimilarityAnalysisComponent()
    
    @staticmethod
    def create_download() -> DownloadComponent:
        """Create a download component"""
        return DownloadComponent()

class UIComponentManager:
    """Manager for UI components (Manager pattern)"""
    
    def __init__(self):
        self.components: Dict[str, UIComponent] = {}
        self.logger = logging.getLogger(f"{__name__}.UIComponentManager")
    
    def register_component(self, name: str, component: UIComponent) -> None:
        """Register a UI component"""
        self.components[name] = component
        self.logger.info(f"Registered component: {name}")
    
    def get_component(self, name: str) -> Optional[UIComponent]:
        """Get a UI component by name"""
        return self.components.get(name)
    
    def render_component(self, name: str, **kwargs) -> Any:
        """Render a UI component"""
        component = self.get_component(name)
        if component:
            return component.render(**kwargs)
        else:
            self.logger.error(f"Component not found: {name}")
            st.error(f"Component not found: {name}")
    
    def render_all(self, **kwargs) -> None:
        """Render all registered components"""
        for name, component in self.components.items():
            try:
                component.render(**kwargs)
            except Exception as e:
                self.logger.error(f"Failed to render component {name}: {e}")
                st.error(f"Failed to render component {name}: {e}")
