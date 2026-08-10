"""
Progress observer implementing Observer pattern for UI updates.
"""

import streamlit as st
from typing import Any, Dict, List
from datetime import datetime
import logging

from core.interfaces import IProgressObserver
from core.exceptions import AnalysisError

class StreamlitProgressObserver(IProgressObserver):
    """Observer for Streamlit UI progress updates"""
    
    def __init__(self, progress_bar=None, status_text=None, results_container=None):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.results_container = results_container
        self.logger = logging.getLogger(f"{__name__}.StreamlitProgressObserver")
        self._errors: List[Exception] = []
    
    def update_progress(self, step: str, progress: float, message: str) -> None:
        """Update progress information in Streamlit UI"""
        try:
            if self.progress_bar:
                self.progress_bar.progress(progress / 100.0)
            
            if self.status_text:
                self.status_text.text(message)
            
            if self.results_container:
                with self.results_container:
                    st.info(f"📊 {step}: {message}")
            
            self.logger.info(f"Progress update: {step} - {progress}% - {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to update progress: {e}")
    
    def on_error(self, error: Exception, step: str) -> None:
        """Handle errors during processing"""
        try:
            self._errors.append(error)
            
            if self.results_container:
                with self.results_container:
                    st.error(f"❌ Error in {step}: {str(error)}")
            
            self.logger.error(f"Error in {step}: {error}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle error: {e}")
    
    def get_errors(self) -> List[Exception]:
        """Get all errors that occurred"""
        return self._errors.copy()
    
    def clear_errors(self) -> None:
        """Clear all errors"""
        self._errors.clear()

class ConsoleProgressObserver(IProgressObserver):
    """Observer for console progress updates"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConsoleProgressObserver")
        self._errors: List[Exception] = []
    
    def update_progress(self, step: str, progress: float, message: str) -> None:
        """Update progress information in console"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {step}: {progress:.1f}% - {message}")
            self.logger.info(f"Progress update: {step} - {progress}% - {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to update progress: {e}")
    
    def on_error(self, error: Exception, step: str) -> None:
        """Handle errors during processing"""
        try:
            self._errors.append(error)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ERROR in {step}: {str(error)}")
            self.logger.error(f"Error in {step}: {error}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle error: {e}")
    
    def get_errors(self) -> List[Exception]:
        """Get all errors that occurred"""
        return self._errors.copy()
    
    def clear_errors(self) -> None:
        """Clear all errors"""
        self._errors.clear()

class MultiProgressObserver(IProgressObserver):
    """Observer that delegates to multiple observers (Composite pattern)"""
    
    def __init__(self, observers: List[IProgressObserver]):
        self.observers = observers
        self.logger = logging.getLogger(f"{__name__}.MultiProgressObserver")
    
    def add_observer(self, observer: IProgressObserver) -> None:
        """Add a new observer"""
        self.observers.append(observer)
    
    def remove_observer(self, observer: IProgressObserver) -> None:
        """Remove an observer"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def update_progress(self, step: str, progress: float, message: str) -> None:
        """Update all observers"""
        for observer in self.observers:
            try:
                observer.update_progress(step, progress, message)
            except Exception as e:
                self.logger.error(f"Observer failed to update progress: {e}")
    
    def on_error(self, error: Exception, step: str) -> None:
        """Handle errors in all observers"""
        for observer in self.observers:
            try:
                observer.on_error(error, step)
            except Exception as e:
                self.logger.error(f"Observer failed to handle error: {e}")
