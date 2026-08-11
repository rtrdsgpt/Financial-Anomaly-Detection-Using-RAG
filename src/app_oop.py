"""
OOP-based Streamlit application implementing design patterns.
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.streamlit_app import StreamlitApp

def main():
    """Main function to run the OOP-based Streamlit app"""
    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {e}")
        st.info("💡 Try using the command line interface: `python main_oop.py`")

if __name__ == "__main__":
    main()
