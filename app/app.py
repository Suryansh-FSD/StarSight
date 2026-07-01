import streamlit as st
import sys
import os
import importlib.util
from pathlib import Path

# Add project base directory to system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="StarSight Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Render brand header
from utils import inject_custom_css
inject_custom_css()

# Import and execute Home page
home_path = Path(__file__).parent / "pages" / "1_Home.py"
if home_path.exists():
    spec = importlib.util.spec_from_file_location("Home", str(home_path))
    home_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(home_module)
    if hasattr(home_module, "main"):
        home_module.main()
else:
    st.title("StarSight Dashboard")
    st.info("Navigate using the sidebar to explore features.")
