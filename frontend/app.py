"""
Health Tracker Application - Main Entry Point
"""

import streamlit as st
import pandas as pd
from datetime import date
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from data_utils import load_data, save_data, impute_missing_values

# Import page modules
from sections import data_entry, analytics_dashboard, deep_dive, weekly_report

# Page configuration
st.set_page_config(
    page_title="Health Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = 'data/health_data.csv'
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'height_cm': None,
        'age': None,
        'sex': None,
        'activity_level': 'moderate'
    }
if 'imputation_method' not in st.session_state:
    st.session_state.imputation_method = 'none'

# Sidebar navigation
st.sidebar.title("🏃 Health Tracker")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["📊 Analytics Dashboard", "➕ Data Entry", "🔬 Deep Dive", "📋 Weekly Report"]
)

# Load data
try:
    df = load_data(st.session_state.csv_path)
except FileNotFoundError:
    # Create sample data if file doesn't exist
    df = pd.DataFrame(columns=['user_id', 'date', 'steps', 'sleep_min', 
                            'workout_duration_min_tot', 'weight', 
                            'calories_burned', 'calories_consumed'])
    save_data(df, st.session_state.csv_path)

# Apply imputation if selected
if st.session_state.imputation_method != 'none' and not df.empty:
    df = impute_missing_values(df, method=st.session_state.imputation_method)

# Route to appropriate page
if page == "📊 Analytics Dashboard":
    analytics_dashboard.render(df)
elif page == "➕ Data Entry":
    data_entry.render(df)
elif page == "🔬 Deep Dive":
    deep_dive.render(df)
elif page == "📋 Weekly Report":
    weekly_report.render(df)