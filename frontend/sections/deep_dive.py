"""
Deep Dive Page - Advanced analytics with weekly analysis and detailed metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, date
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.deep_dive_helpers import (
    load_personal_info,
    calculate_enhanced_metrics,
    render_metrics_glossary,
    render_weekly_overview,
    render_body_composition,
    render_metabolic_analysis,
    render_activity_performance,
    render_daily_trends
)


def render(df):
    """Render the deep dive analytics page"""
    st.title("🔬 Deep Dive Analytics")
    
    # Metrics Glossary
    render_metrics_glossary()
    
    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
        return
    
    # Load personal info
    personal_info = load_personal_info()
    has_complete_profile = ('birth_date' in personal_info and personal_info['birth_date'] and
                           'height_cm' in personal_info and personal_info['height_cm'] and
                           'sex' in personal_info and personal_info['sex'])
    has_body_fat = 'body_fat_prct' in personal_info and personal_info['body_fat_prct']
    
    if not has_complete_profile:
        st.warning("Create a personal_info.json file in the data folder to unlock advanced body composition and metabolic analytics.")
        with st.expander("Personal Info JSON Format"):
            st.code('''
{
    "birth_date": "21-01-1992",
    "height_cm": 185,
    "sex": "Male",
    "body_fat_prct": 13
}''')
    
    # Calculate enhanced metrics
    df_enhanced = calculate_enhanced_metrics(df, personal_info, has_complete_profile, has_body_fat)
    
    # Week Selection Interface
    st.markdown("---")
    
    df_enhanced['week_start'] = df_enhanced['date'].dt.to_period('W').dt.start_time
    available_weeks = sorted(df_enhanced['week_start'].unique(), reverse=True)
    
    if len(available_weeks) > 0:
        selected_week_start = _render_week_selector(available_weeks)
        week_end = selected_week_start + pd.Timedelta(days=6)
        
        # Filter data for selected week
        week_mask = (df_enhanced['date'] >= selected_week_start) & (df_enhanced['date'] <= week_end)
        week_data = df_enhanced[week_mask].copy()
        
        complete_week_data = week_data.dropna(subset=['steps', 'calories_burned', 'calories_consumed'])
        days_with_data = len(complete_week_data)
        
        # Week header
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h3 style="color: white; margin: 0; text-align: center;">
                📊 Weekly Analysis: {selected_week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}
            </h3>
            <p style="color: white; margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;">
                Analyzing {days_with_data} days of complete data
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if days_with_data > 0:
            # Render weekly analysis sections
            render_weekly_overview(complete_week_data, has_complete_profile)
            render_body_composition(complete_week_data, has_complete_profile, has_body_fat, personal_info)
            render_metabolic_analysis(complete_week_data, has_complete_profile)
            render_activity_performance(complete_week_data)
    
    # Daily Trends for last 30 days
    render_daily_trends(df_enhanced, has_complete_profile, has_body_fat)


def _render_week_selector(available_weeks):
    """Render week selection dropdown"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        week_options = []
        for week in available_weeks:
            week_end = week + pd.Timedelta(days=6)
            if week == available_weeks[0]:
                week_options.append(f"Current Week ({week.strftime('%b %d')} - {week_end.strftime('%b %d')})")
            else:
                week_options.append(f"{week.strftime('%b %d')} - {week_end.strftime('%b %d')}")
        
        selected_week_idx = st.selectbox("📅 Select Analysis Period", range(len(week_options)), 
                                       format_func=lambda x: week_options[x])
        return available_weeks[selected_week_idx]