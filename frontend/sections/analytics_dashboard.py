"""
Analytics Dashboard Page - Overview of health metrics with visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from utils.metrics_helpers import calculate_weight_trend, format_sleep_time
from utils.visualization_helpers import create_time_series_plot


def render(df):
    """Render the analytics dashboard page"""
    st.title("📊 Health Analytics Dashboard")
    
    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
        return
    
    # Time Period Selection
    _render_time_period_selection()
    
    # Get date range from session state
    start_date, end_date = st.session_state.date_range
    days_shown = (end_date - start_date).days + 1
    st.info(f"📊 Showing data from **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}** ({days_shown} days)")
    
    # Aggregation control
    col_agg = st.columns([1, 2])[0]
    with col_agg:
        aggregation = st.selectbox("📊 Data Aggregation", ["Daily", "Weekly", "Monthly"])
    
    # Filter data by date range
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        st.warning("No data in selected date range.")
        return
    
    # Store original daily data for KPIs and trends
    daily_df = filtered_df.copy()
    
    # Apply aggregation for visualizations
    filtered_df = _apply_aggregation(filtered_df, aggregation)
    
    # Render dashboard sections
    _render_weight_trends(daily_df)
    _render_kpis(daily_df)
    _render_time_series(filtered_df, aggregation)
    _render_sleep_analysis(daily_df)


def _render_time_period_selection():
    """Render time period selection controls"""
    st.subheader("📅 Time Period")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    today = date.today()
    
    with col1:
        if st.button("7 Days", use_container_width=True):
            st.session_state.date_range = (today - timedelta(days=7), today)
    with col2:
        if st.button("14 Days", use_container_width=True):
            st.session_state.date_range = (today - timedelta(days=14), today)
    with col3:
        if st.button("30 Days", use_container_width=True):
            st.session_state.date_range = (today - timedelta(days=30), today)
    with col4:
        if st.button("3 Months", use_container_width=True):
            st.session_state.date_range = (today - timedelta(days=90), today)
    with col5:
        if st.button("This Year", use_container_width=True):
            st.session_state.date_range = (date(today.year, 1, 1), today)
    with col6:
        if st.button("All Time", use_container_width=True):
            if 'df' in st.session_state and not st.session_state.df.empty:
                st.session_state.date_range = (st.session_state.df['date'].min().date(), 
                                              st.session_state.df['date'].max().date())
    
    # Initialize default if not set
    if 'date_range' not in st.session_state:
        st.session_state.date_range = (today - timedelta(days=30), today)
    
    # Custom date range in expander
    with st.expander("📝 Custom Date Range"):
        start_date, end_date = st.session_state.date_range
        col_start, col_end = st.columns(2)
        with col_start:
            custom_start = st.date_input("Start Date", value=start_date, key="custom_start")
        with col_end:
            custom_end = st.date_input("End Date", value=end_date, key="custom_end")
        
        if st.button("Apply Custom Range", type="primary"):
            st.session_state.date_range = (custom_start, custom_end)
            st.rerun()


def _apply_aggregation(df, aggregation):
    """Apply data aggregation based on selection"""
    if aggregation == "Weekly":
        return df.set_index('date').resample('W').agg({
            'steps': 'sum',
            'sleep_min': 'mean',
            'workout_duration_min_tot': 'sum',
            'weight': 'mean',
            'calories_burned': 'sum',
            'calories_consumed': 'sum'
        }).reset_index()
    elif aggregation == "Monthly":
        return df.set_index('date').resample('M').agg({
            'steps': 'sum',
            'sleep_min': 'mean',
            'workout_duration_min_tot': 'sum',
            'weight': 'mean',
            'calories_burned': 'sum',
            'calories_consumed': 'sum'
        }).reset_index()
    return df


def _render_weight_trends(daily_df):
    """Render weight trends section"""
    st.subheader("⚖️ Weight Trends")
    trend_cols = st.columns(3)
    
    trend_periods = [3, 5, 7]
    trend_labels = ["3-Day Trend", "5-Day Trend", "7-Day Trend"]
    
    for col, days, label in zip(trend_cols, trend_periods, trend_labels):
        trend = calculate_weight_trend(daily_df, days)
        if trend is not None:
            trend_text = f"+{trend:.2f}%" if trend > 0 else f"{trend:.2f}%"
            col.metric(
                label=f"📊 {label}",
                value=trend_text,
                help=f"Weight change over the last {days} days"
            )
        else:
            col.metric(
                label=f"📊 {label}",
                value="N/A",
                help=f"Insufficient data for {days}-day trend"
            )


def _render_kpis(daily_df):
    """Render key performance indicators"""
    st.subheader("📋 Key Performance Indicators")
    kpi_cols = st.columns(4)
    
    avg_sleep_min = daily_df['sleep_min'].mean()
    avg_sleep_formatted = format_sleep_time(avg_sleep_min)
    
    metrics = [
        ("🚶 Avg Steps", daily_df['steps'].mean(), "{:.0f}"),
        ("😴 Avg Sleep", avg_sleep_formatted, "{}"),
        ("🔥 Avg Burned", daily_df['calories_burned'].mean(), "{:.0f} kcal"),
        ("🍽️ Avg Consumed", daily_df['calories_consumed'].mean(), "{:.0f} kcal")
    ]
    
    for col, (label, value, fmt) in zip(kpi_cols, metrics):
        if isinstance(value, str):
            col.metric(label, value)
        elif pd.notna(value):
            col.metric(label, fmt.format(value))
        else:
            col.metric(label, "N/A")


def _render_time_series(filtered_df, aggregation):
    """Render time series analysis section"""
    st.subheader("📈 Time Series Analysis")
    
    metrics_to_plot = st.multiselect(
        "Select metrics to visualize",
        ['steps', 'sleep_min', 'workout_duration_min_tot', 'weight', 
         'calories_burned', 'calories_consumed'],
        default=['weight', 'calories_burned', 'calories_consumed'],
        help="Choose which health metrics you want to analyze"
    )
    
    for metric in metrics_to_plot:
        if filtered_df[metric].notna().any():
            fig = create_time_series_plot(filtered_df, metric, aggregation)
            st.plotly_chart(fig, use_container_width=True)


def _render_sleep_analysis(daily_df):
    """Render sleep consistency analysis"""
    if daily_df['sleep_min'].notna().sum() >= 7:
        st.subheader("😴 Sleep Consistency Analysis")
        
        sleep_data = daily_df[daily_df['sleep_min'].notna()].copy()
        sleep_data['sleep_hours'] = sleep_data['sleep_min'] / 60
        
        sleep_std = sleep_data['sleep_hours'].std()
        sleep_mean = sleep_data['sleep_hours'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if sleep_std < 0.5:
                st.success(f"🎯 Excellent sleep consistency! (±{sleep_std*60:.0f} minutes)")
            elif sleep_std < 1:
                st.warning(f"⚠️ Moderate sleep variability (±{sleep_std*60:.0f} minutes)")
            else:
                st.error(f"❗ High sleep variability (±{sleep_std*60:.0f} minutes)")
        
        with col2:
            if sleep_mean >= 7 and sleep_mean <= 9:
                st.success(f"✅ Healthy average sleep: {sleep_mean:.1f} hours")
            elif sleep_mean < 7:
                st.warning(f"😴 Below recommended: {sleep_mean:.1f} hours")
            else:
                st.info(f"😴 Above average: {sleep_mean:.1f} hours")