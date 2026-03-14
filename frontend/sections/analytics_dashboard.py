"""
Analytics Dashboard Page - Overview of health metrics with visualizations
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
from utils.metrics_helpers import calculate_weight_trend, calculate_weight_rate_kgperweek, format_sleep_time
from utils.visualization_helpers import (
    create_time_series_plot,
    create_weight_chart,
    create_energy_balance_chart,
    create_steps_workout_chart,
    create_sleep_chart,
)


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
    st.info(
        f"📊 Showing data from **{start_date.strftime('%b %d, %Y')}** to "
        f"**{end_date.strftime('%b %d, %Y')}** ({days_shown} days)"
    )

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

    daily_df = filtered_df.copy()
    filtered_df = _apply_aggregation(filtered_df, aggregation)

    # Render dashboard sections
    _render_kpi_cards(daily_df)
    _render_chart_grid(filtered_df, daily_df)


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
                st.session_state.date_range = (
                    st.session_state.df['date'].min().date(),
                    st.session_state.df['date'].max().date(),
                )

    if 'date_range' not in st.session_state:
        st.session_state.date_range = (today - timedelta(days=30), today)

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


def _kpi_card(label, value, unit, color):
    """Return HTML for a color-coded KPI card."""
    color_map = {
        'green': ('#10b981', '#064e3b'),
        'amber': ('#f59e0b', '#451a03'),
        'red':   ('#ef4444', '#450a0a'),
        'neutral': ('#667eea', '#1e1b4b'),
    }
    border, bg = color_map.get(color, color_map['neutral'])
    return f"""
    <div style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1rem 0.75rem;
        text-align: center;
        min-height: 90px;
    ">
        <div style="color: #9ca3af; font-size: 0.75rem; margin-bottom: 0.25rem;">{label}</div>
        <div style="color: {border}; font-size: 1.5rem; font-weight: 700;">{value}</div>
        <div style="color: #6b7280; font-size: 0.7rem;">{unit}</div>
    </div>
    """


def _threshold_color(value, green_ranges, amber_ranges):
    """
    Return 'green', 'amber', or 'red' based on threshold ranges.
    Each range is a (lo, hi) tuple (inclusive). Returns 'neutral' if value is NaN/None.
    """
    if value is None or pd.isna(value):
        return 'neutral'
    for lo, hi in green_ranges:
        if lo <= value <= hi:
            return 'green'
    for lo, hi in amber_ranges:
        if lo <= value <= hi:
            return 'amber'
    return 'red'


def _render_kpi_cards(daily_df):
    """Render 6 color-coded KPI cards in a single row."""
    st.subheader("📋 Key Metrics")

    latest_weight = daily_df['weight'].dropna().iloc[-1] if daily_df['weight'].notna().any() else None
    weekly_rate = calculate_weight_rate_kgperweek(daily_df, days=7)
    avg_surplus = (daily_df['calories_consumed'] - daily_df['calories_burned']).mean()
    avg_sleep_h = daily_df['sleep_min'].mean() / 60
    avg_steps = daily_df['steps'].mean()
    avg_workout = daily_df['workout_duration_min_tot'].mean()

    cards = [
        (
            "Current Weight",
            f"{latest_weight:.1f}" if not pd.isna(latest_weight) else "N/A",
            "kg",
            'neutral',
        ),
        (
            "Weekly Gain",
            f"{weekly_rate:+.2f}" if not pd.isna(weekly_rate) else "N/A",
            "kg/week",
            _threshold_color(weekly_rate, [(0.20, 0.50)], [(0.10, 0.20), (0.50, 0.70)]),
        ),
        (
            "Daily Surplus",
            f"{avg_surplus:+.0f}" if not pd.isna(avg_surplus) else "N/A",
            "kcal",
            _threshold_color(avg_surplus, [(200, 600)], [(100, 200), (600, 900)]),
        ),
        (
            "Avg Sleep",
            f"{avg_sleep_h:.1f}" if not pd.isna(avg_sleep_h) else "N/A",
            "hours",
            _threshold_color(avg_sleep_h, [(7.5, 99)], [(7.0, 7.5)]),
        ),
        (
            "Avg Steps",
            f"{avg_steps:,.0f}" if not pd.isna(avg_steps) else "N/A",
            "steps/day",
            _threshold_color(avg_steps, [(8000, 1e9)], [(5000, 8000)]),
        ),
        (
            "Avg Workout",
            f"{avg_workout:.0f}" if not pd.isna(avg_workout) else "N/A",
            "min/day",
            _threshold_color(avg_workout, [(60, 1e9)], [(30, 60)]),
        ),
    ]

    cols = st.columns(6)
    for col, (label, value, unit, color) in zip(cols, cards):
        col.markdown(_kpi_card(label, value, unit, color), unsafe_allow_html=True)

    st.markdown("")  # spacing


def _render_chart_grid(filtered_df, daily_df):
    """Render 2×2 chart grid — always visible, no selection needed."""
    st.subheader("📈 Dashboard Charts")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(create_weight_chart(filtered_df), use_container_width=True)
    with row1_col2:
        st.plotly_chart(create_energy_balance_chart(filtered_df), use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(create_steps_workout_chart(filtered_df), use_container_width=True)
    with row2_col2:
        st.plotly_chart(create_sleep_chart(filtered_df), use_container_width=True)
