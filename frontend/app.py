"""
Health Tracker Application
A comprehensive health metrics tracking and analysis tool
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
import shutil
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import utility modules (we'll create these next)
from data_utils import (
    load_data, save_data, append_record, update_record, delete_record,
    validate_record, create_backup, parse_date
)
from analytics import (
    compute_bmi, estimate_bmr, compute_tdee, compute_energy_balance,
    compute_sleep_consistency, compute_activity_load, detect_anomalies,
    compute_trend_with_significance, rolling_stats
)
from visualization import (
    create_time_series_plot, create_correlation_heatmap,
    create_distribution_plot, create_kpi_cards
)

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
    ["📊 Dashboard", "➕ Data Entry", "📈 Visualization", "🔬 Deep Dive"]
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
    numeric_cols = ['steps', 'sleep_min', 'workout_duration_min_tot', 
                    'weight', 'calories_burned', 'calories_consumed']
    
    if st.session_state.imputation_method == 'forward_fill':
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    elif st.session_state.imputation_method == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

# Dashboard Page
if page == "📊 Dashboard":
    st.title("📊 Health Dashboard")
    
    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
    else:
        # Quick Date Filter - Option 1 Implementation
        st.subheader("📅 Time Period")
        
        # Quick selection buttons
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
                if not df.empty:
                    st.session_state.date_range = (df['date'].min().date(), df['date'].max().date())
        
        # Initialize default if not set
        if 'date_range' not in st.session_state:
            st.session_state.date_range = (today - timedelta(days=30), today)
        
        # Get current selection
        start_date, end_date = st.session_state.date_range
        
        # Show current selection
        days_shown = (end_date - start_date).days + 1
        st.info(f"📊 Showing data from **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}** ({days_shown} days)")
        
        # Optional: Add custom date range in expander
        with st.expander("📝 Custom Date Range"):
            col_start, col_end = st.columns(2)
            with col_start:
                custom_start = st.date_input("Start Date", value=start_date, key="custom_start")
            with col_end:
                custom_end = st.date_input("End Date", value=end_date, key="custom_end")
            
            if st.button("Apply Custom Range", type="primary"):
                st.session_state.date_range = (custom_start, custom_end)
                st.rerun()
        
        # Filter data by date range
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            st.warning("No data in selected date range.")
        else:
            # Helper function to calculate weight trend
            def calculate_weight_trend(df, days):
                """Calculate weight trend percentage over the last N days"""
                if len(df) < 2 or df['weight'].isna().all():
                    return None
                
                # Sort by date and get the last N days with weight data
                weight_data = df[df['weight'].notna()].sort_values('date')
                
                if len(weight_data) < 2:
                    return None
                
                # Get the last N days of data
                if len(weight_data) >= days:
                    recent_data = weight_data.tail(days)
                else:
                    recent_data = weight_data
                
                if len(recent_data) < 2:
                    return None
                
                # Calculate trend as percentage change
                first_weight = recent_data['weight'].iloc[0]
                last_weight = recent_data['weight'].iloc[-1]
                
                if first_weight == 0:
                    return None
                
                trend_percent = ((last_weight - first_weight) / first_weight) * 100
                return trend_percent
            
            # Helper function to format sleep time
            def format_sleep_time(minutes):
                """Convert minutes to hours and minutes format"""
                if pd.isna(minutes):
                    return "N/A"
                hours = int(minutes // 60)
                mins = int(minutes % 60)
                return f"{hours}h {mins}m"
            
            # Weight trends section
            st.subheader("⚖️ Weight Trends")
            trend_cols = st.columns(3)
            
            # Calculate trends for 3, 5, and 7 days
            trend_periods = [3, 5, 7]
            trend_labels = ["3-Day Trend", "5-Day Trend", "7-Day Trend"]
            
            for col, days, label in zip(trend_cols, trend_periods, trend_labels):
                trend = calculate_weight_trend(filtered_df, days)
                if trend is not None:
                    trend_color = "normal"
                    if trend > 0:
                        trend_text = f"+{trend:.2f}%"
                        trend_color = "normal"  # Weight gain - neutral color
                    else:
                        trend_text = f"{trend:.2f}%"
                        trend_color = "normal"  # Weight loss - neutral color
                    
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
            
            # KPI Cards (updated - removed Total Workout and Avg Weight)
            st.subheader("Key Performance Indicators")
            kpi_cols = st.columns(4)
            
            # Calculate average sleep in hours and minutes format
            avg_sleep_min = filtered_df['sleep_min'].mean()
            avg_sleep_formatted = format_sleep_time(avg_sleep_min)
            
            metrics = [
                ("🚶 Avg Steps", filtered_df['steps'].mean(), "{:.0f}"),
                ("😴 Avg Sleep", avg_sleep_formatted, "{}"),
                ("🔥 Avg Burned", filtered_df['calories_burned'].mean(), "{:.0f} kcal"),
                ("🍽️ Avg Consumed", filtered_df['calories_consumed'].mean(), "{:.0f} kcal")
            ]
            
            for col, (label, value, fmt) in zip(kpi_cols, metrics):
                if isinstance(value, str):  # For formatted sleep time
                    col.metric(label, value)
                elif pd.notna(value):
                    col.metric(label, fmt.format(value))
                else:
                    col.metric(label, "N/A")
            
            # Quick visualization
            st.subheader("Recent Trends")
            
            # Weight trend chart (full width)
            if filtered_df['weight'].notna().any():
                fig = px.line(filtered_df, x='date', y='weight', 
                             title='Weight Trend (kg)', markers=True)
                fig.update_layout(height=350)
                
                # Add trend line if we have enough data
                weight_data = filtered_df[filtered_df['weight'].notna()].sort_values('date')
                if len(weight_data) >= 2:
                    # Add a trend line
                    z = np.polyfit(range(len(weight_data)), weight_data['weight'], 1)
                    p = np.poly1d(z)
                    fig.add_scatter(x=weight_data['date'], 
                                  y=p(range(len(weight_data))),
                                  mode='lines', 
                                  name='Trend Line',
                                  line=dict(dash='dash', color='red'))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Calorie Balance chart (full width)
            if (filtered_df['calories_burned'].notna().any() and 
                filtered_df['calories_consumed'].notna().any()):
                
                # Create figure with dual lines
                fig = go.Figure()
                
                # Add calories burned line
                fig.add_trace(go.Scatter(
                    x=filtered_df['date'],
                    y=filtered_df['calories_burned'],
                    mode='lines+markers',
                    name='🔥 Burned',
                    line=dict(color='#FF6B6B', width=2),
                    marker=dict(size=6)
                ))
                
                # Add calories consumed line
                fig.add_trace(go.Scatter(
                    x=filtered_df['date'],
                    y=filtered_df['calories_consumed'],
                    mode='lines+markers',
                    name='🍽️ Consumed',
                    line=dict(color='#4ECDC4', width=2),
                    marker=dict(size=6)
                ))
                
                # Calculate and add balance area (optional)
                calories_df = filtered_df[['date', 'calories_burned', 'calories_consumed']].dropna()
                if not calories_df.empty:
                    # Add fill between curves to show surplus/deficit
                    fig.add_trace(go.Scatter(
                        x=calories_df['date'].tolist() + calories_df['date'].tolist()[::-1],
                        y=calories_df['calories_burned'].tolist() + calories_df['calories_consumed'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 107, 107, 0.1)',  # Light red for deficit
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                fig.update_layout(
                    title='Calorie Balance',
                    xaxis_title='Date',
                    yaxis_title='Calories (kcal)',
                    height=350,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif filtered_df['calories_burned'].notna().any():
                # Show only calories burned if consumed data is missing
                fig = px.line(filtered_df, x='date', y='calories_burned', 
                             title='Calories Burned', markers=True)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            elif filtered_df['calories_consumed'].notna().any():
                # Show only calories consumed if burned data is missing
                fig = px.line(filtered_df, x='date', y='calories_consumed', 
                             title='Calories Consumed', markers=True)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.subheader("📋 Quick Insights")
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                # Sleep consistency
                if filtered_df['sleep_min'].notna().sum() >= 7:
                    sleep_std = filtered_df['sleep_min'].std()
                    if sleep_std < 30:
                        st.success("🎯 Great sleep consistency!")
                    elif sleep_std < 60:
                        st.warning("⚠️ Moderate sleep variability")
                    else:
                        st.error("❗ High sleep variability - consider a more regular schedule")
            
            with insights_col2:
                # Calorie balance trend
                if (filtered_df['calories_burned'].notna().any() and 
                    filtered_df['calories_consumed'].notna().any()):
                    avg_balance = (filtered_df['calories_burned'].mean() - 
                                 filtered_df['calories_consumed'].mean())
                    if avg_balance > 200:
                        st.info("📉 Caloric deficit - good for weight loss")
                    elif avg_balance < -200:
                        st.info("📈 Caloric surplus - good for weight gain")
                    else:
                        st.info("⚖️ Balanced caloric intake")
# Data Entry Page
elif page == "➕ Data Entry":
    st.title("➕ Data Entry & Management")
    
    # Convert date column to datetime if it's not already
    if not df.empty and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Main interface
    st.subheader("📅 Select or Add Date")
    
    # Date selection/input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_date = st.date_input(
            "Choose date to edit or add new record:",
            value=date.today(),
            help="Select an existing date to edit, or a new date to create a record"
        )
    
    with col2:
        # Check if date exists in data
        date_exists = False
        existing_row = None
        
        if not df.empty:
            df_date_check = df[df['date'].dt.date == selected_date]
            if len(df_date_check) > 0:
                date_exists = True
                existing_row = df_date_check.iloc[0]
                st.success("📝 Date exists - Edit mode")
            else:
                st.info("🆕 New date - Add mode")
        else:
            st.info("🆕 New date - Add mode")
    
    # Data entry form
    st.subheader("📊 Health Data Entry")
    
    with st.form("health_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Activity & Sleep**")
            steps = st.number_input(
                "Steps (count)", 
                min_value=0, 
                value=int(existing_row['steps']) if date_exists and pd.notna(existing_row['steps']) else 0,
                help="Daily step count"
            )
            
            sleep_min = st.number_input(
                "Sleep (minutes)", 
                min_value=0, 
                max_value=1440,
                value=int(existing_row['sleep_min']) if date_exists and pd.notna(existing_row['sleep_min']) else 0,
                help="Total sleep time in minutes"
            )
            
            workout_min = st.number_input(
                "Workout duration (minutes)", 
                min_value=0,
                value=int(existing_row['workout_duration_min_tot']) if date_exists and pd.notna(existing_row['workout_duration_min_tot']) else 0,
                help="Total workout time in minutes"
            )
        
        with col2:
            st.write("**Body & Nutrition**")
            weight = st.number_input(
                "Weight (kg)", 
                min_value=30.0, 
                max_value=300.0,
                value=float(existing_row['weight']) if date_exists and pd.notna(existing_row['weight']) else 70.0,
                step=0.1,
                help="Body weight in kilograms"
            )
            
            calories_burned = st.number_input(
                "Calories burned (kcal)", 
                min_value=0,
                value=int(existing_row['calories_burned']) if date_exists and pd.notna(existing_row['calories_burned']) else 0,
                help="Total calories burned"
            )
            
            calories_consumed = st.number_input(
                "Calories consumed (kcal)", 
                min_value=0,
                value=int(existing_row['calories_consumed']) if date_exists and pd.notna(existing_row['calories_consumed']) else 0,
                help="Total calories consumed"
            )
        
        # Form buttons
        st.markdown("---")
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            submit_btn = st.form_submit_button("💾 Save Record", type="primary", use_container_width=True)
        
        with col_btn2:
            if date_exists:
                delete_btn = st.form_submit_button("🗑️ Delete Record", type="secondary", use_container_width=True)
            else:
                delete_btn = False
        
        with col_btn3:
            clear_btn = st.form_submit_button("🧹 Clear Form", use_container_width=True)
        
        with col_btn4:
            # Show calorie balance
            balance = calories_burned - calories_consumed
            if balance > 0:
                st.success(f"📉 Deficit: {balance} kcal")
            elif balance < 0:
                st.error(f"📈 Surplus: {abs(balance)} kcal")
            else:
                st.info("⚖️ Balanced")
    
    # Handle form submissions
    if submit_btn:
        # Prepare the new record
        new_record = {
            'user_id': '',
            'date': pd.Timestamp(selected_date),
            'steps': steps if steps > 0 else None,
            'sleep_min': sleep_min if sleep_min > 0 else None,
            'workout_duration_min_tot': workout_min if workout_min > 0 else None,
            'weight': weight if weight > 0 else None,
            'calories_burned': calories_burned if calories_burned > 0 else None,
            'calories_consumed': calories_consumed if calories_consumed > 0 else None
        }
        
        try:
            if date_exists:
                # Update existing record
                df.loc[df['date'].dt.date == selected_date, 'steps'] = new_record['steps']
                df.loc[df['date'].dt.date == selected_date, 'sleep_min'] = new_record['sleep_min']
                df.loc[df['date'].dt.date == selected_date, 'workout_duration_min_tot'] = new_record['workout_duration_min_tot']
                df.loc[df['date'].dt.date == selected_date, 'weight'] = new_record['weight']
                df.loc[df['date'].dt.date == selected_date, 'calories_burned'] = new_record['calories_burned']
                df.loc[df['date'].dt.date == selected_date, 'calories_consumed'] = new_record['calories_consumed']
                
                # Save the updated dataframe
                save_data(df, st.session_state.csv_path)
                st.success(f"✅ Record for {selected_date} updated successfully!")
                
            else:
                # Add new record
                new_row_df = pd.DataFrame([new_record])
                df_updated = pd.concat([df, new_row_df], ignore_index=True)
                df_updated = df_updated.sort_values('date')
                
                # Save the updated dataframe
                save_data(df_updated, st.session_state.csv_path)
                st.success(f"✅ New record for {selected_date} added successfully!")
            
            # Refresh the page to show updated data
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error saving record: {str(e)}")
    
    if delete_btn and date_exists:
        try:
            # Remove the record for the selected date
            df_updated = df[df['date'].dt.date != selected_date].copy()
            
            # Save the updated dataframe
            save_data(df_updated, st.session_state.csv_path)
            st.success(f"🗑️ Record for {selected_date} deleted successfully!")
            
            # Refresh the page
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error deleting record: {str(e)}")
    
    if clear_btn:
        st.rerun()
    
    # Display recent data table
    st.markdown("---")
    st.subheader("📋 Recent Records")
    
    if df.empty:
        st.info("No records available. Add your first record above!")
    else:
        # Show last 10 records
        display_df = df.sort_values('date', ascending=False).head(10).copy()
        
        # Format the data for display
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        # Format numeric columns
        numeric_columns = ['steps', 'sleep_min', 'workout_duration_min_tot', 'weight', 'calories_burned', 'calories_consumed']
        for col in numeric_columns:
            if col == 'weight':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
        
        # Display the table
        st.dataframe(
            display_df[['date', 'steps', 'sleep_min', 'workout_duration_min_tot', 'weight', 'calories_burned', 'calories_consumed']],
            use_container_width=True,
            column_config={
                'date': st.column_config.TextColumn('Date', width='medium'),
                'steps': st.column_config.TextColumn('Steps', width='small'),
                'sleep_min': st.column_config.TextColumn('Sleep (min)', width='small'), 
                'workout_duration_min_tot': st.column_config.TextColumn('Workout (min)', width='small'),
                'weight': st.column_config.TextColumn('Weight (kg)', width='small'),
                'calories_burned': st.column_config.TextColumn('Cal. Burned', width='small'),
                'calories_consumed': st.column_config.TextColumn('Cal. Consumed', width='small')
            },
            hide_index=True
        )
        
        # Quick summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = len(df)
            st.metric("📊 Total Records", total_records)
        
# Visualization Page
elif page == "📈 Visualization":
    st.title("📈 Data Visualization")
    
    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
    else:
        # Quick Date Filter - Same as Dashboard
        st.subheader("📅 Time Period")
        
        # Quick selection buttons
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        today = date.today()
        
        with col1:
            if st.button("7 Days", use_container_width=True, key="viz_7d"):
                st.session_state.viz_date_range = (today - timedelta(days=7), today)
        with col2:
            if st.button("14 Days", use_container_width=True, key="viz_14d"):
                st.session_state.viz_date_range = (today - timedelta(days=14), today)
        with col3:
            if st.button("30 Days", use_container_width=True, key="viz_30d"):
                st.session_state.viz_date_range = (today - timedelta(days=30), today)
        with col4:
            if st.button("3 Months", use_container_width=True, key="viz_3m"):
                st.session_state.viz_date_range = (today - timedelta(days=90), today)
        with col5:
            if st.button("This Year", use_container_width=True, key="viz_year"):
                st.session_state.viz_date_range = (date(today.year, 1, 1), today)
        with col6:
            if st.button("All Time", use_container_width=True, key="viz_all"):
                if not df.empty:
                    st.session_state.viz_date_range = (df['date'].min().date(), df['date'].max().date())
        
        # Initialize default if not set
        if 'viz_date_range' not in st.session_state:
            st.session_state.viz_date_range = (today - timedelta(days=30), today)
        
        # Get current selection
        start_date, end_date = st.session_state.viz_date_range
        
        # Show current selection
        days_shown = (end_date - start_date).days + 1
        st.info(f"📊 Showing data from **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}** ({days_shown} days)")
        
        # Optional: Add custom date range in expander
        with st.expander("📝 Custom Date Range"):
            col_start, col_end = st.columns(2)
            with col_start:
                custom_start = st.date_input("Start Date", value=start_date, key="viz_custom_start")
            with col_end:
                custom_end = st.date_input("End Date", value=end_date, key="viz_custom_end")
            
            if st.button("Apply Custom Range", type="primary", key="viz_custom_apply"):
                st.session_state.viz_date_range = (custom_start, custom_end)
                st.rerun()
        
        # Aggregation control
        col_agg = st.columns([1, 2])[0]  # Take only first column for aggregation
        with col_agg:
            aggregation = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"])
        
        # Filter data by date range
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            st.warning("No data in selected date range.")
        else:
            # Aggregation
            if aggregation == "Weekly":
                filtered_df = filtered_df.set_index('date').resample('W').agg({
                    'steps': 'sum',
                    'sleep_min': 'mean',
                    'workout_duration_min_tot': 'sum',
                    'weight': 'mean',
                    'calories_burned': 'sum',
                    'calories_consumed': 'sum'
                }).reset_index()
            elif aggregation == "Monthly":
                filtered_df = filtered_df.set_index('date').resample('M').agg({
                    'steps': 'sum',
                    'sleep_min': 'mean',
                    'workout_duration_min_tot': 'sum',
                    'weight': 'mean',
                    'calories_burned': 'sum',
                    'calories_consumed': 'sum'
                }).reset_index()
            
            # Time series plots with moving averages
            st.subheader("Time Series Analysis")
            
            metrics_to_plot = st.multiselect(
                "Select metrics to plot",
                ['steps', 'sleep_min', 'workout_duration_min_tot', 'weight', 'calories_burned', 'calories_consumed'],
                default=['calories_burned', 'calories_consumed']
            )
            
            for metric in metrics_to_plot:
                if filtered_df[metric].notna().any():
                    fig = go.Figure()
                    
                    # Original data
                    fig.add_trace(go.Scatter(
                        x=filtered_df['date'],
                        y=filtered_df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))
                    
                    # Add moving averages if daily data
                    if aggregation == "Daily" and len(filtered_df) > 7:
                        # 7-day moving average
                        ma7 = filtered_df[metric].rolling(window=7, min_periods=1).mean()
                        fig.add_trace(go.Scatter(
                            x=filtered_df['date'],
                            y=ma7,
                            mode='lines',
                            name='7-day MA',
                            line=dict(dash='dash', width=1)
                        ))
                        
                        # 30-day moving average
                        if len(filtered_df) > 30:
                            ma30 = filtered_df[metric].rolling(window=30, min_periods=1).mean()
                            fig.add_trace(go.Scatter(
                                x=filtered_df['date'],
                                y=ma30,
                                mode='lines',
                                name='30-day MA',
                                line=dict(dash='dot', width=1)
                            ))
                    
                    fig.update_layout(
                        title=f"{metric.replace('_', ' ').title()} Over Time",
                        xaxis_title="Date",
                        yaxis_title=metric.replace('_', ' ').title(),
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            st.subheader("Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if filtered_df['steps'].notna().any():
                    fig = px.histogram(filtered_df, x='steps', title='Steps Distribution',
                                      nbins=20, marginal='box')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if filtered_df['sleep_min'].notna().any():
                    fig = px.histogram(filtered_df, x='sleep_min', title='Sleep Distribution',
                                      nbins=20, marginal='box')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

# Deep Dive Page
elif page == "🔬 Deep Dive":
    st.title("🔬 Deep Dive Analytics")
    
    st.info("⚠️ **Disclaimer**: This is informational only and not medical advice. Please consult healthcare professionals for medical guidance.")
    
    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
    else:
        # Load personal info from JSON
        personal_info_path = 'data/personal_info.json'
        personal_info = {}
        
        try:
            if os.path.exists(personal_info_path):
                import json
                with open(personal_info_path, 'r') as f:
                    personal_info = json.load(f)
        except Exception as e:
            st.error(f"Error loading personal info: {e}")
        
        # Check if personal info is available
        has_birth_date = 'birth_date' in personal_info and personal_info['birth_date']
        has_height = 'height_cm' in personal_info and personal_info['height_cm']
        has_sex = 'sex' in personal_info and personal_info['sex']
        has_body_fat = 'body_fat_prct' in personal_info and personal_info['body_fat_prct']
        has_complete_profile = has_birth_date and has_height and has_sex
        
        if not has_complete_profile:
            st.warning("Create a personal_info.json file in the data folder to unlock advanced body composition and metabolic analytics.")
            with st.expander("Personal Info JSON Format"):
                st.code('''
                        {
                        "birth_date": "21-01-1992",
                        "height_cm": 185,
                        "sex": "Male",
                        "body_fat_prct": 13
                        }'''
                    )
        
        # Calculate daily-level metrics for all records
        df_enhanced = df.copy()
        
        if has_complete_profile:
            # Calculate age
            from datetime import datetime
            birth_date = datetime.strptime(personal_info['birth_date'], '%d-%m-%Y').date()
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            height_cm = personal_info['height_cm']
            height_m = height_cm / 100
            sex = personal_info['sex']
            body_fat_prct = personal_info.get('body_fat_prct', None)
            
            # Calculate daily BMI (using current day's weight for body composition metrics)
            df_enhanced['bmi'] = df_enhanced['weight'] / (height_m ** 2)
            
            # Calculate Fat-Free Mass Index (FFMI) if body fat is provided (using current day's weight)
            if body_fat_prct is not None:
                df_enhanced['fat_mass'] = df_enhanced['weight'] * (body_fat_prct / 100)
                df_enhanced['fat_free_mass'] = df_enhanced['weight'] * (1 - body_fat_prct/100)
                df_enhanced['ffmi'] = df_enhanced['fat_free_mass'] / (height_m ** 2)
                df_enhanced['fmi'] = df_enhanced['fat_mass'] / (height_m ** 2)
                df_enhanced['normalized_ffmi'] = df_enhanced['ffmi'] + 6.1 * (1.8 - height_m)
            
            # For metabolic calculations, use previous day's weight (lag by 1)
            # Today's calories burned are based on yesterday's body composition
            df_enhanced['prev_weight'] = df_enhanced['weight'].shift(1)
            
            # Calculate daily BMR using previous day's weight (Mifflin-St Jeor equation)
            if sex.lower() == 'male':
                df_enhanced['bmr'] = 10 * df_enhanced['prev_weight'] + 6.25 * height_cm - 5 * age + 5
            else:
                df_enhanced['bmr'] = 10 * df_enhanced['prev_weight'] + 6.25 * height_cm - 5 * age - 161
            
            # For the first day where prev_weight is NaN, use current weight as fallback
            df_enhanced['bmr'] = df_enhanced['bmr'].fillna(
                10 * df_enhanced['weight'] + 6.25 * height_cm - 5 * age + (5 if sex.lower() == 'male' else -161)
            )
            
            # Calculate daily TDEE based on actual calories burned (NO LAG - current day values)
            df_enhanced['actual_tdee'] = df_enhanced['calories_burned']
            
            # Calculate daily energy balance (NO LAG - current day values)
            df_enhanced['energy_balance'] = df_enhanced['calories_burned'] - df_enhanced['calories_consumed']
            
            # Calculate daily deficit/surplus relative to BMR (using lagged BMR with current intake)
            df_enhanced['bmr_surplus'] = df_enhanced['calories_consumed'] - df_enhanced['bmr']
            
            # Calculate additional metabolic metrics (using lagged BMR with current values)
            df_enhanced['bmr_ratio'] = df_enhanced['calories_burned'] / df_enhanced['bmr']  # Metabolic multiple
            df_enhanced['calories_per_step'] = df_enhanced['calories_burned'] / df_enhanced['steps']  # Calorie efficiency
            
            # Estimate NEAT using previous day's weight for BMR calculation
            # NEAT = Total Calories - BMR - TEF - Exercise Calories
            # User's activities:
            # - Weight lifting (6-7/10 intensity): ~8-10 kcal/min
            # - Basketball (8-9/10 intensity): ~12-15 kcal/min  
            # - Cycling commute (3-4/10 intensity): ~5-7 kcal/min - borderline NEAT/exercise
            # - Walking: ~3-4 kcal/min - part of NEAT
            
            # Strategy: Only subtract high-intensity structured exercise (gym sessions)
            # Keep cycling and walking as part of NEAT since they're transportation/lifestyle
            
            # Assume structured gym sessions are 60-75 minutes
            # Cycling commutes and walking remain in NEAT calculation
            structured_exercise_minutes = np.minimum(df_enhanced['workout_duration_min_tot'], 75)
            
            # Only subtract calories from deliberate gym training
            # Use 10 kcal/min for structured gym sessions (weight lifting + basketball average)
            estimated_exercise_calories = structured_exercise_minutes * 10
            
            estimated_tef = df_enhanced['calories_consumed'] * 0.1  # 10% TEF
            df_enhanced['neat_estimate'] = df_enhanced['calories_burned'] - df_enhanced['bmr'] - estimated_tef - estimated_exercise_calories
        
        # Calculate daily sleep efficiency and recovery metrics (NO LAG - current day values)
        df_enhanced['sleep_hours'] = df_enhanced['sleep_min'] / 60
        df_enhanced['sleep_efficiency'] = np.where(
            df_enhanced['sleep_min'] >= 420,  # 7+ hours
            np.minimum(100, (df_enhanced['sleep_min'] / 480) * 100),  # Cap at 100% for 8+ hours
            (df_enhanced['sleep_min'] / 420) * 100  # Scale below 7 hours
        )
        
        # Calculate daily activity intensity (NO LAG - current day values)
        df_enhanced['steps_per_workout_min'] = np.where(
            df_enhanced['workout_duration_min_tot'] > 0,
            df_enhanced['steps'] / df_enhanced['workout_duration_min_tot'],
            np.nan
        )
        
        # Current Status Dashboard
        # Use most recent day with complete data, not necessarily today
        df_complete = df_enhanced.dropna(subset=['steps', 'calories_burned', 'calories_consumed']).sort_values('date')
        
        if len(df_complete) > 0:
            latest_data = df_complete.iloc[-1]
        else:
            # Fallback to most recent row if no complete data
            latest_data = df_enhanced.sort_values('date').iloc[-1]
        
        st.subheader("📊 Current Health Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if has_complete_profile and pd.notna(latest_data.get('bmr')):
                st.metric("BMR", f"{latest_data['bmr']:.0f} kcal")
                st.caption("Basal Metabolic Rate")
            else:
                st.metric("BMR", "N/A")
        
        with col2:
            if pd.notna(latest_data.get('sleep_efficiency')):
                eff = latest_data['sleep_efficiency']
                st.metric("Sleep Efficiency", f"{eff:.0f}%")
                if eff >= 90:
                    st.success("Excellent")
                elif eff >= 75:
                    st.info("Good")
                else:
                    st.warning("Needs improvement")
            else:
                st.metric("Sleep Efficiency", "N/A")
        
        with col3:
            if pd.notna(latest_data.get('energy_balance')):
                balance = latest_data['energy_balance']
                st.metric("Energy Balance", f"{balance:+.0f} kcal")
                if balance > 200:
                    st.info("Deficit")
                elif balance < -200:
                    st.warning("Surplus")
                else:
                    st.success("Balanced")
            else:
                st.metric("Energy Balance", "N/A")
        
        with col4:
            if pd.notna(latest_data.get('weight')):
                st.metric("Weight", f"{latest_data['weight']:.1f} kg")
                st.caption("Current body weight")
            else:
                st.metric("Weight", "N/A")
        
        # Body Composition Analysis (prioritized over BMI)
        st.subheader("💪 Advanced Body Composition Analysis")
        
        if has_complete_profile and has_body_fat:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if pd.notna(latest_data.get('ffmi')):
                    ffmi_val = latest_data['ffmi']
                    st.metric("FFMI", f"{ffmi_val:.1f}")
                    
                    # FFMI interpretation
                    if sex.lower() == 'male':
                        if ffmi_val < 17:
                            st.info("Below average muscle mass")
                        elif ffmi_val < 20:
                            st.success("Average muscle mass")
                        elif ffmi_val < 23:
                            st.success("Above average muscle mass")
                        elif ffmi_val < 25:
                            st.warning("Excellent muscle mass")
                        else:
                            st.error("Exceptional (may indicate PED use)")
                    else:
                        if ffmi_val < 14:
                            st.info("Below average muscle mass")
                        elif ffmi_val < 17:
                            st.success("Average muscle mass")
                        elif ffmi_val < 19:
                            st.success("Above average muscle mass")
                        elif ffmi_val < 21:
                            st.warning("Excellent muscle mass")
                        else:
                            st.error("Exceptional (may indicate PED use)")
                    
                    st.caption("Fat-Free Mass Index")
                else:
                    st.metric("FFMI", "N/A")
            
            with col2:
                if pd.notna(latest_data.get('fmi')):
                    fmi_val = latest_data['fmi']
                    st.metric("FMI", f"{fmi_val:.1f}")
                    
                    # FMI interpretation (Fat Mass Index)
                    if sex.lower() == 'male':
                        if fmi_val < 3:
                            st.warning("Very low body fat")
                        elif fmi_val < 6:
                            st.success("Healthy range")
                        elif fmi_val < 9:
                            st.info("Moderate")
                        else:
                            st.warning("High fat mass")
                    else:
                        if fmi_val < 5:
                            st.warning("Very low body fat")
                        elif fmi_val < 9:
                            st.success("Healthy range")
                        elif fmi_val < 13:
                            st.info("Moderate")
                        else:
                            st.warning("High fat mass")
                    
                    st.caption("Fat Mass Index")
                else:
                    st.metric("FMI", "N/A")
            
            with col3:
                if pd.notna(latest_data.get('fat_free_mass')):
                    ffm = latest_data['fat_free_mass']
                    st.metric("Fat-Free Mass", f"{ffm:.1f} kg")
                    st.caption(f"Body fat: {body_fat_prct}%")
                else:
                    st.metric("Fat-Free Mass", "N/A")
            
            with col4:
                if pd.notna(latest_data.get('normalized_ffmi')):
                    norm_ffmi = latest_data['normalized_ffmi']
                    st.metric("Normalized FFMI", f"{norm_ffmi:.1f}")
                    st.caption("Height-adjusted FFMI")
                else:
                    st.metric("Normalized FFMI", "N/A")
        
        else:
            st.info("Add 'body_fat_prct' to your personal_info.json to unlock advanced body composition metrics (FFMI, FMI, fat-free mass analysis)")
            
            # Show BMI only when body composition data is unavailable
            if has_complete_profile:
                st.subheader("⚠️ Basic Body Weight Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if pd.notna(latest_data.get('bmi')):
                        bmi_val = latest_data['bmi']
                        st.metric("BMI", f"{bmi_val:.1f}")
                        
                        if bmi_val < 18.5:
                            st.info("BMI: Underweight")
                        elif bmi_val < 25:
                            st.success("BMI: Normal range")
                        elif bmi_val < 30:
                            st.warning("BMI: Overweight")
                        else:
                            st.error("BMI: Obese range")
                    else:
                        st.metric("BMI", "N/A")
                
                with col2:
                    st.warning("⚠️ BMI Limitations")
                    st.caption("BMI doesn't distinguish between muscle and fat mass. For athletes and those building muscle, body composition metrics (FFMI, body fat %) are far more meaningful.")
        
        # Metabolic Health Analysis
        st.subheader("🔥 Advanced Metabolic Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if has_complete_profile and pd.notna(latest_data.get('bmr_ratio')):
                ratio = latest_data['bmr_ratio']
                st.metric("Metabolic Multiple", f"{ratio:.1f}x")
                
                if ratio < 1.3:
                    st.warning("Low activity level")
                elif ratio < 1.6:
                    st.info("Moderate activity")
                elif ratio < 2.0:
                    st.success("Active lifestyle")
                else:
                    st.success("Very active")
                
                st.caption("Total burn / BMR")
            else:
                st.metric("Metabolic Multiple", "N/A")
        
        with col2:
            if has_complete_profile and pd.notna(latest_data.get('neat_estimate')):
                neat = latest_data['neat_estimate']
                st.metric("NEAT Estimate", f"{neat:.0f} kcal")
                
                if neat < 200:
                    st.warning("Low NEAT - try more movement")
                elif neat < 400:
                    st.info("Moderate NEAT")
                else:
                    st.success("High NEAT - good fidgeting!")
                
                st.caption("Non-Exercise Activity")
            else:
                st.metric("NEAT Estimate", "N/A")
        
        with col3:
            if pd.notna(latest_data.get('calories_per_step')):
                cal_per_step = latest_data['calories_per_step']
                st.metric("Calorie Efficiency", f"{cal_per_step:.3f}")
                st.caption("Calories per step")
            else:
                st.metric("Calorie Efficiency", "N/A")
        
        with col4:
            if has_complete_profile and pd.notna(latest_data.get('bmr_surplus')):
                surplus = latest_data['bmr_surplus']
                st.metric("BMR Surplus", f"{surplus:+.0f} kcal")
                
                if surplus < 0:
                    st.error("Eating below BMR")
                elif surplus < 200:
                    st.warning("Very low intake")
                else:
                    st.success("Adequate intake")
                
                st.caption("Intake above BMR")
            else:
                st.metric("BMR Surplus", "N/A")
        
        # Activity & Performance Metrics
        st.subheader("🏃 Activity & Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pd.notna(latest_data.get('steps')):
                steps = latest_data['steps']
                st.metric("Daily Steps", f"{steps:,}")
                
                if steps >= 12000:
                    st.success("Very active")
                elif steps >= 8000:
                    st.info("Active")
                elif steps >= 5000:
                    st.warning("Moderately active")
                else:
                    st.error("Sedentary")
            else:
                st.metric("Daily Steps", "N/A")
        
        with col2:
            if pd.notna(latest_data.get('workout_duration_min_tot')):
                workout_min = latest_data['workout_duration_min_tot']
                st.metric("Workout Duration", f"{workout_min} min")
                
                if workout_min >= 60:
                    st.success("High volume")
                elif workout_min >= 30:
                    st.info("Moderate volume")
                elif workout_min > 0:
                    st.warning("Light activity")
                else:
                    st.error("Rest day")
            else:
                st.metric("Workout Duration", "N/A")
        
        with col3:
            if pd.notna(latest_data.get('steps_per_workout_min')):
                intensity = latest_data['steps_per_workout_min']
                st.metric("Activity Intensity", f"{intensity:.0f} steps/min")
                
                if intensity >= 100:
                    st.success("High intensity")
                elif intensity >= 50:
                    st.info("Moderate intensity")
                else:
                    st.warning("Low intensity")
                st.caption("During workout periods")
            else:
                st.metric("Activity Intensity", "N/A")
        
        # Metabolic Health Trends
        st.subheader("🔥 Metabolic Health Trends")
        
        # Filter last 30 days for trend analysis
        last_30_days = df_enhanced.tail(30)
        
        if len(last_30_days) > 7:
            # Create metabolic health visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Energy Balance Trend", 
                    "BMR vs Actual Intake",
                    "Sleep Efficiency", 
                    "Weight Trend"
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": True}]]
            )
            
            # Energy Balance
            if last_30_days['energy_balance'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=last_30_days['date'],
                        y=last_30_days['energy_balance'],
                        mode='lines+markers',
                        name='Energy Balance',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
            
            # BMR vs Intake
            if has_complete_profile and last_30_days['bmr'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=last_30_days['date'],
                        y=last_30_days['bmr'],
                        mode='lines',
                        name='BMR',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=2
                )
                
                if last_30_days['calories_consumed'].notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=last_30_days['date'],
                            y=last_30_days['calories_consumed'],
                            mode='lines+markers',
                            name='Intake',
                            line=dict(color='green')
                        ),
                        row=1, col=2
                    )
            
            # Sleep Efficiency
            if last_30_days['sleep_efficiency'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=last_30_days['date'],
                        y=last_30_days['sleep_efficiency'],
                        mode='lines+markers',
                        name='Sleep Efficiency',
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=75, line_dash="dash", line_color="orange", row=2, col=1)
            
            # Weight and BMI trends
            if last_30_days['weight'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=last_30_days['date'],
                        y=last_30_days['weight'],
                        mode='lines+markers',
                        name='Weight',
                        line=dict(color='black')
                    ),
                    row=2, col=2
                )
            
            if has_complete_profile and last_30_days['bmi'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=last_30_days['date'],
                        y=last_30_days['bmi'],
                        mode='lines',
                        name='BMI',
                        line=dict(color='orange'),
                        yaxis='y2'
                    ),
                    row=2, col=2, secondary_y=True
                )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="kcal", row=1, col=1)
            fig.update_yaxes(title_text="kcal/day", row=1, col=2)
            fig.update_yaxes(title_text="%", row=2, col=1)
            fig.update_yaxes(title_text="kg", row=2, col=2)
            fig.update_yaxes(title_text="BMI", row=2, col=2, secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recovery and Performance Analysis
        st.subheader("🛌 Recovery & Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep quality correlation with next-day performance
            if len(df_enhanced) > 1:
                # Create shifted data to compare sleep with next day activity
                df_shifted = df_enhanced.copy()
                df_shifted['next_day_steps'] = df_shifted['steps'].shift(-1)
                df_shifted['next_day_calories'] = df_shifted['calories_burned'].shift(-1)
                
                sleep_performance_corr = df_shifted[['sleep_hours', 'next_day_steps']].corr().iloc[0, 1]
                
                if pd.notna(sleep_performance_corr):
                    st.metric("Sleep-Performance Correlation", f"{sleep_performance_corr:.2f}")
                    
                    if sleep_performance_corr > 0.3:
                        st.success("Strong positive correlation - good sleep improves performance")
                    elif sleep_performance_corr > 0.1:
                        st.info("Moderate correlation between sleep and performance")
                    else:
                        st.warning("Weak correlation - other factors may dominate performance")
                else:
                    st.metric("Sleep-Performance Correlation", "N/A")
        
        with col2:
            # Weekly recovery score based on sleep consistency and energy balance
            if len(last_30_days) >= 7:
                sleep_consistency = 100 - (last_30_days['sleep_hours'].std() * 10)  # Lower std = higher score
                energy_stability = 100 - (abs(last_30_days['energy_balance'].std()) / 50)  # More stable = higher score
                
                recovery_score = (sleep_consistency + energy_stability) / 2
                recovery_score = max(0, min(100, recovery_score))  # Cap between 0-100
                
                st.metric("Recovery Score", f"{recovery_score:.0f}/100")
                
                if recovery_score >= 80:
                    st.success("Excellent recovery patterns")
                elif recovery_score >= 60:
                    st.info("Good recovery patterns")
                else:
                    st.warning("Recovery patterns need attention")
        
        # Recommendations Section
        st.subheader("💡 Personalized Recommendations")
        
        recommendations = []
        
        # Energy balance recommendations
        if pd.notna(latest_data.get('energy_balance')):
            balance = latest_data['energy_balance']
            if balance > 500:
                recommendations.append("Consider reducing caloric deficit to prevent metabolic slowdown. Aim for 300-500 kcal deficit for sustainable fat loss.")
            elif balance < -500:
                recommendations.append("Large caloric surplus detected. Consider reducing intake if weight gain is not your goal.")
        
        # Sleep recommendations
        if pd.notna(latest_data.get('sleep_hours')):
            sleep_hrs = latest_data['sleep_hours']
            if sleep_hrs < 7:
                recommendations.append("Aim for 7-9 hours of sleep per night for optimal recovery and metabolic health.")
            elif sleep_hrs > 9:
                recommendations.append("You're getting plenty of sleep. Ensure sleep quality is high rather than just quantity.")
        
        # BMI recommendations (only if no body fat data available)
        if has_complete_profile and not has_body_fat and pd.notna(latest_data.get('bmi')):
            bmi_val = latest_data['bmi']
            if bmi_val > 25:
                recommendations.append("Consider consulting a healthcare provider about healthy weight management strategies. Note: BMI may not be accurate for muscular individuals.")
            elif bmi_val < 18.5:
                recommendations.append("Consider consulting a healthcare provider about healthy weight gain strategies.")
        
        # Activity recommendations
        if pd.notna(latest_data.get('steps')):
            steps = latest_data['steps']
            if steps < 8000:
                recommendations.append("Try to increase daily activity. Aim for 8,000-10,000 steps per day for general health.")
        
        # FFMI-based recommendations
        if has_body_fat and pd.notna(latest_data.get('ffmi')):
            ffmi_val = latest_data['ffmi']
            if sex.lower() == 'male' and ffmi_val < 18:
                recommendations.append("Consider incorporating resistance training to build lean muscle mass. Your FFMI suggests room for muscle development.")
            elif sex.lower() == 'female' and ffmi_val < 15:
                recommendations.append("Consider incorporating resistance training to build lean muscle mass. Your FFMI suggests room for muscle development.")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("Your current metrics look well-balanced. Keep up the good work!")