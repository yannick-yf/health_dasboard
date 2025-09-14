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
    ["📊 Dashboard", "➕ Data Entry", "📈 Visualization", "🔬 Deep Dive", "⚙️ Settings"]
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
        # Date range and aggregation controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Start Date", 
                                       value=df['date'].min() if not df.empty else date.today())
        with col2:
            end_date = st.date_input("End Date", 
                                     value=df['date'].max() if not df.empty else date.today())
        with col3:
            aggregation = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"])
        
        # Filter data
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
                default=['steps', 'weight']
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
            
            # Correlation matrix
            st.subheader("Correlation Analysis")
            
            numeric_cols = ['steps', 'sleep_min', 'workout_duration_min_tot', 
                          'weight', 'calories_burned', 'calories_consumed']
            corr_matrix = filtered_df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                          labels=dict(color="Correlation"),
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          color_continuous_scale='RdBu_r',
                          zmin=-1, zmax=1,
                          title="Correlation Matrix")
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Missing data visualization
            st.subheader("Data Completeness")
            
            missing_data = filtered_df[numeric_cols].isnull().sum()
            total_records = len(filtered_df)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=missing_data.index,
                    y=(total_records - missing_data.values) / total_records * 100,
                    text=[f"{(total_records - v) / total_records * 100:.1f}%" for v in missing_data.values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Data Completeness by Metric",
                xaxis_title="Metric",
                yaxis_title="Completeness (%)",
                height=400,
                yaxis_range=[0, 105]
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Deep Dive Page
elif page == "🔬 Deep Dive":
    st.title("🔬 Deep Dive Analytics")
    
    st.info("⚠️ **Disclaimer**: This is informational only and not medical advice. Please consult healthcare professionals for medical guidance.")
    
    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
    else:
        # Check if user profile is complete
        profile = st.session_state.user_profile
        has_profile = all([profile['height_cm'], profile['age'], profile['sex']])
        
        if not has_profile:
            st.warning("Please complete your user profile in Settings to see all analytics.")
        
        # Latest metrics
        latest_record = df.sort_values('date').iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        # BMI Calculation
        with col1:
            st.subheader("📊 BMI")
            if profile['height_cm'] and pd.notna(latest_record['weight']):
                height_m = profile['height_cm'] / 100
                bmi = compute_bmi(latest_record['weight'], height_m)
                st.metric("Body Mass Index", f"{bmi:.1f}")
                
                if bmi < 18.5:
                    st.info("Category: Underweight")
                elif bmi < 25:
                    st.success("Category: Normal weight")
                elif bmi < 30:
                    st.warning("Category: Overweight")
                else:
                    st.error("Category: Obese")
                
                st.caption("BMI = weight(kg) / height(m)²")
            else:
                st.info("Insufficient data (need height and weight)")
        
        # BMR Calculation
        with col2:
            st.subheader("🔥 BMR")
            if has_profile and pd.notna(latest_record['weight']):
                bmr = estimate_bmr(
                    latest_record['weight'],
                    profile['height_cm'],
                    profile['age'],
                    profile['sex']
                )
                st.metric("Basal Metabolic Rate", f"{bmr:.0f} kcal/day")
                st.caption("Calories burned at rest")
            else:
                st.info("Complete profile to calculate")
        
        # TDEE Calculation
        with col3:
            st.subheader("⚡ TDEE")
            if has_profile and pd.notna(latest_record['weight']):
                bmr = estimate_bmr(
                    latest_record['weight'],
                    profile['height_cm'],
                    profile['age'],
                    profile['sex']
                )
                
                activity_factors = {
                    'sedentary': 1.2,
                    'light': 1.375,
                    'moderate': 1.55,
                    'active': 1.725
                }
                
                # Estimated TDEE
                est_tdee = compute_tdee(bmr, activity_factors.get(profile['activity_level'], 1.55))
                st.metric("Estimated TDEE", f"{est_tdee:.0f} kcal/day")
                st.caption(f"Based on profile activity level: {profile['activity_level']}")
                
                # Real TDEE from Apple Watch (average of last 30 days)
                if 'calories_burned' in df.columns:
                    # Ensure 'date' column is datetime
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    cutoff = df['date'].max() - pd.Timedelta(days=30)
                    
                    last_30d = df[(df['date'] > cutoff) & df['calories_burned'].notna()]
                    
                    if not last_30d.empty:
                        real_tdee = last_30d['calories_burned'].mean()
                        st.metric("Real TDEE (Apple Watch)", f"{real_tdee:.0f} kcal/day")
                        st.caption("30-day average from Apple Watch")
            else:
                st.info("Complete profile to calculate")
        
        # Energy Balance
        st.subheader("⚖️ Energy Balance Analysis")
        
        # Last 7 days analysis
        last_7_days = df.sort_values('date').tail(7)
        
        if not last_7_days.empty:
            avg_consumed = last_7_days['calories_consumed'].mean()
            avg_burned = last_7_days['calories_burned'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pd.notna(avg_consumed) and pd.notna(avg_burned):
                    daily_balance = compute_energy_balance(avg_consumed, avg_burned)
                    st.metric("Avg Daily Energy Balance (7 days)", f"{daily_balance:+.0f} kcal")
                    
                    if daily_balance > 500:
                        st.warning("Significant caloric surplus detected")
                    elif daily_balance < -500:
                        st.warning("Significant caloric deficit detected")
                    else:
                        st.success("Energy balance within normal range")
            
            with col2:
                if has_profile and pd.notna(avg_consumed) and pd.notna(latest_record['weight']):
                    bmr = estimate_bmr(
                        latest_record['weight'],
                        profile['height_cm'],
                        profile['age'],
                        profile['sex']
                    )
                    tdee = compute_tdee(bmr, activity_factors.get(profile['activity_level'], 1.55))
                    
                    deficit_surplus = avg_consumed - tdee
                    st.metric("Estimated Daily Deficit/Surplus", f"{deficit_surplus:+.0f} kcal")
                    
                    if deficit_surplus < -500:
                        weekly_loss = abs(deficit_surplus) * 7 / 7700  # 7700 kcal ≈ 1kg fat
                        st.info(f"Projected weight loss: ~{weekly_loss:.2f} kg/week")
                    elif deficit_surplus > 500:
                        weekly_gain = deficit_surplus * 7 / 7700
                        st.info(f"Projected weight gain: ~{weekly_gain:.2f} kg/week")
        
        # Sleep Consistency
        st.subheader("😴 Sleep Analysis")
        
        if df['sleep_min'].notna().any():
            sleep_consistency = compute_sleep_consistency(df['sleep_min'].dropna())
            avg_sleep = df['sleep_min'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Sleep Duration", f"{avg_sleep:.0f} min ({avg_sleep/60:.1f} hours)")
                if avg_sleep < 420:  # Less than 7 hours
                    st.warning("Below recommended 7-9 hours")
                elif avg_sleep > 540:  # More than 9 hours
                    st.warning("Above recommended 7-9 hours")
                else:
                    st.success("Within recommended range")
            
            with col2:
                st.metric("Sleep Consistency (SD)", f"{sleep_consistency:.0f} min")
                if sleep_consistency < 30:
                    st.success("Very consistent sleep schedule")
                elif sleep_consistency < 60:
                    st.info("Moderately consistent sleep")
                else:
                    st.warning("Inconsistent sleep schedule")
        
        # Activity Load
        st.subheader("🏃 Activity Analysis")
        
        weekly_loads = compute_activity_load(df)
        
        if not weekly_loads.empty:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Weekly Steps", "Weekly Workout Minutes")
            )
            
            fig.add_trace(
                go.Bar(x=weekly_loads.index, y=weekly_loads['steps'], name='Steps'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=weekly_loads.index, y=weekly_loads['workout_duration_min_tot'], name='Workout'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Check for significant changes
            if len(weekly_loads) >= 2:
                latest_week_steps = weekly_loads['steps'].iloc[-1]
                prev_week_steps = weekly_loads['steps'].iloc[-2]
                
                if pd.notna(latest_week_steps) and pd.notna(prev_week_steps) and prev_week_steps > 0:
                    change = (latest_week_steps - prev_week_steps) / prev_week_steps * 100
                    
                    if abs(change) > 20:
                        if change > 0:
                            st.info(f"📈 Activity increased by {change:.0f}% this week")
                        else:
                            st.warning(f"📉 Activity decreased by {abs(change):.0f}% this week")
        
        # Anomaly Detection
        st.subheader("🔍 Anomaly Detection")
        
        anomaly_metrics = ['steps', 'calories_consumed', 'sleep_min']
        anomalies_found = False
        
        for metric in anomaly_metrics:
            if df[metric].notna().sum() > 10:  # Need sufficient data
                anomalies = detect_anomalies(df[metric])
                anomaly_dates = df[anomalies]['date'].dt.date.tolist()
                
                if anomaly_dates:
                    anomalies_found = True
                    st.warning(f"**{metric.replace('_', ' ').title()}** anomalies detected on: {', '.join(map(str, anomaly_dates))}")
        
        if not anomalies_found:
            st.success("No significant anomalies detected in recent data")
        
        # Trends
        st.subheader("📈 Trend Analysis")
        
        trend_metrics = ['weight', 'steps', 'calories_consumed']
        
        for metric in trend_metrics:
            if df[metric].notna().sum() > 7:  # Need at least a week of data
                slope, p_value = compute_trend_with_significance(df['date'], df[metric])
                
                if pd.notna(slope):
                    trend_direction = "📈 Increasing" if slope > 0 else "📉 Decreasing"
                    significance = "significant" if p_value < 0.05 else "not significant"
                    
                    st.write(f"**{metric.replace('_', ' ').title()}**: {trend_direction} "
                           f"(slope: {slope:.2f}/day, {significance})")

# Settings Page
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    # User Profile
    st.subheader("👤 User Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        height = st.number_input("Height (cm)", 
                                min_value=100, 
                                max_value=250,
                                value=st.session_state.user_profile['height_cm'] or 170)
        
        age = st.number_input("Age", 
                             min_value=1, 
                             max_value=120,
                             value=st.session_state.user_profile['age'] or 30)
    
    with col2:
        sex = st.selectbox("Sex", 
                          ["Male", "Female"],
                          index=0 if st.session_state.user_profile['sex'] == "Male" else 1 if st.session_state.user_profile['sex'] == "Female" else 0)
        
        activity_level = st.selectbox("Activity Level",
                                     ["sedentary", "light", "moderate", "active"],
                                     index=["sedentary", "light", "moderate", "active"].index(
                                         st.session_state.user_profile['activity_level']))
    
    if st.button("Save Profile"):
        st.session_state.user_profile = {
            'height_cm': height,
            'age': age,
            'sex': sex,
            'activity_level': activity_level
        }
        st.success("Profile saved successfully!")
    
    # Data Settings
    st.subheader("📁 Data Settings")
    
    csv_path = st.text_input("CSV File Path", value=st.session_state.csv_path)
    
    if csv_path != st.session_state.csv_path:
        st.session_state.csv_path = csv_path
        st.rerun()
    
    # Imputation Settings
    st.subheader("🔧 Missing Data Handling")
    
    imputation_method = st.selectbox(
        "Imputation Method",
        ["none", "forward_fill", "median"],
        index=["none", "forward_fill", "median"].index(st.session_state.imputation_method)
    )
    
    if imputation_method != st.session_state.imputation_method:
        st.session_state.imputation_method = imputation_method
        st.info(f"Imputation method changed to: {imputation_method}")
        st.rerun()
    
    # Backup Settings
    st.subheader("💾 Backup Settings")
    
    backup_dir = Path("backups")
    if backup_dir.exists():
        backups = list(backup_dir.glob("*.csv"))
        if backups:
            st.write(f"Found {len(backups)} backup(s)")
            
            # Show recent backups
            recent_backups = sorted(backups, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            for backup in recent_backups:
                st.text(f"  • {backup.name}")
            
            if st.button("Clear All Backups", type="secondary"):
                for backup in backups:
                    backup.unlink()
                st.success("All backups cleared")
                st.rerun()
        else:
            st.info("No backups found")
    else:
        st.info("Backup directory will be created on first backup")
    
    # About
    st.subheader("ℹ️ About")
    st.write("""
    **Health Tracker v1.0**
    
    A comprehensive health metrics tracking and analysis application.
    
    **Features:**
    - Daily health data entry and management
    - Interactive visualizations with trends
    - Research-backed health indicators (BMI, BMR, TDEE)
    - Anomaly detection and trend analysis
    - Data import/export capabilities
    
    **Formulas Used:**
    - BMI: weight(kg) / height(m)²
    - BMR (Mifflin-St Jeor): 
        - Men: 10×weight + 6.25×height - 5×age + 5
        - Women: 10×weight + 6.25×height - 5×age - 161
    - TDEE: BMR × activity_factor
    
    **Note:** This application is for informational purposes only and should not be used as a substitute for professional medical advice.
    """)