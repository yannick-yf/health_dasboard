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
    ["📊 Analytics Dashboard", "➕ Data Entry", "🔬 Deep Dive"]
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

# Consolidated Dashboard & Visualization Page
if page == "📊 Analytics Dashboard":
    st.title("📊 Health Analytics Dashboard")
    
    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
    else:
        # ==================== TIME PERIOD SELECTION ====================
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
        
        # Aggregation control
        col_agg = st.columns([1, 2])[0]
        with col_agg:
            aggregation = st.selectbox("📊 Data Aggregation", ["Daily", "Weekly", "Monthly"])
        
        # Filter data by date range
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            st.warning("No data in selected date range.")
        else:
            # Store original daily data for KPIs and trends
            daily_df = filtered_df.copy()
            
            # Apply aggregation for visualizations
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
            
            # ==================== HELPER FUNCTIONS ====================
            def calculate_weight_trend(df, days):
                """Calculate weight trend percentage over the last N days"""
                if len(df) < 2 or df['weight'].isna().all():
                    return None
                
                weight_data = df[df['weight'].notna()].sort_values('date')
                
                if len(weight_data) < 2:
                    return None
                
                if len(weight_data) >= days:
                    recent_data = weight_data.tail(days)
                else:
                    recent_data = weight_data
                
                if len(recent_data) < 2:
                    return None
                
                first_weight = recent_data['weight'].iloc[0]
                last_weight = recent_data['weight'].iloc[-1]
                
                if first_weight == 0:
                    return None
                
                trend_percent = ((last_weight - first_weight) / first_weight) * 100
                return trend_percent
            
            def format_sleep_time(minutes):
                """Convert minutes to hours and minutes format"""
                if pd.isna(minutes):
                    return "N/A"
                hours = int(minutes // 60)
                mins = int(minutes % 60)
                return f"{hours}h {mins}m"
            
            # ==================== WEIGHT TRENDS SECTION ====================
            st.subheader("⚖️ Weight Trends")
            trend_cols = st.columns(3)
            
            trend_periods = [3, 5, 7]
            trend_labels = ["3-Day Trend", "5-Day Trend", "7-Day Trend"]
            
            for col, days, label in zip(trend_cols, trend_periods, trend_labels):
                trend = calculate_weight_trend(daily_df, days)
                if trend is not None:
                    if trend > 0:
                        trend_text = f"+{trend:.2f}%"
                    else:
                        trend_text = f"{trend:.2f}%"
                    
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
            
            # ==================== KEY PERFORMANCE INDICATORS ====================
            st.subheader("📋 Key Performance Indicators")
            kpi_cols = st.columns(4)
            
            # Calculate KPIs from daily data
            avg_sleep_min = daily_df['sleep_min'].mean()
            avg_sleep_formatted = format_sleep_time(avg_sleep_min)
            
            metrics = [
                ("🚶 Avg Steps", daily_df['steps'].mean(), "{:.0f}"),
                ("😴 Avg Sleep", avg_sleep_formatted, "{}"),
                ("🔥 Avg Burned", daily_df['calories_burned'].mean(), "{:.0f} kcal"),
                ("🍽️ Avg Consumed", daily_df['calories_consumed'].mean(), "{:.0f} kcal")
            ]
            
            for col, (label, value, fmt) in zip(kpi_cols, metrics):
                if isinstance(value, str):  # For formatted sleep time
                    col.metric(label, value)
                elif pd.notna(value):
                    col.metric(label, fmt.format(value))
                else:
                    col.metric(label, "N/A")
            
            # ==================== TIME SERIES ANALYSIS ====================
            st.subheader("📈 Time Series Analysis")
            
            # Metric selection
            metrics_to_plot = st.multiselect(
                "Select metrics to visualize",
                ['steps', 'sleep_min', 'workout_duration_min_tot', 'weight', 'calories_burned', 'calories_consumed'],
                default=['weight', 'calories_burned', 'calories_consumed'],
                help="Choose which health metrics you want to analyze"
            )
            
            # Create enhanced time series plots
            for metric in metrics_to_plot:
                if filtered_df[metric].notna().any():
                    fig = go.Figure()
                    
                    # Determine appropriate units and colors
                    metric_config = {
                        'steps': {'unit': 'steps', 'color': '#1f77b4'},
                        'sleep_min': {'unit': 'minutes', 'color': '#9467bd'},
                        'workout_duration_min_tot': {'unit': 'minutes', 'color': '#ff7f0e'},
                        'weight': {'unit': 'kg', 'color': '#d62728'},
                        'calories_burned': {'unit': 'kcal', 'color': '#ff6b6b'},
                        'calories_consumed': {'unit': 'kcal', 'color': '#4ecdc4'}
                    }
                    
                    config = metric_config.get(metric, {'unit': '', 'color': '#2ca02c'})
                    
                    # Main data line
                    fig.add_trace(go.Scatter(
                        x=filtered_df['date'],
                        y=filtered_df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=3, color=config['color']),
                        marker=dict(size=6),
                        hovertemplate='<b>%{y:.1f}</b> ' + config['unit'] + '<br>%{x}<extra></extra>'
                    ))
                    
                    # Add moving averages for daily data
                    if aggregation == "Daily" and len(filtered_df) > 7:
                        # 7-day moving average
                        ma7 = filtered_df[metric].rolling(window=7, min_periods=1).mean()
                        fig.add_trace(go.Scatter(
                            x=filtered_df['date'],
                            y=ma7,
                            mode='lines',
                            name='7-day Average',
                            line=dict(dash='dash', width=2, color=config['color']),
                            opacity=0.7,
                            hovertemplate='7-day avg: <b>%{y:.1f}</b> ' + config['unit'] + '<extra></extra>'
                        ))
                        
                        # 30-day moving average for longer periods
                        if len(filtered_df) > 30:
                            ma30 = filtered_df[metric].rolling(window=30, min_periods=1).mean()
                            fig.add_trace(go.Scatter(
                                x=filtered_df['date'],
                                y=ma30,
                                mode='lines',
                                name='30-day Average',
                                line=dict(dash='dot', width=2, color=config['color']),
                                opacity=0.5,
                                hovertemplate='30-day avg: <b>%{y:.1f}</b> ' + config['unit'] + '<extra></extra>'
                            ))
                    
                    # Add trend line if we have enough data points
                    if len(filtered_df[filtered_df[metric].notna()]) >= 3:
                        trend_data = filtered_df[filtered_df[metric].notna()].reset_index(drop=True)
                        if len(trend_data) > 1:
                            x_numeric = np.arange(len(trend_data))
                            z = np.polyfit(x_numeric, trend_data[metric], 1)
                            p = np.poly1d(z)
                            
                            # Determine trend direction for color
                            trend_color = '#28a745' if z[0] > 0 else '#dc3545' if z[0] < 0 else '#6c757d'
                            
                            fig.add_trace(go.Scatter(
                                x=trend_data['date'],
                                y=p(x_numeric),
                                mode='lines',
                                name='Trend',
                                line=dict(dash='dashdot', width=2, color=trend_color),
                                opacity=0.8,
                                hovertemplate='Trend: <b>%{y:.1f}</b> ' + config['unit'] + '<extra></extra>'
                            ))
                    
                    # Enhanced layout
                    fig.update_layout(
                        title={
                            'text': f"{metric.replace('_', ' ').title()} Over Time ({aggregation} View)",
                            'x': 0.5,
                            'font': {'size': 18}
                        },
                        xaxis_title="Date",
                        yaxis_title=f"{metric.replace('_', ' ').title()} ({config['unit']})",
                        height=450,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    # Add grid
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # ==================== SPECIAL COMBINED VISUALIZATIONS ====================
            
            # Enhanced Calorie Balance Chart (always show if data available)
            if (daily_df['calories_burned'].notna().any() and daily_df['calories_consumed'].notna().any()):
                st.subheader("🔥 Calorie Balance Analysis")
                
                fig = go.Figure()
                
                # Use aggregated data for the chart
                calories_df = filtered_df[['date', 'calories_burned', 'calories_consumed']].dropna()
                
                if not calories_df.empty:
                    # Calculate balance
                    calories_df['balance'] = calories_df['calories_burned'] - calories_df['calories_consumed']
                    
                    # Add calories burned
                    fig.add_trace(go.Scatter(
                        x=calories_df['date'],
                        y=calories_df['calories_burned'],
                        mode='lines+markers',
                        name='🔥 Burned',
                        line=dict(color='#FF6B6B', width=3),
                        marker=dict(size=8),
                        fill='tonexty',
                        hovertemplate='Burned: <b>%{y:.0f}</b> kcal<extra></extra>'
                    ))
                    
                    # Add calories consumed
                    fig.add_trace(go.Scatter(
                        x=calories_df['date'],
                        y=calories_df['calories_consumed'],
                        mode='lines+markers',
                        name='🍽️ Consumed',
                        line=dict(color='#4ECDC4', width=3),
                        marker=dict(size=8),
                        hovertemplate='Consumed: <b>%{y:.0f}</b> kcal<extra></extra>'
                    ))
                    
                    # Add balance as bar chart on secondary y-axis
                    fig.add_trace(go.Bar(
                        x=calories_df['date'],
                        y=calories_df['balance'],
                        name='Balance',
                        marker_color=['#28a745' if x < 0 else '#dc3545' for x in calories_df['balance']],
                        opacity=0.3,
                        yaxis='y2',
                        hovertemplate='Balance: <b>%{y:+.0f}</b> kcal<extra></extra>'
                    ))
                    
                    # Calculate average balance for insight
                    avg_balance = calories_df['balance'].mean()
                    
                    fig.update_layout(
                        title='Calorie Balance Analysis',
                        xaxis_title='Date',
                        yaxis=dict(title='Calories (kcal)', side='left'),
                        yaxis2=dict(title='Balance (kcal)', side='right', overlaying='y'),
                        height=400,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Balance insight
                    col_insight = st.columns(1)[0]
                    with col_insight:
                        if avg_balance > 200:
                            st.success(f"📉 Average daily deficit: {abs(avg_balance):.0f} kcal - Good for weight loss")
                        elif avg_balance < -200:
                            st.warning(f"📈 Average daily surplus: {abs(avg_balance):.0f} kcal - Good for weight gain")
                        else:
                            st.info(f"⚖️ Balanced caloric intake (±{abs(avg_balance):.0f} kcal)")
            
            # Sleep Consistency Analysis
            if daily_df['sleep_min'].notna().sum() >= 7:
                st.subheader("😴 Sleep Consistency Analysis")
                
                sleep_data = daily_df[daily_df['sleep_min'].notna()].copy()
                sleep_data['sleep_hours'] = sleep_data['sleep_min'] / 60
                
                sleep_std = sleep_data['sleep_hours'].std()
                sleep_mean = sleep_data['sleep_hours'].mean()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if sleep_std < 0.5:  # Less than 30 minutes variation
                        st.success(f"🎯 Excellent sleep consistency! (±{sleep_std*60:.0f} minutes)")
                    elif sleep_std < 1:  # Less than 1 hour variation
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
        
# Deep Dive Page - Complete with Weekly Analysis
elif page == "🔬 Deep Dive":
    st.title("🔬 Deep Dive Analytics")
    
    # Metrics Glossary Section
    with st.expander("📖 Metrics Glossary - Click to expand definitions"):
        st.markdown("### 📊 Weekly Health Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **BMR (Basal Metabolic Rate)**  
            Calories your body burns at rest for basic functions like breathing and circulation. Calculated using the Mifflin-St Jeor equation based on weight, height, age, and sex.
            
            **Energy Balance**  
            Difference between calories burned and consumed (Burned - Consumed). Positive = deficit, Negative = surplus.
            
            **FFMI (Fat-Free Mass Index)**  
            Muscle mass relative to height, calculated as lean body mass ÷ height². Superior to BMI for athletes as it separates muscle from fat.
            
            **FMI (Fat Mass Index)**  
            Fat mass relative to height. Provides direct assessment of body fat distribution independent of total weight.
            
            **Normalized FFMI**  
            Height-adjusted FFMI that accounts for the advantage taller individuals have in building muscle mass.
            """)
        
        with col2:
            st.markdown("""
            **Metabolic Multiple**  
            Total daily calories burned ÷ BMR. Shows how active your lifestyle is (1.2 = sedentary, 1.9+ = very active).
            
            **NEAT (Non-Exercise Activity Thermogenesis)**  
            Calories burned through daily activities excluding sleeping, eating, and structured exercise. Includes fidgeting, maintaining posture, and spontaneous movement.
            
            **BMR Surplus**  
            Calories consumed above your BMR. Negative values indicate eating below metabolic baseline, which can slow metabolism.
            
            **Sleep Efficiency**  
            Quality score based on sleep duration (7-8 hours = optimal). Accounts for both adequacy and consistency of sleep patterns.
            
            **Calorie Efficiency**  
            Calories burned per step taken. Varies based on body weight, walking speed, and terrain.
            """)
        
        st.markdown("### 🏃 Activity & Performance Metrics")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            **Activity Intensity**  
            Steps per minute during workout periods. Indicates exercise intensity and movement efficiency during training sessions.
            
            **Recovery Score**  
            Composite score based on sleep consistency and energy balance stability. Higher scores indicate better recovery patterns.
            """)
        
        with col4:
            st.markdown("""
            **Sleep-Performance Correlation**  
            Statistical relationship between sleep quality and next-day activity levels. Values above 0.3 indicate strong positive correlation.
            
            **Weight Change**  
            Weekly weight difference from first to last measurement. Accounts for natural daily fluctuations by using week-over-week comparison.
            """)
        
        st.markdown("### ⚠️ Important Notes")
        st.markdown("""
        - **BMI Limitations**: BMI doesn't distinguish muscle from fat. For athletes, FFMI and body fat percentage are more meaningful.
        - **Data Requirements**: Advanced metrics require personal profile data (height, age, sex, body fat %).
        - **Lag Calculations**: BMR uses previous day's weight since today's energy burn reflects yesterday's body composition.
        - **NEAT Estimation**: Only structured gym sessions (60-75 min) are subtracted as exercise; cycling commutes and walking remain in NEAT.
        - **Medical Disclaimer**: This is informational only. Consult healthcare professionals for medical guidance.
        """)
    
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
            birth_date = datetime.strptime(personal_info['birth_date'], '%d-%m-%Y').date()
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            height_cm = personal_info['height_cm']
            height_m = height_cm / 100
            sex = personal_info['sex']
            body_fat_prct = personal_info.get('body_fat_prct', None)
            
            # Calculate daily BMI
            df_enhanced['bmi'] = df_enhanced['weight'] / (height_m ** 2)
            
            # Calculate Fat-Free Mass Index (FFMI) if body fat is provided
            if body_fat_prct is not None:
                df_enhanced['fat_mass'] = df_enhanced['weight'] * (body_fat_prct / 100)
                df_enhanced['fat_free_mass'] = df_enhanced['weight'] * (1 - body_fat_prct/100)
                df_enhanced['ffmi'] = df_enhanced['fat_free_mass'] / (height_m ** 2)
                df_enhanced['fmi'] = df_enhanced['fat_mass'] / (height_m ** 2)
                df_enhanced['normalized_ffmi'] = df_enhanced['ffmi'] + 6.1 * (1.8 - height_m)
            
            # For metabolic calculations, use previous day's weight (lag by 1)
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
            
            # Calculate daily TDEE and energy balance
            df_enhanced['actual_tdee'] = df_enhanced['calories_burned']
            df_enhanced['energy_balance'] = df_enhanced['calories_burned'] - df_enhanced['calories_consumed']
            df_enhanced['bmr_surplus'] = df_enhanced['calories_consumed'] - df_enhanced['bmr']
            df_enhanced['bmr_ratio'] = df_enhanced['calories_burned'] / df_enhanced['bmr']
            
            # Enhanced NEAT calculation based on steps
            # Research shows approximately 0.04-0.05 kcal per step for average adults
            # This varies by weight, so we'll use 0.04 * weight_kg as baseline
            kcal_per_step = 0.04 * df_enhanced['weight']
            estimated_steps_calories = df_enhanced['steps'] * (kcal_per_step / 1000)  # Convert to kcal
            
            # Calculate structured exercise calories
            structured_exercise_minutes = np.minimum(df_enhanced['workout_duration_min_tot'], 75)
            estimated_exercise_calories = structured_exercise_minutes * 10  # 10 kcal/min for gym sessions
            
            # Calculate TEF (Thermic Effect of Food)
            estimated_tef = df_enhanced['calories_consumed'] * 0.1
            
            # NEAT = Total Calories - BMR - TEF - Exercise - Steps Calories
            df_enhanced['neat_estimate'] = (df_enhanced['calories_burned'] - df_enhanced['bmr'] - 
                                          estimated_tef - estimated_exercise_calories - estimated_steps_calories)
            
            # Ensure NEAT is not negative (minimum 50 kcal)
            df_enhanced['neat_estimate'] = np.maximum(df_enhanced['neat_estimate'], 50)
            
            # Calculate step efficiency (calories per 1000 steps)
            df_enhanced['calories_per_1k_steps'] = (df_enhanced['calories_burned'] / df_enhanced['steps']) * 1000
        
        # Calculate sleep and activity metrics
        df_enhanced['sleep_hours'] = df_enhanced['sleep_min'] / 60
        df_enhanced['sleep_efficiency'] = np.where(
            df_enhanced['sleep_min'] >= 420,  # 7+ hours
            np.minimum(100, (df_enhanced['sleep_min'] / 480) * 100),  # Cap at 100% for 8+ hours
            (df_enhanced['sleep_min'] / 420) * 100  # Scale below 7 hours
        )
        
        df_enhanced['steps_per_workout_min'] = np.where(
            df_enhanced['workout_duration_min_tot'] > 0,
            df_enhanced['steps'] / df_enhanced['workout_duration_min_tot'],
            np.nan
        )
        
        # Week Selection Interface
        st.markdown("---")
        
        # Calculate available weeks
        df_enhanced['week_start'] = df_enhanced['date'].dt.to_period('W').dt.start_time
        available_weeks = sorted(df_enhanced['week_start'].unique(), reverse=True)
        
        if len(available_weeks) > 0:
            # Create week selection in a centered column
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Format week options for display
                week_options = []
                for week in available_weeks:
                    week_end = week + pd.Timedelta(days=6)
                    if week == available_weeks[0]:
                        week_options.append(f"Current Week ({week.strftime('%b %d')} - {week_end.strftime('%b %d')})")
                    else:
                        week_options.append(f"{week.strftime('%b %d')} - {week_end.strftime('%b %d')}")
                
                selected_week_idx = st.selectbox("📅 Select Analysis Period", range(len(week_options)), 
                                               format_func=lambda x: week_options[x])
                selected_week_start = available_weeks[selected_week_idx]
            
            # Filter data for selected week
            week_end = selected_week_start + pd.Timedelta(days=6)
            week_mask = (df_enhanced['date'] >= selected_week_start) & (df_enhanced['date'] <= week_end)
            week_data = df_enhanced[week_mask].copy()
            
            # Calculate weekly statistics
            complete_week_data = week_data.dropna(subset=['steps', 'calories_burned', 'calories_consumed'])
            days_with_data = len(complete_week_data)
            
            # Header with period info
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
                # === WEEKLY OVERVIEW METRICS ===
                st.subheader("📈 Weekly Overview")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if has_complete_profile and complete_week_data['bmr'].notna().any():
                        avg_bmr = complete_week_data['bmr'].mean()
                        st.metric("Avg BMR", f"{avg_bmr:.0f}", help="Basal Metabolic Rate")
                    else:
                        st.metric("Avg BMR", "N/A")
                
                with col2:
                    if complete_week_data['energy_balance'].notna().any():
                        avg_balance = complete_week_data['energy_balance'].mean()
                        delta = "Deficit" if avg_balance > 0 else "Surplus" if avg_balance < 0 else "Balanced"
                        st.metric("Energy Balance", f"{avg_balance:+.0f}", delta=delta)
                    else:
                        st.metric("Energy Balance", "N/A")
                
                with col3:
                    if complete_week_data['sleep_efficiency'].notna().any():
                        avg_sleep_eff = complete_week_data['sleep_efficiency'].mean()
                        st.metric("Sleep Quality", f"{avg_sleep_eff:.0f}%", 
                                help="Sleep efficiency score")
                    else:
                        st.metric("Sleep Quality", "N/A")
                
                with col4:
                    if complete_week_data['steps'].notna().any():
                        avg_steps = complete_week_data['steps'].mean()
                        st.metric("Daily Steps", f"{avg_steps:,.0f}", 
                                help="Average daily step count")
                    else:
                        st.metric("Daily Steps", "N/A")
                
                with col5:
                    if complete_week_data['weight'].notna().any():
                        week_weights = complete_week_data['weight'].dropna()
                        if len(week_weights) >= 2:
                            weight_change = week_weights.iloc[-1] - week_weights.iloc[0]
                            delta_color = "inverse" if weight_change > 0 else "normal"
                            st.metric("Weight Change", f"{weight_change:+.1f} kg", 
                                    delta=f"{weight_change:+.1f}", delta_color=delta_color)
                        else:
                            avg_weight = week_weights.mean()
                            st.metric("Avg Weight", f"{avg_weight:.1f} kg")
                    else:
                        st.metric("Weight", "N/A")
                
                
                # === BODY COMPOSITION ANALYSIS ===
                st.markdown("---")
                st.subheader("💪 Weekly Body Composition Analysis")
                
                if has_complete_profile and has_body_fat:
                    # Create body composition visualization
                    fig_composition = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("Fat-Free Mass Index", "Fat Mass Index", "Normalized FFMI"),
                        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
                    )
                    
                    if complete_week_data['ffmi'].notna().any():
                        avg_ffmi = complete_week_data['ffmi'].mean()
                        
                        # FFMI ranges for interpretation
                        if sex.lower() == 'male':
                            ffmi_max, ffmi_good = 25, 20
                        else:
                            ffmi_max, ffmi_good = 21, 17
                        
                        fig_composition.add_trace(
                            go.Indicator(
                                mode="gauge+number+delta",
                                value=avg_ffmi,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "FFMI"},
                                gauge={
                                    'axis': {'range': [None, ffmi_max]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, ffmi_good*0.8], 'color': "lightgray"},
                                        {'range': [ffmi_good*0.8, ffmi_good], 'color': "yellow"},
                                        {'range': [ffmi_good, ffmi_max*0.9], 'color': "green"},
                                        {'range': [ffmi_max*0.9, ffmi_max], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': ffmi_max*0.9
                                    }
                                }
                            ),
                            row=1, col=1
                        )
                    
                    if complete_week_data['fmi'].notna().any():
                        avg_fmi = complete_week_data['fmi'].mean()
                        
                        if sex.lower() == 'male':
                            fmi_max, fmi_good = 12, 6
                        else:
                            fmi_max, fmi_good = 15, 9
                        
                        fig_composition.add_trace(
                            go.Indicator(
                                mode="gauge+number",
                                value=avg_fmi,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "FMI"},
                                gauge={
                                    'axis': {'range': [0, fmi_max]},
                                    'bar': {'color': "orange"},
                                    'steps': [
                                        {'range': [0, fmi_good*0.8], 'color': "green"},
                                        {'range': [fmi_good*0.8, fmi_good], 'color': "yellow"},
                                        {'range': [fmi_good, fmi_max], 'color': "red"}
                                    ]
                                }
                            ),
                            row=1, col=2
                        )
                    
                    if complete_week_data['normalized_ffmi'].notna().any():
                        avg_norm_ffmi = complete_week_data['normalized_ffmi'].mean()
                        
                        fig_composition.add_trace(
                            go.Indicator(
                                mode="gauge+number",
                                value=avg_norm_ffmi,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Normalized FFMI"},
                                gauge={
                                    'axis': {'range': [0, 30]},
                                    'bar': {'color': "purple"},
                                    'steps': [
                                        {'range': [0, 18], 'color': "lightgray"},
                                        {'range': [18, 22], 'color': "yellow"},
                                        {'range': [22, 25], 'color': "green"},
                                        {'range': [25, 30], 'color': "red"}
                                    ]
                                }
                            ),
                            row=1, col=3
                        )
                    
                    fig_composition.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_composition, use_container_width=True)
                    
                    # Additional metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if complete_week_data['fat_free_mass'].notna().any():
                            avg_ffm = complete_week_data['fat_free_mass'].mean()
                            st.metric("Fat-Free Mass", f"{avg_ffm:.1f} kg", 
                                    help=f"Body fat: {body_fat_prct}%")
                    
                    with col2:
                        if complete_week_data['fat_mass'].notna().any():
                            avg_fm = complete_week_data['fat_mass'].mean()
                            st.metric("Fat Mass", f"{avg_fm:.1f} kg")
                    
                    with col3:
                        muscle_mass_trend = "📈 Building" if avg_ffmi > 20 else "📊 Maintaining" if avg_ffmi > 17 else "📉 Low"
                        st.metric("Muscle Status", muscle_mass_trend)
                
                else:
                    st.info("💡 Add 'body_fat_prct' to your personal_info.json to unlock advanced body composition metrics")
                    
                    if has_complete_profile and complete_week_data['bmi'].notna().any():
                        avg_bmi = complete_week_data['bmi'].mean()
                        
                        # BMI Gauge
                        fig_bmi = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=avg_bmi,
                            title={'text': "BMI (Limited Metric)"},
                            gauge={
                                'axis': {'range': [None, 35]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 18.5], 'color': "lightgray"},
                                    {'range': [18.5, 25], 'color': "green"},
                                    {'range': [25, 30], 'color': "yellow"},
                                    {'range': [30, 35], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 30
                                }
                            }
                        ))
                        fig_bmi.update_layout(height=300)
                        st.plotly_chart(fig_bmi, use_container_width=True)
                        
                        st.warning("⚠️ BMI doesn't distinguish between muscle and fat. For athletes, body composition metrics (FFMI) are more meaningful.")
                
                # === METABOLIC ANALYSIS ===
                st.markdown("---")
                st.subheader("🔥 Weekly Metabolic Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if has_complete_profile and complete_week_data['bmr_ratio'].notna().any():
                        avg_ratio = complete_week_data['bmr_ratio'].mean()
                        st.metric("Metabolic Multiple", f"{avg_ratio:.2f}x", 
                                help="Total daily energy expenditure / BMR")
                        
                        if avg_ratio < 1.3:
                            st.error("🟥 Low Activity")
                        elif avg_ratio < 1.6:
                            st.warning("🟨 Moderate Activity")
                        elif avg_ratio < 2.0:
                            st.success("🟩 Active")
                        else:
                            st.success("🟢 Very Active")
                
                with col2:
                    if has_complete_profile and complete_week_data['neat_estimate'].notna().any():
                        avg_neat = complete_week_data['neat_estimate'].mean()
                        st.metric("NEAT", f"{avg_neat:.0f} kcal", 
                                help="Non-Exercise Activity Thermogenesis")
                        
                        if avg_neat < 200:
                            st.error("🟥 Low NEAT")
                        elif avg_neat < 400:
                            st.warning("🟨 Moderate NEAT")
                        else:
                            st.success("🟩 High NEAT")
                
                with col3:
                    if complete_week_data['calories_per_1k_steps'].notna().any():
                        avg_cal_per_1k = complete_week_data['calories_per_1k_steps'].mean()
                        st.metric("Step Efficiency", f"{avg_cal_per_1k:.0f} kcal/1k", 
                                help="Calories burned per 1000 steps")
                
                with col4:
                    if has_complete_profile and complete_week_data['bmr_surplus'].notna().any():
                        avg_surplus = complete_week_data['bmr_surplus'].mean()
                        st.metric("BMR Surplus", f"{avg_surplus:+.0f} kcal", 
                                help="Daily intake above BMR")
                        
                        if avg_surplus < 0:
                            st.error("🚨 Below BMR")
                        elif avg_surplus < 200:
                            st.warning("⚠️ Very Low")
                        else:
                            st.success("✅ Adequate")
                
                # === ACTIVITY & PERFORMANCE ===
                st.markdown("---")
                st.subheader("🏃 Weekly Activity & Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if complete_week_data['steps'].notna().any():
                        avg_steps = complete_week_data['steps'].mean()
                        total_steps = complete_week_data['steps'].sum()
                        st.metric("Daily Steps", f"{avg_steps:,.0f}", 
                                help=f"Weekly total: {total_steps:,}")
                        
                        if avg_steps >= 12000:
                            st.success("🟢 Very Active")
                        elif avg_steps >= 8000:
                            st.info("🔵 Active")
                        elif avg_steps >= 5000:
                            st.warning("🟡 Moderate")
                        else:
                            st.error("🔴 Sedentary")
                
                with col2:
                    if complete_week_data['workout_duration_min_tot'].notna().any():
                        avg_workout = complete_week_data['workout_duration_min_tot'].mean()
                        total_workout = complete_week_data['workout_duration_min_tot'].sum()
                        st.metric("Workout Duration", f"{avg_workout:.0f} min", 
                                help=f"Weekly total: {total_workout:.0f} min")
                        
                        if avg_workout >= 60:
                            st.success("🟢 High Volume")
                        elif avg_workout >= 30:
                            st.info("🔵 Moderate")
                        elif avg_workout > 0:
                            st.warning("🟡 Light")
                        else:
                            st.error("🔴 Rest Week")
                
                with col3:
                    if complete_week_data['steps_per_workout_min'].notna().any():
                        avg_intensity = complete_week_data['steps_per_workout_min'].mean()
                        st.metric("Workout Intensity", f"{avg_intensity:.0f} steps/min", 
                                help="Activity intensity during workouts")
                        
                        if avg_intensity >= 100:
                            st.success("🟢 High Intensity")
                        elif avg_intensity >= 50:
                            st.info("🔵 Moderate")
                        else:
                            st.warning("🟡 Low Intensity")
        
        # === DAILY TRENDS VISUALIZATION ===
        st.markdown("---")
        st.subheader("📈 Daily Health Trends (Last 30 Days)")
        
        # Filter last 30 days for trend analysis
        last_30_days = df_enhanced.tail(30)
        
        if len(last_30_days) > 7:
            # Tab system for different trend views
            tab1, tab2, tab3, tab4 = st.tabs(["🔥 Metabolic Trends", "💪 Body Composition", "🏃 Activity Patterns", "😴 Recovery Metrics"])
            
            with tab1:
                # Metabolic trends
                fig_metabolic = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Energy Balance", "BMR vs Intake", "Metabolic Multiple", "NEAT Estimation"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Energy Balance
                if last_30_days['energy_balance'].notna().any():
                    fig_metabolic.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['energy_balance'],
                                 mode='lines+markers', name='Energy Balance',
                                 line=dict(color='#3498db', width=2), marker=dict(size=4)),
                        row=1, col=1
                    )
                    fig_metabolic.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                
                # BMR vs Intake
                if has_complete_profile and last_30_days['bmr'].notna().any():
                    fig_metabolic.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['bmr'],
                                 mode='lines', name='BMR', line=dict(color='#e74c3c', dash='dash')),
                        row=1, col=2
                    )
                    if last_30_days['calories_consumed'].notna().any():
                        fig_metabolic.add_trace(
                            go.Scatter(x=last_30_days['date'], y=last_30_days['calories_consumed'],
                                     mode='lines+markers', name='Intake',
                                     line=dict(color='#27ae60', width=2), marker=dict(size=4)),
                            row=1, col=2
                        )
                
                # Metabolic Multiple
                if has_complete_profile and last_30_days['bmr_ratio'].notna().any():
                    fig_metabolic.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['bmr_ratio'],
                                 mode='lines+markers', name='Metabolic Multiple',
                                 line=dict(color='#9b59b6', width=2), marker=dict(size=4)),
                        row=2, col=1
                    )
                    fig_metabolic.add_hline(y=1.5, line_dash="dash", line_color="orange", row=2, col=1)
                
                # NEAT
                if has_complete_profile and last_30_days['neat_estimate'].notna().any():
                    fig_metabolic.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['neat_estimate'],
                                 mode='lines+markers', name='NEAT',
                                 line=dict(color='#f39c12', width=2), marker=dict(size=4)),
                        row=2, col=2
                    )
                
                fig_metabolic.update_layout(height=500, showlegend=False, 
                                          title_text="Daily Metabolic Health Indicators")
                fig_metabolic.update_xaxes(title_text="Date")
                fig_metabolic.update_yaxes(title_text="kcal", row=1, col=1)
                fig_metabolic.update_yaxes(title_text="kcal/day", row=1, col=2)
                fig_metabolic.update_yaxes(title_text="Ratio", row=2, col=1)
                fig_metabolic.update_yaxes(title_text="kcal", row=2, col=2)
                
                st.plotly_chart(fig_metabolic, use_container_width=True)

            with tab2:
                # Body composition trends
                if has_complete_profile:
                    fig_body = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Weight & BMI Trends", "Body Composition Indices"),
                        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
                    )
                    
                    # Weight and BMI
                    if last_30_days['weight'].notna().any():
                        fig_body.add_trace(
                            go.Scatter(x=last_30_days['date'], y=last_30_days['weight'],
                                     mode='lines+markers', name='Weight',
                                     line=dict(color='#2c3e50', width=3), marker=dict(size=5)),
                            row=1, col=1
                        )
                    
                    if last_30_days['bmi'].notna().any():
                        fig_body.add_trace(
                            go.Scatter(x=last_30_days['date'], y=last_30_days['bmi'],
                                     mode='lines', name='BMI',
                                     line=dict(color='#e67e22', width=2, dash='dot')),
                            row=1, col=1, secondary_y=True
                        )
                    
                    # Body composition indices (if available)
                    if has_body_fat:
                        if last_30_days['ffmi'].notna().any():
                            fig_body.add_trace(
                                go.Scatter(x=last_30_days['date'], y=last_30_days['ffmi'],
                                         mode='lines+markers', name='FFMI',
                                         line=dict(color='#8e44ad', width=2), marker=dict(size=4)),
                                row=1, col=2
                            )
                        
                        if last_30_days['fmi'].notna().any():
                            fig_body.add_trace(
                                go.Scatter(x=last_30_days['date'], y=last_30_days['fmi'],
                                         mode='lines+markers', name='FMI',
                                         line=dict(color='#e74c3c', width=2), marker=dict(size=4)),
                                row=1, col=2
                            )
                    
                    fig_body.update_layout(height=400, showlegend=True)
                    fig_body.update_xaxes(title_text="Date")
                    fig_body.update_yaxes(title_text="Weight (kg)", row=1, col=1)
                    fig_body.update_yaxes(title_text="BMI", row=1, col=1, secondary_y=True)
                    fig_body.update_yaxes(title_text="Index", row=1, col=2)
                    
                    st.plotly_chart(fig_body, use_container_width=True)
                else:
                    st.info("Complete your personal profile to unlock body composition trends")
            
            with tab3:
                # Activity patterns
                fig_activity = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Daily Steps", "Workout Duration", "Step Efficiency", "Activity Intensity"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Daily Steps
                if last_30_days['steps'].notna().any():
                    fig_activity.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['steps'],
                                 mode='lines+markers', name='Steps',
                                 line=dict(color='#3498db', width=2), marker=dict(size=4)),
                        row=1, col=1
                    )
                    fig_activity.add_hline(y=10000, line_dash="dash", line_color="green", row=1, col=1)
                
                # Workout Duration
                if last_30_days['workout_duration_min_tot'].notna().any():
                    fig_activity.add_trace(
                        go.Bar(x=last_30_days['date'], y=last_30_days['workout_duration_min_tot'],
                               name='Workout Duration', marker_color='#e74c3c'),
                        row=1, col=2
                    )
                
                # Step Efficiency
                if last_30_days['calories_per_1k_steps'].notna().any():
                    fig_activity.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['calories_per_1k_steps'],
                                 mode='lines+markers', name='Cal/1k Steps',
                                 line=dict(color='#f39c12', width=2), marker=dict(size=4)),
                        row=2, col=1
                    )
                
                # Activity Intensity
                if last_30_days['steps_per_workout_min'].notna().any():
                    fig_activity.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['steps_per_workout_min'],
                                 mode='lines+markers', name='Steps/Min',
                                 line=dict(color='#9b59b6', width=2), marker=dict(size=4)),
                        row=2, col=2
                    )
                
                fig_activity.update_layout(height=500, showlegend=False)
                fig_activity.update_xaxes(title_text="Date")
                fig_activity.update_yaxes(title_text="Steps", row=1, col=1)
                fig_activity.update_yaxes(title_text="Minutes", row=1, col=2)
                fig_activity.update_yaxes(title_text="kcal/1k steps", row=2, col=1)
                fig_activity.update_yaxes(title_text="Steps/min", row=2, col=2)
                
                st.plotly_chart(fig_activity, use_container_width=True)
                
                # Activity summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_daily_steps = last_30_days['steps'].mean()
                    st.metric("30-Day Avg Steps", f"{avg_daily_steps:,.0f}")
                
                with col2:
                    total_workout_time = last_30_days['workout_duration_min_tot'].sum()
                    st.metric("Total Workout Time", f"{total_workout_time:.0f} min")
                
                with col3:
                    active_days = len(last_30_days[last_30_days['workout_duration_min_tot'] > 0])
                    st.metric("Active Days", f"{active_days}/30")
            
            with tab4:
                # Recovery metrics
                fig_recovery = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Sleep Duration & Efficiency", "Recovery Score"),
                    specs=[[{"secondary_y": True}, {"secondary_y": False}]]
                )
                
                # Sleep metrics
                if last_30_days['sleep_hours'].notna().any():
                    fig_recovery.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['sleep_hours'],
                                 mode='lines+markers', name='Sleep Hours',
                                 line=dict(color='#2980b9', width=2), marker=dict(size=4)),
                        row=1, col=1
                    )
                    fig_recovery.add_hline(y=7, line_dash="dash", line_color="green", row=1, col=1)
                    fig_recovery.add_hline(y=8, line_dash="dash", line_color="blue", row=1, col=1)
                
                if last_30_days['sleep_efficiency'].notna().any():
                    fig_recovery.add_trace(
                        go.Scatter(x=last_30_days['date'], y=last_30_days['sleep_efficiency'],
                                 mode='lines+markers', name='Sleep Efficiency %',
                                 line=dict(color='#8e44ad', width=2), marker=dict(size=4)),
                        row=1, col=1, secondary_y=True
                    )
                    fig_recovery.add_hline(y=75, line_dash="dash", line_color="orange", 
                                         row=1, col=1, secondary_y=True)
                
                # Recovery score (combining sleep and activity)
                if (last_30_days['sleep_efficiency'].notna().any() and 
                    last_30_days['energy_balance'].notna().any()):
                    
                    # Create a simple recovery score
                    recovery_score = (
                        (last_30_days['sleep_efficiency'] / 100) * 0.6 +  # 60% weight to sleep
                        (np.minimum(np.abs(last_30_days['energy_balance']) / 500, 1)) * 0.4  # 40% to energy balance
                    ) * 100
                    
                    fig_recovery.add_trace(
                        go.Scatter(x=last_30_days['date'], y=recovery_score,
                                 mode='lines+markers', name='Recovery Score',
                                 line=dict(color='#27ae60', width=3), marker=dict(size=5)),
                        row=1, col=2
                    )
                    fig_recovery.add_hline(y=80, line_dash="dash", line_color="green", row=1, col=2)
                
                fig_recovery.update_layout(height=400, showlegend=True)
                fig_recovery.update_xaxes(title_text="Date")
                fig_recovery.update_yaxes(title_text="Hours", row=1, col=1)
                fig_recovery.update_yaxes(title_text="Efficiency %", row=1, col=1, secondary_y=True)
                fig_recovery.update_yaxes(title_text="Score", row=1, col=2)
                
                st.plotly_chart(fig_recovery, use_container_width=True)
                
                # Sleep summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_sleep = last_30_days['sleep_hours'].mean()
                    st.metric("Avg Sleep", f"{avg_sleep:.1f} hours")
                
                with col2:
                    avg_efficiency = last_30_days['sleep_efficiency'].mean()
                    st.metric("Avg Efficiency", f"{avg_efficiency:.0f}%")
                
                with col3:
                    good_sleep_days = len(last_30_days[last_30_days['sleep_hours'] >= 7])
                    st.metric("Good Sleep Days", f"{good_sleep_days}/30")