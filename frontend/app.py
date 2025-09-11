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
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                       value=df['date'].min() if not df.empty else date.today())
        with col2:
            end_date = st.date_input("End Date", 
                                     value=df['date'].max() if not df.empty else date.today())
        
        # Filter data by date range
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        filtered_df = df[mask].copy()
        
        if filtered_df.empty:
            st.warning("No data in selected date range.")
        else:
            # KPI Cards
            st.subheader("Key Performance Indicators")
            kpi_cols = st.columns(6)
            
            metrics = [
                ("🚶 Avg Steps", filtered_df['steps'].mean(), "{:.0f}"),
                ("😴 Avg Sleep", filtered_df['sleep_min'].mean(), "{:.0f} min"),
                ("💪 Total Workout", filtered_df['workout_duration_min_tot'].sum(), "{:.0f} min"),
                ("⚖️ Avg Weight", filtered_df['weight'].mean(), "{:.1f} kg"),
                ("🔥 Avg Burned", filtered_df['calories_burned'].mean(), "{:.0f} kcal"),
                ("🍽️ Avg Consumed", filtered_df['calories_consumed'].mean(), "{:.0f} kcal")
            ]
            
            for col, (label, value, fmt) in zip(kpi_cols, metrics):
                if pd.notna(value):
                    col.metric(label, fmt.format(value))
                else:
                    col.metric(label, "N/A")
            
            # Quick visualization
            st.subheader("Recent Trends")
            
            # Steps trend
            if filtered_df['steps'].notna().any():
                fig = px.line(filtered_df, x='date', y='steps', 
                             title='Daily Steps', markers=True)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Weight trend
            if filtered_df['weight'].notna().any():
                fig = px.line(filtered_df, x='date', y='weight', 
                             title='Weight Trend', markers=True)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# Data Entry Page
elif page == "➕ Data Entry":
    st.title("➕ Data Entry")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Add Record", "Edit Record", "Delete Record", "Import/Export"])
    
    with tab1:
        st.subheader("Add New Daily Record")
        
        with st.form("add_record_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                record_date = st.date_input("Date", value=date.today())
                steps = st.number_input("Steps (count)", min_value=0, value=None)
                sleep_min = st.number_input("Sleep (minutes)", min_value=0, value=None)
                workout_min = st.number_input("Workout duration (min)", min_value=0, value=None)
            
            with col2:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=None)
                calories_burned = st.number_input("Calories burned (kcal)", min_value=0, value=None)
                calories_consumed = st.number_input("Calories consumed (kcal)", min_value=0, value=None)
            
            submit = st.form_submit_button("Add Record")
            
            if submit:
                record = {
                    'user_id': '',
                    'date': record_date,
                    'steps': steps,
                    'sleep_min': sleep_min,
                    'workout_duration_min_tot': workout_min,
                    'weight': weight,
                    'calories_burned': calories_burned,
                    'calories_consumed': calories_consumed
                }
                
                # Validate record
                is_valid, message = validate_record(record)
                
                if is_valid:
                    # Check if date already exists
                    if not df.empty and (df['date'] == pd.Timestamp(record_date)).any():
                        st.error(f"Record for {record_date} already exists. Use Edit Record to update.")
                    else:
                        create_backup(st.session_state.csv_path)
                        df_new = append_record(df, record)
                        save_data(df_new, st.session_state.csv_path)
                        st.success(f"Record for {record_date} added successfully!")
                        st.rerun()
                else:
                    st.error(message)
    
    with tab2:
        st.subheader("Edit Existing Record")
        
        if df.empty:
            st.info("No records to edit.")
        else:
            edit_date = st.selectbox("Select date to edit", 
                                     df['date'].dt.date.sort_values(ascending=False).unique())
            
            if edit_date:
                existing = df[df['date'] == pd.Timestamp(edit_date)].iloc[0]
                
                with st.form("edit_record_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        steps = st.number_input("Steps", min_value=0, 
                                              value=int(existing['steps']) if pd.notna(existing['steps']) else None)
                        sleep_min = st.number_input("Sleep (minutes)", min_value=0,
                                                   value=int(existing['sleep_min']) if pd.notna(existing['sleep_min']) else None)
                        workout_min = st.number_input("Workout duration (min)", min_value=0,
                                                     value=int(existing['workout_duration_min_tot']) if pd.notna(existing['workout_duration_min_tot']) else None)
                    
                    with col2:
                        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0,
                                               value=float(existing['weight']) if pd.notna(existing['weight']) else None)
                        calories_burned = st.number_input("Calories burned (kcal)", min_value=0,
                                                        value=int(existing['calories_burned']) if pd.notna(existing['calories_burned']) else None)
                        calories_consumed = st.number_input("Calories consumed (kcal)", min_value=0,
                                                          value=int(existing['calories_consumed']) if pd.notna(existing['calories_consumed']) else None)
                    
                    update = st.form_submit_button("Update Record")
                    
                    if update:
                        updates = {
                            'steps': steps,
                            'sleep_min': sleep_min,
                            'workout_duration_min_tot': workout_min,
                            'weight': weight,
                            'calories_burned': calories_burned,
                            'calories_consumed': calories_consumed
                        }
                        
                        create_backup(st.session_state.csv_path)
                        df_new = update_record(df, edit_date, updates)
                        save_data(df_new, st.session_state.csv_path)
                        st.success(f"Record for {edit_date} updated successfully!")
                        st.rerun()
    
    with tab3:
        st.subheader("Delete Record")
        
        if df.empty:
            st.info("No records to delete.")
        else:
            delete_date = st.selectbox("Select date to delete",
                                      df['date'].dt.date.sort_values(ascending=False).unique())
            
            if st.button("Delete Record", type="secondary"):
                create_backup(st.session_state.csv_path)
                df_new = delete_record(df, delete_date)
                save_data(df_new, st.session_state.csv_path)
                st.success(f"Record for {delete_date} deleted successfully!")
                st.rerun()
    
    with tab4:
        st.subheader("Import/Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Current Data**")
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No data to export.")
        
        with col2:
            st.write("**Import Data**")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    new_df = pd.read_csv(uploaded_file)
                    # Validate columns
                    required_cols = ['date', 'steps', 'sleep_min', 'workout_duration_min_tot',
                                    'weight', 'calories_burned', 'calories_consumed']
                    
                    if all(col in new_df.columns for col in required_cols):
                        create_backup(st.session_state.csv_path)
                        save_data(new_df, st.session_state.csv_path)
                        st.success("Data imported successfully!")
                        st.rerun()
                    else:
                        st.error("CSV missing required columns.")
                except Exception as e:
                    st.error(f"Error importing file: {e}")

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