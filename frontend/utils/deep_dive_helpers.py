"""
Helper functions for Deep Dive analytics page
Part 1: Data loading, calculations, and glossary
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_personal_info():
    """Load personal information from JSON file"""
    personal_info_path = 'data/personal_info.json'
    personal_info = {}
    
    try:
        if os.path.exists(personal_info_path):
            with open(personal_info_path, 'r') as f:
                personal_info = json.load(f)
    except Exception as e:
        st.error(f"Error loading personal info: {e}")
    
    return personal_info


def calculate_enhanced_metrics(df, personal_info, has_complete_profile, has_body_fat):
    """Calculate enhanced daily-level metrics"""
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
        
        # Enhanced NEAT calculation
        KCAL_PER_STEP = 0.04
        COMMUTE_BIKE_DAYS_PER_WEEK = 5
        COMMUTE_BIKE_MINUTES_PER_DAY = 40
        COMMUTE_BIKE_KCAL_PER_MIN = 7
        
        avg_commute_kcal_per_day = (COMMUTE_BIKE_DAYS_PER_WEEK * COMMUTE_BIKE_MINUTES_PER_DAY * COMMUTE_BIKE_KCAL_PER_MIN) / 7
        
        estimated_commute_bike_kcal = np.where(
            df_enhanced['workout_duration_min_tot'] > 0,
            np.minimum(avg_commute_kcal_per_day * 1.5, df_enhanced['workout_duration_min_tot'] * COMMUTE_BIKE_KCAL_PER_MIN),
            0
        )
        
        structured_exercise_minutes = np.maximum(
            0, 
            np.minimum(df_enhanced['workout_duration_min_tot'] - COMMUTE_BIKE_MINUTES_PER_DAY, 65)
        )
        estimated_exercise_calories = structured_exercise_minutes * 10
        
        estimated_steps_calories = df_enhanced['steps'] * KCAL_PER_STEP
        estimated_tef = df_enhanced['calories_consumed'] * 0.1
        
        df_enhanced['neat_estimate'] = (
            df_enhanced['calories_burned']
            - df_enhanced['bmr']
            - estimated_tef
            - estimated_exercise_calories
        )
        
        df_enhanced['neat_from_steps'] = estimated_steps_calories
        df_enhanced['neat_from_commute_bike'] = estimated_commute_bike_kcal
        df_enhanced['neat_other'] = df_enhanced['neat_estimate'] - estimated_steps_calories - estimated_commute_bike_kcal
        
        df_enhanced['neat_estimate'] = np.maximum(df_enhanced['neat_estimate'], 50)
        df_enhanced['calories_per_1k_steps'] = (df_enhanced['calories_burned'] / df_enhanced['steps']) * 1000
    
    # Calculate sleep and activity metrics
    df_enhanced['sleep_hours'] = df_enhanced['sleep_min'] / 60
    df_enhanced['sleep_efficiency'] = np.where(
        df_enhanced['sleep_min'] >= 420,
        np.minimum(100, (df_enhanced['sleep_min'] / 480) * 100),
        (df_enhanced['sleep_min'] / 420) * 100
    )
    
    df_enhanced['steps_per_workout_min'] = np.where(
        df_enhanced['workout_duration_min_tot'] > 0,
        df_enhanced['steps'] / df_enhanced['workout_duration_min_tot'],
        np.nan
    )
    
    return df_enhanced


def render_metrics_glossary():
    """Render the metrics glossary section"""
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

def render_weekly_overview(complete_week_data, has_complete_profile):
    """Render weekly overview metrics"""
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
            st.metric("Sleep Quality", f"{avg_sleep_eff:.0f}%", help="Sleep efficiency score")
        else:
            st.metric("Sleep Quality", "N/A")
    
    with col4:
        if complete_week_data['steps'].notna().any():
            avg_steps = complete_week_data['steps'].mean()
            st.metric("Daily Steps", f"{avg_steps:,.0f}", help="Average daily step count")
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


def render_body_composition(complete_week_data, has_complete_profile, has_body_fat, personal_info):
    """Render body composition analysis"""
    st.markdown("---")
    st.subheader("💪 Weekly Body Composition Analysis")
    
    if has_complete_profile and has_body_fat:
        fig_composition = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Fat-Free Mass Index", "Fat Mass Index", "Normalized FFMI"),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        sex = personal_info['sex']
        
        if complete_week_data['ffmi'].notna().any():
            avg_ffmi = complete_week_data['ffmi'].mean()
            
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if complete_week_data['fat_free_mass'].notna().any():
                avg_ffm = complete_week_data['fat_free_mass'].mean()
                body_fat_prct = personal_info.get('body_fat_prct')
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


def render_metabolic_analysis(complete_week_data, has_complete_profile):
    """Render metabolic analysis section"""
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
            
            if avg_neat < 250:
                st.error("🟥 Very Low NEAT - Consider increasing daily movement")
            elif avg_neat < 400:
                st.warning("🟨 Low NEAT - Typical sedentary lifestyle")
            elif avg_neat < 600:
                st.info("🟦 Moderate NEAT - Good daily activity level")
            else:
                st.success("🟩 High NEAT - Excellent daily movement!")
    
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

def render_activity_performance(complete_week_data):
    """Render activity and performance metrics"""
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


def render_daily_trends(df_enhanced, has_complete_profile, has_body_fat):
    """Render daily trends visualization for last 30 days"""
    st.markdown("---")
    st.subheader("📈 Daily Health Trends (Last 30 Days)")
    
    last_30_days = df_enhanced.tail(30)
    
    if len(last_30_days) > 7:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔥 Metabolic Trends", 
            "💪 Body Composition", 
            "🏃 Activity Patterns", 
            "😴 Recovery Metrics"
        ])
        
        with tab1:
            _render_metabolic_trends(last_30_days, has_complete_profile)
        
        with tab2:
            _render_body_composition_trends(last_30_days, has_complete_profile, has_body_fat)
        
        with tab3:
            _render_activity_patterns(last_30_days)
        
        with tab4:
            _render_recovery_metrics(last_30_days)


def _render_metabolic_trends(last_30_days, has_complete_profile):
    """Render metabolic trends charts"""
    fig_metabolic = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Energy Balance", "BMR vs Intake", "Metabolic Multiple", "NEAT Estimation"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if last_30_days['energy_balance'].notna().any():
        fig_metabolic.add_trace(
            go.Scatter(x=last_30_days['date'], y=last_30_days['energy_balance'],
                     mode='lines+markers', name='Energy Balance',
                     line=dict(color='#3498db', width=2), marker=dict(size=4)),
            row=1, col=1
        )
        fig_metabolic.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
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
    
    if has_complete_profile and last_30_days['bmr_ratio'].notna().any():
        fig_metabolic.add_trace(
            go.Scatter(x=last_30_days['date'], y=last_30_days['bmr_ratio'],
                     mode='lines+markers', name='Metabolic Multiple',
                     line=dict(color='#9b59b6', width=2), marker=dict(size=4)),
            row=2, col=1
        )
        fig_metabolic.add_hline(y=1.5, line_dash="dash", line_color="orange", row=2, col=1)
    
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


def _render_body_composition_trends(last_30_days, has_complete_profile, has_body_fat):
    """Render body composition trends"""
    if has_complete_profile:
        fig_body = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Weight & BMI Trends", "Body Composition Indices"),
            specs=[[{"secondary_y": True}, {"secondary_y": False}]]
        )
        
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


def _render_activity_patterns(last_30_days):
    """Render activity patterns"""
    fig_activity = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Daily Steps", "Workout Duration", "Step Efficiency", "Activity Intensity"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if last_30_days['steps'].notna().any():
        fig_activity.add_trace(
            go.Scatter(x=last_30_days['date'], y=last_30_days['steps'],
                     mode='lines+markers', name='Steps',
                     line=dict(color='#3498db', width=2), marker=dict(size=4)),
            row=1, col=1
        )
        fig_activity.add_hline(y=10000, line_dash="dash", line_color="green", row=1, col=1)
    
    if last_30_days['workout_duration_min_tot'].notna().any():
        fig_activity.add_trace(
            go.Bar(x=last_30_days['date'], y=last_30_days['workout_duration_min_tot'],
                   name='Workout Duration', marker_color='#e74c3c'),
            row=1, col=2
        )
    
    if last_30_days['calories_per_1k_steps'].notna().any():
        fig_activity.add_trace(
            go.Scatter(x=last_30_days['date'], y=last_30_days['calories_per_1k_steps'],
                     mode='lines+markers', name='Cal/1k Steps',
                     line=dict(color='#f39c12', width=2), marker=dict(size=4)),
            row=2, col=1
        )
    
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


def _render_recovery_metrics(last_30_days):
    """Render recovery metrics"""
    fig_recovery = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Sleep Duration & Efficiency", "Recovery Score"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
    )
    
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
    
    if (last_30_days['sleep_efficiency'].notna().any() and 
        last_30_days['energy_balance'].notna().any()):
        
        recovery_score = (
            (last_30_days['sleep_efficiency'] / 100) * 0.6 +
            (np.minimum(np.abs(last_30_days['energy_balance']) / 500, 1)) * 0.4
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