"""
Helper functions for calculating and formatting health metrics
"""

import pandas as pd


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