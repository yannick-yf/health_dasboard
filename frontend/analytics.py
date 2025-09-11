"""
Analytics functions for Health Tracker application
Implements research-backed health indicators and statistical analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def compute_bmi(weight_kg: float, height_m: float) -> float:
    """
    Calculate Body Mass Index (BMI)
    
    Formula: BMI = weight(kg) / height(m)²
    
    Args:
        weight_kg: Weight in kilograms
        height_m: Height in meters
    
    Returns:
        BMI value
    """
    if height_m <= 0:
        raise ValueError("Height must be positive")
    
    return weight_kg / (height_m ** 2)


def estimate_bmr(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
    """
    Estimate Basal Metabolic Rate using Mifflin-St Jeor equation
    
    Formulas:
    - Men: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age + 5
    - Women: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age - 161
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        sex: 'Male' or 'Female'
    
    Returns:
        BMR in kcal/day
    """
    base_bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age
    
    if sex.lower() in ['male', 'm', 'man']:
        return base_bmr + 5
    elif sex.lower() in ['female', 'f', 'woman']:
        return base_bmr - 161
    else:
        raise ValueError("Sex must be 'Male' or 'Female'")


def compute_tdee(bmr: float, activity_factor: float) -> float:
    """
    Calculate Total Daily Energy Expenditure
    
    Formula: TDEE = BMR × activity_factor
    
    Activity factors:
    - Sedentary (little/no exercise): 1.2
    - Lightly active (1-3 days/week): 1.375
    - Moderately active (3-5 days/week): 1.55
    - Very active (6-7 days/week): 1.725
    - Extra active (physical job): 1.9
    
    Args:
        bmr: Basal Metabolic Rate in kcal/day
        activity_factor: Activity multiplication factor
    
    Returns:
        TDEE in kcal/day
    """
    if activity_factor < 1.0 or activity_factor > 2.5:
        raise ValueError("Activity factor should be between 1.0 and 2.5")
    
    return bmr * activity_factor


def compute_energy_balance(calories_consumed: float, calories_burned: float) -> float:
    """
    Calculate daily energy balance
    
    Formula: energy_balance = calories_consumed - calories_burned
    
    Args:
        calories_consumed: Calories consumed in kcal
        calories_burned: Calories burned in kcal
    
    Returns:
        Energy balance in kcal (positive = surplus, negative = deficit)
    """
    return calories_consumed - calories_burned


def compute_sleep_consistency(sleep_series: pd.Series) -> float:
    """
    Calculate sleep consistency metric (standard deviation of sleep duration)
    
    Lower values indicate more consistent sleep patterns
    
    Args:
        sleep_series: Series of sleep durations in minutes
    
    Returns:
        Standard deviation of sleep duration
    """
    if len(sleep_series) < 2:
        return 0.0
    
    # Remove NaN values
    clean_series = sleep_series.dropna()
    
    if len(clean_series) < 2:
        return 0.0
    
    return clean_series.std()


def compute_activity_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weekly activity load (total steps and workout minutes)
    
    Args:
        df: DataFrame with date, steps, and workout_duration_min_tot columns
    
    Returns:
        DataFrame with weekly aggregated activity metrics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Ensure date column is datetime
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Set date as index for resampling
    df_copy = df_copy.set_index('date')
    
    # Resample to weekly
    weekly = df_copy.resample('W')[['steps', 'workout_duration_min_tot']].sum()
    
    return weekly


def detect_anomalies(series: pd.Series, method: str = 'iqr', threshold: float = 2.5) -> pd.Series:
    """
    Detect anomalies in a data series using IQR or Z-score method
    
    Args:
        series: Data series to analyze
        method: 'iqr' for Interquartile Range or 'zscore' for Z-score
        threshold: Threshold for anomaly detection (2.5 for IQR multiplier, 3 for z-score)
    
    Returns:
        Boolean series indicating anomalies (True = anomaly)
    """
    # Remove NaN values for calculation
    clean_series = series.dropna()
    
    if len(clean_series) < 3:
        return pd.Series([False] * len(series), index=series.index)
    
    if method == 'iqr':
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        anomalies = (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        mean = clean_series.mean()
        std = clean_series.std()
        
        if std == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        z_scores = np.abs((series - mean) / std)
        anomalies = z_scores > threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return anomalies


def compute_trend_with_significance(dates: pd.Series, values: pd.Series) -> Tuple[float, float]:
    """
    Compute linear trend with statistical significance using linear regression
    
    Args:
        dates: Series of dates
        values: Series of values
    
    Returns:
        Tuple of (slope, p_value) where slope is change per day
    """
    # Remove NaN values
    mask = values.notna()
    clean_dates = dates[mask]
    clean_values = values[mask]
    
    if len(clean_values) < 2:
        return np.nan, np.nan
    
    # Convert dates to numeric (days since first date)
    date_numeric = (clean_dates - clean_dates.min()).dt.days.values
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(date_numeric, clean_values)
    
    return slope, p_value


def rolling_stats(df: pd.DataFrame, col: str, window: int = 7) -> pd.Series:
    """
    Calculate rolling statistics for a column
    
    Args:
        df: DataFrame containing the data
        col: Column name to calculate rolling stats for
        window: Window size in days
    
    Returns:
        Series with rolling mean values
    """
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in DataFrame")
    
    return df[col].rolling(window=window, min_periods=1).mean()


def calculate_weekly_change(series: pd.Series, weeks_back: int = 1) -> float:
    """
    Calculate percentage change from previous week
    
    Args:
        series: Weekly aggregated series
        weeks_back: Number of weeks to look back
    
    Returns:
        Percentage change
    """
    if len(series) < weeks_back + 1:
        return np.nan
    
    current = series.iloc[-1]
    previous = series.iloc[-(weeks_back + 1)]
    
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    
    return ((current - previous) / previous) * 100


def estimate_caloric_needs(tdee: float, goal: str = 'maintain') -> Dict[str, float]:
    """
    Estimate caloric needs based on goals
    
    Args:
        tdee: Total Daily Energy Expenditure
        goal: 'maintain', 'lose', or 'gain'
    
    Returns:
        Dictionary with caloric recommendations
    """
    recommendations = {
        'maintain': tdee,
        'mild_deficit': tdee - 250,  # ~0.25 kg/week loss
        'moderate_deficit': tdee - 500,  # ~0.5 kg/week loss
        'mild_surplus': tdee + 250,  # ~0.25 kg/week gain
        'moderate_surplus': tdee + 500  # ~0.5 kg/week gain
    }
    
    if goal == 'lose':
        return {
            'target_calories': recommendations['moderate_deficit'],
            'weekly_change_kg': -0.5,
            'description': "Moderate deficit for ~0.5 kg/week weight loss"
        }
    elif goal == 'gain':
        return {
            'target_calories': recommendations['moderate_surplus'],
            'weekly_change_kg': 0.5,
            'description': "Moderate surplus for ~0.5 kg/week weight gain"
        }
    else:
        return {
            'target_calories': recommendations['maintain'],
            'weekly_change_kg': 0,
            'description': "Maintenance calories to maintain current weight"
        }


def analyze_workout_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze workout patterns and consistency
    
    Args:
        df: DataFrame with workout data
    
    Returns:
        Dictionary with workout analysis
    """
    if df.empty or 'workout_duration_min_tot' not in df.columns:
        return {}
    
    workout_data = df['workout_duration_min_tot'].dropna()
    
    if workout_data.empty:
        return {}
    
    analysis = {
        'total_workouts': (workout_data > 0).sum(),
        'avg_duration': workout_data[workout_data > 0].mean() if (workout_data > 0).any() else 0,
        'total_minutes': workout_data.sum(),
        'workout_days_pct': (workout_data > 0).sum() / len(df) * 100,
        'longest_workout': workout_data.max(),
        'consistency_score': 100 - (workout_data.std() / workout_data.mean() * 100) if workout_data.mean() > 0 else 0
    }
    
    return analysis


def analyze_sleep_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze sleep patterns and quality metrics
    
    Args:
        df: DataFrame with sleep data
    
    Returns:
        Dictionary with sleep analysis
    """
    if df.empty or 'sleep_min' not in df.columns:
        return {}
    
    sleep_data = df['sleep_min'].dropna()
    
    if sleep_data.empty:
        return {}
    
    analysis = {
        'avg_sleep_hours': sleep_data.mean() / 60,
        'sleep_consistency': compute_sleep_consistency(sleep_data),
        'nights_under_7h': (sleep_data < 420).sum(),
        'nights_over_9h': (sleep_data > 540).sum(),
        'optimal_nights': ((sleep_data >= 420) & (sleep_data <= 540)).sum(),
        'optimal_pct': ((sleep_data >= 420) & (sleep_data <= 540)).sum() / len(sleep_data) * 100
    }
    
    return analysis


from typing import Dict, Any