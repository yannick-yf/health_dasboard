"""
Unit tests for metrics_helpers.py
Tests weight trend calculation and sleep time formatting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
# Adjust path to point to parent of 'utils' directory
# Assuming structure: health_dashboard/frontend/utils/
project_root = Path(__file__).parent.parent  # Goes to health_dashboard/tests/../
sys.path.insert(0, str(project_root / 'frontend'))  # Add frontend to path

from utils.metrics_helpers import calculate_weight_trend, format_sleep_time

@pytest.fixture
def real_health_data():
    """Load actual health data matching the CSV format"""
    data = {
        'user_id': ['', '', '', '', '', ''],
        'date': ['04/08/2025', '05/08/2025', '06/08/2025', '07/08/2025', '08/08/2025', '09/08/2025'],
        'steps': [8754.0, 9184.0, 6810.0, 9321.0, 14292.0, 17114.0],
        'sleep_min': [357, 443, 445, 473, 445, 504],
        'workout_duration_min_tot': [121.0, 127.0, 54.0, 128.0, 153.0, 124.0],
        'weight': [74.0, 73.8, 73.7, 74.2, 73.1, 73.7],
        'calories_burned': [3148.0, 3176.0, 2701.0, 3449.0, 3174.0, 3073.0],
        'calories_consumed': [2598.0, 2608.0, 2610.0, 3007.0, 2758.0, 2780.0]
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    return df


@pytest.fixture
def weight_trend_data():
    """Create sample weight trend data"""
    dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
    
    # Simulate weight loss trend: 75kg -> 74kg over 10 days
    weights = [75.0, 74.8, 74.7, 74.5, 74.6, 74.4, 74.3, 74.2, 74.1, 74.0]
    
    df = pd.DataFrame({
        'date': dates,
        'weight': weights
    })
    
    return df


@pytest.fixture
def weight_with_nulls():
    """Create weight data with null values"""
    dates = pd.date_range(start='2025-01-01', periods=8, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'weight': [75.0, np.nan, 74.5, np.nan, 74.0, np.nan, 73.5, 73.0]
    })
    
    return df


class TestCalculateWeightTrend:
    """Test suite for calculate_weight_trend function"""
    
    def test_basic_weight_loss_trend(self, weight_trend_data):
        """Test calculation of weight loss trend"""
        # Calculate 7-day trend (should show weight loss)
        trend = calculate_weight_trend(weight_trend_data, days=7)
        
        assert trend is not None
        assert trend < 0  # Negative trend indicates weight loss
        
        # Check approximate value
        # From 75.0 to 74.2 over 7 days = -1.07% approximately
        assert -1.2 < trend < -0.1
    
    def test_real_health_data_trend(self, real_health_data):
        """Test with actual health data from CSV"""
        # Calculate 3-day trend on real data
        trend_3day = calculate_weight_trend(real_health_data, days=3)
        
        assert trend_3day is not None
        
        # Last 3 weights: 74.2, 73.1, 73.7
        # From 74.2 to 73.7 = -0.67% approximately
        assert isinstance(trend_3day, float)
        assert -1.0 < trend_3day < 0.0
        
        # Calculate 5-day trend
        trend_5day = calculate_weight_trend(real_health_data, days=5)
        assert trend_5day is not None
    
    def test_weight_gain_trend(self):
        """Test calculation with weight gain"""
        dates = pd.date_range(start='2025-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'weight': [70.0, 70.3, 70.5, 70.8, 71.0]
        })
        
        trend = calculate_weight_trend(df, days=5)
        
        assert trend is not None
        assert trend > 0  # Positive trend indicates weight gain
        
        # From 70.0 to 71.0 = +1.43% approximately
        assert 1.3 < trend < 1.5
    
    def test_stable_weight(self):
        """Test with stable weight (no trend)"""
        dates = pd.date_range(start='2025-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'weight': [75.0, 75.0, 75.0, 75.0, 75.0]
        })
        
        trend = calculate_weight_trend(df, days=5)
        
        assert trend is not None
        assert trend == 0.0  # No change
    
    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None"""
        # Only 1 data point
        df = pd.DataFrame({
            'date': [datetime(2025, 1, 1)],
            'weight': [75.0]
        })
        
        trend = calculate_weight_trend(df, days=3)
        assert trend is None
    
    def test_empty_dataframe_returns_none(self):
        """Test that empty dataframe returns None"""
        df = pd.DataFrame(columns=['date', 'weight'])
        
        trend = calculate_weight_trend(df, days=3)
        assert trend is None
    
    def test_all_null_weights_returns_none(self):
        """Test that all null weights returns None"""
        dates = pd.date_range(start='2025-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'weight': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        trend = calculate_weight_trend(df, days=3)
        assert trend is None
    
    def test_handles_null_values(self, weight_with_nulls):
        """Test that function handles null values correctly"""
        # Should skip null values and calculate from available data
        trend = calculate_weight_trend(weight_with_nulls, days=5)
        
        assert trend is not None
        # Available weights: 75.0, 74.5, 74.0, 73.5, 73.0
        # From 75.0 to 73.0 = -2.67% approximately
        assert -3.0 < trend < -2.0
    
    def test_days_greater_than_data_length(self, weight_trend_data):
        """Test when requested days exceed available data"""
        # Request 20 days but only have 10
        trend = calculate_weight_trend(weight_trend_data, days=20)
        
        assert trend is not None
        # Should use all available data (10 days)
        # From 75.0 to 74.0 = -1.33%
        assert -1.5 < trend < -1.2
    
    def test_zero_initial_weight_returns_none(self):
        """Test that zero initial weight returns None (avoids division by zero)"""
        dates = pd.date_range(start='2025-01-01', periods=3, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'weight': [0.0, 70.0, 71.0]
        })
        
        trend = calculate_weight_trend(df, days=3)
        assert trend is None
    
    def test_unsorted_dates_handled_correctly(self):
        """Test that unsorted data is handled (function sorts internally)"""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-05', '2025-01-01', '2025-01-03', '2025-01-02', '2025-01-04']),
            'weight': [74.0, 75.0, 74.5, 74.8, 74.2]
        })
        
        trend = calculate_weight_trend(df, days=5)
        
        assert trend is not None
        # Should sort and calculate from 75.0 to 74.0
        assert trend < 0


class TestFormatSleepTime:
    """Test suite for format_sleep_time function"""
    
    def test_basic_sleep_formatting(self):
        """Test basic conversion of minutes to hours and minutes"""
        # 7 hours
        result = format_sleep_time(420)
        assert result == "7h 0m"
        
        # 7.5 hours
        result = format_sleep_time(450)
        assert result == "7h 30m"
        
        # 8 hours 15 minutes
        result = format_sleep_time(495)
        assert result == "8h 15m"
    
    def test_real_health_data_sleep(self, real_health_data):
        """Test with actual sleep data from CSV"""
        # First entry: 357 minutes = 5h 57m
        result = format_sleep_time(real_health_data['sleep_min'].iloc[0])
        assert result == "5h 57m"
        
        # Second entry: 443 minutes = 7h 23m
        result = format_sleep_time(real_health_data['sleep_min'].iloc[1])
        assert result == "7h 23m"
        
        # Fifth entry: 445 minutes = 7h 25m
        result = format_sleep_time(real_health_data['sleep_min'].iloc[2])
        assert result == "7h 25m"
    
    def test_zero_minutes(self):
        """Test with zero minutes"""
        result = format_sleep_time(0)
        assert result == "0h 0m"
    
    def test_less_than_one_hour(self):
        """Test with less than 60 minutes"""
        result = format_sleep_time(45)
        assert result == "0h 45m"
        
        result = format_sleep_time(30)
        assert result == "0h 30m"
    
    def test_exactly_one_hour(self):
        """Test exactly 60 minutes"""
        result = format_sleep_time(60)
        assert result == "1h 0m"
    
    def test_very_long_sleep(self):
        """Test with unusually long sleep duration"""
        # 12 hours
        result = format_sleep_time(720)
        assert result == "12h 0m"
        
        # 24 hours (max possible)
        result = format_sleep_time(1440)
        assert result == "24h 0m"
    
    def test_null_value_returns_na(self):
        """Test that null/NaN values return 'N/A'"""
        result = format_sleep_time(np.nan)
        assert result == "N/A"
        
        result = format_sleep_time(pd.NA)
        assert result == "N/A"
        
        result = format_sleep_time(None)
        assert result == "N/A"
    
    def test_float_minutes(self):
        """Test that float values are handled correctly"""
        # 7.5 hours as float
        result = format_sleep_time(450.7)
        assert result == "7h 30m"  # Should truncate to int
        
        result = format_sleep_time(420.9)
        assert result == "7h 0m"
    
    def test_all_real_sleep_values(self, real_health_data):
        """Test formatting all sleep values from real data"""
        expected = [
            "5h 57m",  # 357 min
            "7h 23m",  # 443 min
            "7h 25m",  # 445 min
            "7h 53m",  # 473 min
            "7h 25m",  # 445 min
            "8h 24m"   # 504 min
        ]
        
        for i, minutes in enumerate(real_health_data['sleep_min']):
            result = format_sleep_time(minutes)
            assert result == expected[i], f"Failed at index {i}: expected {expected[i]}, got {result}"


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_weight_trend_with_single_valid_weight(self, weight_with_nulls):
        """Test trend calculation with mostly null weights"""
        df = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=5, freq='D'),
            'weight': [75.0, np.nan, np.nan, np.nan, np.nan]
        })
        
        trend = calculate_weight_trend(df, days=3)
        assert trend is None  # Not enough valid data points
    
    def test_negative_sleep_minutes(self):
        """Test sleep formatting with negative value (shouldn't happen but test anyway)"""
        result = format_sleep_time(-30)
        # Should still format, though negative sleep doesn't make sense
        assert "h" in result and "m" in result
    
    def test_very_small_weight_changes(self):
        """Test trend calculation with very small weight changes"""
        dates = pd.date_range(start='2025-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'weight': [75.00, 75.01, 75.02, 75.01, 75.03]
        })
        
        trend = calculate_weight_trend(df, days=5)
        
        assert trend is not None
        assert abs(trend) < 0.1  # Very small trend
    
    def test_integration_metrics_together(self, real_health_data):
        """Test using both functions together on real data"""
        # Calculate weight trend
        trend = calculate_weight_trend(real_health_data, days=3)
        assert trend is not None
        
        # Format all sleep times
        sleep_formatted = [format_sleep_time(m) for m in real_health_data['sleep_min']]
        assert len(sleep_formatted) == len(real_health_data)
        assert all('h' in s and 'm' in s for s in sleep_formatted)

