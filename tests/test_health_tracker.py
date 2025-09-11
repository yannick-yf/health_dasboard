"""
Unit tests for Health Tracker application
Run with: pytest tests/test_health_tracker.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.data_utils import (
    load_data, save_data, parse_date, validate_record,
    append_record, update_record, delete_record
)
from frontend.analytics import (
    compute_bmi, estimate_bmr, compute_tdee, compute_energy_balance,
    compute_sleep_consistency, detect_anomalies, rolling_stats,
    compute_trend_with_significance
)


class TestDataUtils:
    """Test data utility functions"""
    
    def test_parse_date(self):
        """Test date parsing functionality"""
        # Test dd/mm/yyyy format
        result = parse_date("15/08/2025")
        assert result.year == 2025
        assert result.month == 8
        assert result.day == 15
        
        # Test date object
        test_date = date(2025, 8, 15)
        result = parse_date(test_date)
        assert result.year == 2025
        assert result.month == 8
        assert result.day == 15
        
        # Test invalid date
        assert parse_date("invalid") is None
        assert parse_date(None) is None
    
    def test_validate_record(self):
        """Test record validation"""
        # Valid record
        record = {
            'date': date(2025, 8, 15),
            'steps': 10000,
            'sleep_min': 480,
            'workout_duration_min_tot': 60,
            'weight': 70,
            'calories_burned': 2500,
            'calories_consumed': 2000
        }
        is_valid, message = validate_record(record)
        assert is_valid is True
        
        # Missing date
        record_no_date = record.copy()
        del record_no_date['date']
        is_valid, message = validate_record(record_no_date)
        assert is_valid is False
        assert "Date is required" in message
        
        # Invalid weight
        record_invalid_weight = record.copy()
        record_invalid_weight['weight'] = 300
        is_valid, message = validate_record(record_invalid_weight)
        assert is_valid is False
        
        # Negative steps
        record_negative = record.copy()
        record_negative['steps'] = -100
        is_valid, message = validate_record(record_negative)
        assert is_valid is False
    
    def test_save_and_load_data(self):
        """Test saving and loading CSV data"""
        # Create test data
        df = pd.DataFrame({
            'user_id': ['', ''],
            'date': [datetime(2025, 8, 15), datetime(2025, 8, 16)],
            'steps': [10000, 12000],
            'sleep_min': [480, 450],
            'workout_duration_min_tot': [60, 90],
            'weight': [70.0, 70.5],
            'calories_burned': [2500, 2800],
            'calories_consumed': [2000, 2200]
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            save_data(df, tmp_path)
            
            # Load and verify
            loaded_df = load_data(tmp_path)
            assert len(loaded_df) == 2
            assert loaded_df['steps'].iloc[0] == 10000
            assert loaded_df['weight'].iloc[1] == 70.5
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_append_record(self):
        """Test appending records"""
        # Start with empty DataFrame
        df = pd.DataFrame(columns=['date', 'steps', 'weight'])
        
        # Add first record
        record1 = {'date': date(2025, 8, 15), 'steps': 10000, 'weight': 70}
        df = append_record(df, record1)
        assert len(df) == 1
        assert df['steps'].iloc[0] == 10000
        
        # Add second record
        record2 = {'date': date(2025, 8, 16), 'steps': 12000, 'weight': 70.5}
        df = append_record(df, record2)
        assert len(df) == 2
        assert df['steps'].iloc[1] == 12000
    
    def test_update_record(self):
        """Test updating records"""
        df = pd.DataFrame({
            'date': [pd.Timestamp(date(2025, 8, 15))],
            'steps': [10000],
            'weight': [70.0]
        })
        
        # Update record
        updates = {'steps': 15000, 'weight': 69.5}
        df = update_record(df, date(2025, 8, 15), updates)
        
        assert df['steps'].iloc[0] == 15000
        assert df['weight'].iloc[0] == 69.5
    
    def test_delete_record(self):
        """Test deleting records"""
        df = pd.DataFrame({
            'date': [pd.Timestamp(date(2025, 8, 15)), pd.Timestamp(date(2025, 8, 16))],
            'steps': [10000, 12000]
        })
        
        # Delete first record
        df = delete_record(df, date(2025, 8, 15))
        assert len(df) == 1
        assert df['steps'].iloc[0] == 12000


class TestAnalytics:
    """Test analytics functions"""
    
    def test_compute_bmi(self):
        """Test BMI calculation"""
        # Normal BMI
        bmi = compute_bmi(70, 1.75)
        assert abs(bmi - 22.86) < 0.01
        
        # Edge cases
        with pytest.raises(ValueError):
            compute_bmi(70, 0)  # Zero height
    
    def test_estimate_bmr(self):
        """Test BMR calculation"""
        # Male
        bmr_male = estimate_bmr(70, 175, 30, "Male")
        expected_male = 10 * 70 + 6.25 * 175 - 5 * 30 + 5
        assert abs(bmr_male - expected_male) < 0.01
        
        # Female
        bmr_female = estimate_bmr(60, 165, 25, "Female")
        expected_female = 10 * 60 + 6.25 * 165 - 5 * 25 - 161
        assert abs(bmr_female - expected_female) < 0.01
        
        # Invalid sex
        with pytest.raises(ValueError):
            estimate_bmr(70, 175, 30, "Invalid")
    
    def test_compute_tdee(self):
        """Test TDEE calculation"""
        bmr = 1500
        
        # Sedentary
        tdee_sed = compute_tdee(bmr, 1.2)
        assert tdee_sed == 1800
        
        # Active
        tdee_active = compute_tdee(bmr, 1.725)
        assert tdee_active == 2587.5
        
        # Invalid activity factor
        with pytest.raises(ValueError):
            compute_tdee(bmr, 0.5)
    
    def test_compute_energy_balance(self):
        """Test energy balance calculation"""
        # Deficit
        balance = compute_energy_balance(2000, 2500)
        assert balance == -500
        
        # Surplus
        balance = compute_energy_balance(3000, 2500)
        assert balance == 500
        
        # Maintenance
        balance = compute_energy_balance(2500, 2500)
        assert balance == 0
    
    def test_compute_sleep_consistency(self):
        """Test sleep consistency calculation"""
        # Consistent sleep
        consistent = pd.Series([480, 490, 470, 485, 475])
        std = compute_sleep_consistency(consistent)
        assert std < 10
        
        # Inconsistent sleep
        inconsistent = pd.Series([400, 600, 350, 550, 450])
        std = compute_sleep_consistency(inconsistent)
        assert std > 80
        
        # Empty series
        empty = pd.Series([])
        std = compute_sleep_consistency(empty)
        assert std == 0.0
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Create data with clear anomalies
        normal_data = [100, 102, 98, 101, 99, 103, 97, 200, 5, 101]  # 200 and 5 are anomalies
        series = pd.Series(normal_data)
        
        # IQR method
        anomalies_iqr = detect_anomalies(series, method='iqr', threshold=1.5)
        assert anomalies_iqr.iloc[7] == True  # 200 is anomaly
        assert anomalies_iqr.iloc[8] == True  # 5 is anomaly
        
        # Z-score method
        anomalies_zscore = detect_anomalies(series, method='zscore', threshold=2)
        assert anomalies_zscore.iloc[7] == True  # 200 is anomaly
    
    def test_rolling_stats(self):
        """Test rolling statistics"""
        df = pd.DataFrame({
            'date': pd.date_range('2025-08-01', periods=10),
            'steps': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        })
        
        # 3-day rolling average
        rolling_avg = rolling_stats(df, 'steps', window=3)
        
        # Check third value (average of first 3)
        assert rolling_avg.iloc[2] == 2000  # (1000 + 2000 + 3000) / 3
        
        # Check last value
        assert rolling_avg.iloc[9] == 9000  # (8000 + 9000 + 10000) / 3
    
    def test_trend_with_significance(self):
        """Test trend calculation with significance"""
        # Increasing trend
        dates = pd.Series(pd.date_range('2025-08-01', periods=10))
        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        slope, p_value = compute_trend_with_significance(dates, values)
        assert slope > 0  # Positive trend
        assert p_value < 0.05  # Significant
        
        # No trend (random)
        values_random = pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        slope, p_value = compute_trend_with_significance(dates, values_random)
        assert abs(slope) < 0.001  # Near zero slope


class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow(self):
        """Test complete data entry and analysis workflow"""
        # Create initial data
        df = pd.DataFrame()
        
        # Add multiple records
        for i in range(7):
            record = {
                'user_id': '',
                'date': date(2025, 8, 1) + timedelta(days=i),
                'steps': 8000 + i * 1000,
                'sleep_min': 480 - i * 10,
                'workout_duration_min_tot': 30 + i * 10,
                'weight': 70 + i * 0.1,
                'calories_burned': 2500 + i * 100,
                'calories_consumed': 2000 + i * 50
            }
            df = append_record(df, record)
        
        assert len(df) == 7
        
        # Test analytics on the data
        bmi = compute_bmi(df['weight'].iloc[-1], 1.75)
        assert bmi > 0
        
        # Test sleep consistency
        sleep_std = compute_sleep_consistency(df['sleep_min'])
        assert sleep_std > 0
        
        # Test anomaly detection
        anomalies = detect_anomalies(df['steps'])
        assert len(anomalies) == len(df)
        
        # Test trend
        slope, p_value = compute_trend_with_significance(df['date'], df['weight'])
        assert slope > 0  # Weight increasing


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])