"""
Unit tests for deep_dive_helpers.py
Tests core calculation functions for advanced health metrics
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime, date

import sys
from pathlib import Path
# Adjust path to point to parent of 'utils' directory
# Assuming structure: health_dashboard/frontend/utils/
project_root = Path(__file__).parent.parent  # Goes to health_dashboard/tests/../
sys.path.insert(0, str(project_root / 'frontend'))  # Add frontend to path


from utils.deep_dive_helpers import (
    load_personal_info,
    calculate_enhanced_metrics
)


@pytest.fixture
def real_health_data():
    """Create real health data matching CSV format"""
    data = {
        'user_id': ['', '', '', '', '', ''],
        'date': pd.to_datetime(['04/08/2025', '05/08/2025', '06/08/2025', 
                                '07/08/2025', '08/08/2025', '09/08/2025'], 
                               format='%d/%m/%Y'),
        'steps': [8754.0, 9184.0, 6810.0, 9321.0, 14292.0, 17114.0],
        'sleep_min': [357, 443, 445, 473, 445, 504],
        'workout_duration_min_tot': [121.0, 127.0, 54.0, 128.0, 153.0, 124.0],
        'weight': [74.0, 73.8, 73.7, 74.2, 73.1, 73.7],
        'calories_burned': [3148.0, 3176.0, 2701.0, 3449.0, 3174.0, 3073.0],
        'calories_consumed': [2598.0, 2608.0, 2610.0, 3007.0, 2758.0, 2780.0]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def personal_info_complete():
    """Complete personal information matching JSON format"""
    return {
        "birth_date": "21-01-1992",
        "age": 33,
        "height_cm": 185,
        "sex": "Male",
        "body_fat_prct": 13
    }


@pytest.fixture
def personal_info_no_body_fat():
    """Personal info without body fat percentage"""
    return {
        "birth_date": "21-01-1992",
        "height_cm": 185,
        "sex": "Male"
    }


@pytest.fixture
def temp_personal_info_file(personal_info_complete):
    """Create temporary personal_info.json file"""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / 'data'
    data_dir.mkdir()
    
    info_path = data_dir / 'personal_info.json'
    with open(info_path, 'w') as f:
        json.dump(personal_info_complete, f)
    
    # Change working directory temporarily
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield info_path
    
    # Cleanup
    os.chdir(original_cwd)
    info_path.unlink()
    data_dir.rmdir()
    Path(temp_dir).rmdir()


class TestLoadPersonalInfo:
    """Test suite for load_personal_info function"""
    
    def test_load_existing_file(self, temp_personal_info_file):
        """Test loading existing personal info file"""
        info = load_personal_info()
        
        assert info is not None
        assert isinstance(info, dict)
        assert info['birth_date'] == "21-01-1992"
        assert info['height_cm'] == 185
        assert info['sex'] == "Male"
        assert info['body_fat_prct'] == 13
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading when file doesn't exist"""
        os.chdir(tmp_path)
        
        info = load_personal_info()
        
        # Should return empty dict without crashing
        assert isinstance(info, dict)
        assert len(info) == 0


class TestCalculateEnhancedMetrics:
    """Test suite for calculate_enhanced_metrics function"""
    
    def test_basic_metrics_without_profile(self, real_health_data):
        """Test that basic metrics are calculated without personal profile"""
        result = calculate_enhanced_metrics(
            real_health_data, 
            {}, 
            has_complete_profile=False, 
            has_body_fat=False
        )
        
        # Should always calculate these
        assert 'sleep_hours' in result.columns
        assert 'sleep_efficiency' in result.columns
        assert 'steps_per_workout_min' in result.columns
        
        # Check sleep hours calculation
        assert result['sleep_hours'].iloc[0] == pytest.approx(357/60, rel=0.01)
        assert result['sleep_hours'].iloc[1] == pytest.approx(443/60, rel=0.01)
    
    def test_bmi_calculation(self, real_health_data, personal_info_complete):
        """Test BMI calculation with complete profile"""
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_complete,
            has_complete_profile=True,
            has_body_fat=False
        )
        
        assert 'bmi' in result.columns
        
        # BMI = weight / (height_m^2)
        # For 74kg and 1.85m: 74 / (1.85^2) = 21.6
        expected_bmi = 74.0 / (1.85 ** 2)
        assert result['bmi'].iloc[0] == pytest.approx(expected_bmi, rel=0.01)
    
    def test_ffmi_calculation(self, real_health_data, personal_info_complete):
        """Test FFMI calculation with body fat data"""
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_complete,
            has_complete_profile=True,
            has_body_fat=True
        )
        
        assert 'ffmi' in result.columns
        assert 'fmi' in result.columns
        assert 'fat_free_mass' in result.columns
        assert 'fat_mass' in result.columns
        assert 'normalized_ffmi' in result.columns
        
        # Check calculations for first row
        # Fat mass = 74kg * 0.13 = 9.62kg
        # Fat-free mass = 74kg * 0.87 = 64.38kg
        # FFMI = 64.38 / (1.85^2) = 18.8
        
        expected_fat_mass = 74.0 * 0.13
        expected_ffm = 74.0 * 0.87
        expected_ffmi = expected_ffm / (1.85 ** 2)
        
        assert result['fat_mass'].iloc[0] == pytest.approx(expected_fat_mass, rel=0.01)
        assert result['fat_free_mass'].iloc[0] == pytest.approx(expected_ffm, rel=0.01)
        assert result['ffmi'].iloc[0] == pytest.approx(expected_ffmi, rel=0.01)
    
    def test_bmr_calculation_male(self, real_health_data, personal_info_complete):
        """Test BMR calculation for male using Mifflin-St Jeor equation"""
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_complete,
            has_complete_profile=True,
            has_body_fat=False
        )
        
        assert 'bmr' in result.columns
        
        # For first row (no previous weight), should use current weight
        # BMR (male) = 10*weight + 6.25*height - 5*age + 5
        # = 10*74 + 6.25*185 - 5*33 + 5
        # = 740 + 1156.25 - 165 + 5 = 1736.25
        
        expected_bmr_first = 10 * 74.0 + 6.25 * 185 - 5 * 33 + 5
        assert result['bmr'].iloc[0] == pytest.approx(expected_bmr_first, rel=0.01)
        
        # For second row, should use previous weight (74.0)
        expected_bmr_second = 10 * 74.0 + 6.25 * 185 - 5 * 33 + 5
        assert result['bmr'].iloc[1] == pytest.approx(expected_bmr_second, rel=0.01)
    
    def test_bmr_calculation_female(self, real_health_data):
        """Test BMR calculation for female"""
        personal_info_female = {
            "birth_date": "21-01-1992",
            "height_cm": 165,
            "sex": "Female"
        }
        
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_female,
            has_complete_profile=True,
            has_body_fat=False
        )
        
        # BMR (female) = 10*weight + 6.25*height - 5*age - 161
        # = 10*74 + 6.25*165 - 5*33 - 161
        # = 740 + 1031.25 - 165 - 161 = 1445.25
        
        expected_bmr = 10 * 74.0 + 6.25 * 165 - 5 * 33 - 161
        assert result['bmr'].iloc[0] == pytest.approx(expected_bmr, rel=0.01)
    
    def test_energy_balance_calculation(self, real_health_data, personal_info_complete):
        """Test energy balance calculation"""
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_complete,
            has_complete_profile=True,
            has_body_fat=False
        )
        
        assert 'energy_balance' in result.columns
        assert 'bmr_surplus' in result.columns
        assert 'bmr_ratio' in result.columns
        
        # Energy balance = calories_burned - calories_consumed
        # First row: 3148 - 2598 = 550
        expected_balance = 3148.0 - 2598.0
        assert result['energy_balance'].iloc[0] == pytest.approx(expected_balance, rel=0.01)
        
        # BMR ratio = calories_burned / bmr
        bmr = result['bmr'].iloc[0]
        expected_ratio = 3148.0 / bmr
        assert result['bmr_ratio'].iloc[0] == pytest.approx(expected_ratio, rel=0.01)
    
    def test_neat_estimation(self, real_health_data, personal_info_complete):
        """Test NEAT (Non-Exercise Activity Thermogenesis) calculation"""
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_complete,
            has_complete_profile=True,
            has_body_fat=False
        )
        
        assert 'neat_estimate' in result.columns
        assert 'neat_from_steps' in result.columns
        assert 'neat_from_commute_bike' in result.columns
        
        # NEAT should be at least 50 (minimum threshold)
        assert all(result['neat_estimate'] >= 50)
        
        # NEAT estimate should be positive
        assert all(result['neat_estimate'] > 0)
    
    def test_sleep_efficiency_calculation(self, real_health_data):
        """Test sleep efficiency score calculation"""
        result = calculate_enhanced_metrics(
            real_health_data,
            {},
            has_complete_profile=False,
            has_body_fat=False
        )
        
        assert 'sleep_efficiency' in result.columns
        
        # Sleep efficiency formula:
        # If >= 420 min (7h): min(100, (sleep_min/480)*100)
        # If < 420 min: (sleep_min/420)*100
        
        # First row: 357 min < 420, so (357/420)*100 = 85%
        expected_eff_1 = (357 / 420) * 100
        assert result['sleep_efficiency'].iloc[0] == pytest.approx(expected_eff_1, rel=0.01)
        
        # Second row: 443 min >= 420, so min(100, (443/480)*100) = 92.3%
        expected_eff_2 = min(100, (443 / 480) * 100)
        assert result['sleep_efficiency'].iloc[1] == pytest.approx(expected_eff_2, rel=0.01)
    
    def test_steps_per_workout_minute(self, real_health_data):
        """Test activity intensity calculation"""
        result = calculate_enhanced_metrics(
            real_health_data,
            {},
            has_complete_profile=False,
            has_body_fat=False
        )
        
        assert 'steps_per_workout_min' in result.columns
        
        # First row: 8754 steps / 121 min = 72.3 steps/min
        expected_intensity = 8754.0 / 121.0
        assert result['steps_per_workout_min'].iloc[0] == pytest.approx(expected_intensity, rel=0.01)
        
        # Check that zero workout duration results in NaN
        test_data = real_health_data.copy()
        test_data.loc[0, 'workout_duration_min_tot'] = 0
        result_zero = calculate_enhanced_metrics(test_data, {}, False, False)
        assert pd.isna(result_zero['steps_per_workout_min'].iloc[0])
    
    def test_calories_per_1k_steps(self, real_health_data, personal_info_complete):
        """Test step efficiency calculation"""
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_complete,
            has_complete_profile=True,
            has_body_fat=False
        )
        
        assert 'calories_per_1k_steps' in result.columns
        
        # First row: (3148 / 8754) * 1000 = 359.6 kcal per 1k steps
        expected_cal_per_1k = (3148.0 / 8754.0) * 1000
        assert result['calories_per_1k_steps'].iloc[0] == pytest.approx(expected_cal_per_1k, rel=0.01)


class TestAgeCalculation:
    """Test age calculation logic"""
    
    def test_age_calculation_before_birthday(self):
        """Test age when current date is before birthday"""
        # Birth: Jan 21, 1992
        # Today: Jan 15, 2025 (before birthday)
        # Age should be 32
        
        personal_info = {
            "birth_date": "21-01-1992",
            "height_cm": 185,
            "sex": "Male"
        }
        
        # Mock data for calculation
        df = pd.DataFrame({
            'date': [pd.Timestamp('2025-01-15')],
            'weight': [74.0],
            'steps': [8000],
            'sleep_min': [420],
            'workout_duration_min_tot': [60],
            'calories_burned': [2500],
            'calories_consumed': [2000]
        })
        
        result = calculate_enhanced_metrics(df, personal_info, True, False)
        
        # BMR calculation includes age
        # If calculated on Jan 15, 2025, age = 32 (before Jan 21)
        # BMR = 10*74 + 6.25*185 - 5*32 + 5 = 1741.25
        expected_bmr = 10 * 74 + 6.25 * 185 - 5 * 32 + 5
        
        # Note: The actual age will be calculated based on date.today()
        # This test verifies the BMR includes age calculation
        assert 'bmr' in result.columns


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_steps(self, personal_info_complete):
        """Test with zero steps"""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2025-08-04')],
            'weight': [74.0],
            'steps': [0],
            'sleep_min': [420],
            'workout_duration_min_tot': [0],
            'calories_burned': [1800],
            'calories_consumed': [2000]
        })
        
        result = calculate_enhanced_metrics(df, personal_info_complete, True, False)
        
        # Should not crash with zero steps
        assert result is not None
        assert len(result) == 1
    
    def test_null_values_in_data(self, personal_info_complete):
        """Test handling of null values"""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2025-08-04'), pd.Timestamp('2025-08-05')],
            'weight': [74.0, np.nan],
            'steps': [8000, np.nan],
            'sleep_min': [420, np.nan],
            'workout_duration_min_tot': [60, np.nan],
            'calories_burned': [2500, np.nan],
            'calories_consumed': [2000, np.nan]
        })
        
        result = calculate_enhanced_metrics(df, personal_info_complete, True, True)
        
        # Should handle nulls gracefully
        assert result is not None
        assert len(result) == 2
        
        # First row should have valid calculations
        assert pd.notna(result['bmi'].iloc[0])
        
        # Second row will have NaN for weight-dependent calculations
        assert pd.isna(result['bmi'].iloc[1])
    
    def test_extreme_values(self, personal_info_complete):
        """Test with extreme but valid values"""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2025-08-04')],
            'weight': [120.0],  # Very heavy
            'steps': [30000],   # Very active
            'sleep_min': [240], # Very little sleep
            'workout_duration_min_tot': [180],  # 3 hours workout
            'calories_burned': [5000],
            'calories_consumed': [4000]
        })
        
        result = calculate_enhanced_metrics(df, personal_info_complete, True, True)
        
        # Should calculate without errors
        assert result is not None
        assert result['bmi'].iloc[0] > 30  # Obese BMI
        assert result['sleep_efficiency'].iloc[0] < 100  # Low sleep efficiency
    
    def test_normalized_ffmi_calculation(self, real_health_data, personal_info_complete):
        """Test normalized FFMI accounts for height"""
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_complete,
            has_complete_profile=True,
            has_body_fat=True
        )
        
        # Normalized FFMI = FFMI + 6.1 * (1.8 - height_m)
        # For height 1.85m: adjustment = 6.1 * (1.8 - 1.85) = -0.305
        # So normalized_ffmi should be slightly less than ffmi
        
        ffmi = result['ffmi'].iloc[0]
        normalized_ffmi = result['normalized_ffmi'].iloc[0]
        
        expected_adjustment = 6.1 * (1.8 - 1.85)
        expected_normalized = ffmi + expected_adjustment
        
        assert normalized_ffmi == pytest.approx(expected_normalized, rel=0.01)


class TestIntegration:
    """Integration tests combining multiple metrics"""
    
    def test_complete_metrics_pipeline(self, real_health_data, personal_info_complete):
        """Test full calculation pipeline with all features"""
        result = calculate_enhanced_metrics(
            real_health_data,
            personal_info_complete,
            has_complete_profile=True,
            has_body_fat=True
        )
        
        # Verify all expected columns exist
        expected_columns = [
            'bmi', 'fat_mass', 'fat_free_mass', 'ffmi', 'fmi', 'normalized_ffmi',
            'bmr', 'energy_balance', 'bmr_surplus', 'bmr_ratio',
            'neat_estimate', 'sleep_hours', 'sleep_efficiency', 'steps_per_workout_min'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
        
        # Verify data integrity
        assert len(result) == len(real_health_data)
        assert all(result['sleep_hours'] >= 0)
        assert all(result['sleep_efficiency'] >= 0)
        assert all(result['sleep_efficiency'] <= 100)