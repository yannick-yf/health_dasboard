"""
Unit tests for visualization_helpers.py
Tests the create_time_series_plot function
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

import sys
from pathlib import Path
# Adjust path to point to parent of 'utils' directory
# Assuming structure: health_dashboard/frontend/utils/
project_root = Path(__file__).parent.parent  # Goes to health_dashboard/tests/../
sys.path.insert(0, str(project_root / 'frontend'))  # Add frontend to path

from utils.visualization_helpers import create_time_series_plot

@pytest.fixture
def sample_daily_data():
    """Create sample daily health data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=45, freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'steps': np.random.randint(5000, 15000, size=45),
        'sleep_min': np.random.randint(360, 540, size=45),
        'workout_duration_min_tot': np.random.randint(0, 90, size=45),
        'weight': 75 + np.random.randn(45) * 0.5,
        'calories_burned': np.random.randint(2000, 3000, size=45),
        'calories_consumed': np.random.randint(1800, 2500, size=45)
    })
    
    return df


@pytest.fixture
def sample_data_with_nulls():
    """Create sample data with some null values"""
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'weight': [75.0, 74.8, np.nan, 74.5, np.nan, 74.2, 74.0, 73.9, 
                   np.nan, 73.7, 73.6, 73.5, np.nan, 73.3, 73.2, 
                   73.1, np.nan, 72.9, 72.8, 72.7]
    })
    
    return df


@pytest.fixture
def minimal_data():
    """Create minimal dataset (only 2 points)"""
    dates = pd.date_range(start='2024-01-01', periods=2, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'steps': [8000, 10000]
    })
    
    return df


class TestCreateTimeSeriesPlot:
    """Test suite for create_time_series_plot function"""
    
    def test_basic_plot_creation(self, sample_daily_data):
        """Test that function creates a valid plotly figure"""
        fig = create_time_series_plot(sample_daily_data, 'steps', 'Daily')
        
        # Check it returns a plotly Figure
        assert isinstance(fig, go.Figure)
        
        # Check that figure has data traces
        assert len(fig.data) > 0
        
        # Check main trace exists
        assert fig.data[0].name == 'Steps'
        assert fig.data[0].mode == 'lines+markers'
    
    def test_metric_configuration(self, sample_daily_data):
        """Test that different metrics use correct units and colors"""
        # Test weight metric
        fig_weight = create_time_series_plot(sample_daily_data, 'weight', 'Daily')
        
        # Check hover template includes correct unit
        assert 'kg' in fig_weight.data[0].hovertemplate
        
        # Check correct color is applied
        assert fig_weight.data[0].line.color == '#d62728'
        
        # Test calories burned metric
        fig_calories = create_time_series_plot(sample_daily_data, 'calories_burned', 'Daily')
        assert 'kcal' in fig_calories.data[0].hovertemplate
        assert fig_calories.data[0].line.color == '#ff6b6b'
    
    def test_moving_averages_daily(self, sample_daily_data):
        """Test that moving averages are added for daily aggregation with sufficient data"""
        fig = create_time_series_plot(sample_daily_data, 'weight', 'Daily')
        
        # With 45 days of data, should have:
        # 1. Main line
        # 2. 7-day moving average
        # 3. 30-day moving average
        # 4. Trend line
        assert len(fig.data) == 4
        
        # Check 7-day average exists
        assert any('7-day Average' in trace.name for trace in fig.data)
        
        # Check 30-day average exists
        assert any('30-day Average' in trace.name for trace in fig.data)
        
        # Check styling for moving averages
        ma7_trace = [t for t in fig.data if '7-day' in t.name][0]
        assert ma7_trace.line.dash == 'dash'
        assert ma7_trace.opacity == 0.7
    
    def test_no_moving_averages_for_short_data(self, minimal_data):
        """Test that moving averages are not added for short datasets"""
        fig = create_time_series_plot(minimal_data, 'steps', 'Daily')
        
        # With only 2 data points, should only have main line (no MA, no trend)
        assert len(fig.data) == 1
        assert fig.data[0].name == 'Steps'
    
    def test_trend_line_calculation(self, sample_daily_data):
        """Test that trend line is calculated correctly"""
        fig = create_time_series_plot(sample_daily_data, 'weight', 'Daily')
        
        # Find trend trace
        trend_trace = [t for t in fig.data if t.name == 'Trend'][0]
        
        # Check trend line properties
        assert trend_trace.line.dash == 'dashdot'
        assert trend_trace.opacity == 0.8
        
        # Trend color should be green, red, or gray based on direction
        assert trend_trace.line.color in ['#28a745', '#dc3545', '#6c757d']
    
    def test_weekly_aggregation(self, sample_daily_data):
        """Test that weekly aggregation skips moving averages"""
        fig = create_time_series_plot(sample_daily_data, 'steps', 'Weekly')
        
        # Weekly aggregation should not add 7-day or 30-day averages
        trace_names = [trace.name for trace in fig.data]
        assert '7-day Average' not in trace_names
        assert '30-day Average' not in trace_names
    
    def test_monthly_aggregation(self, sample_daily_data):
        """Test that monthly aggregation skips moving averages"""
        fig = create_time_series_plot(sample_daily_data, 'steps', 'Monthly')
        
        # Monthly aggregation should not add moving averages
        trace_names = [trace.name for trace in fig.data]
        assert '7-day Average' not in trace_names
        assert '30-day Average' not in trace_names
    
    def test_null_values_handling(self, sample_data_with_nulls):
        """Test that function handles null values correctly"""
        fig = create_time_series_plot(sample_data_with_nulls, 'weight', 'Daily')
        
        # Should create figure without errors
        assert isinstance(fig, go.Figure)
        
        # Should still calculate trend line with available data
        trace_names = [trace.name for trace in fig.data]
        assert 'Trend' in trace_names
    
    def test_layout_configuration(self, sample_daily_data):
        """Test that layout is configured correctly"""
        fig = create_time_series_plot(sample_daily_data, 'weight', 'Daily')
        
        # Check title
        assert 'Weight Over Time' in fig.layout.title.text
        assert 'Daily View' in fig.layout.title.text
        
        # Check axis labels
        assert fig.layout.xaxis.title.text == 'Date'
        assert 'Weight' in fig.layout.yaxis.title.text
        assert 'kg' in fig.layout.yaxis.title.text
        
        # Check height
        assert fig.layout.height == 450
        
        # Check hover mode
        assert fig.layout.hovermode == 'x unified'
    
    def test_grid_configuration(self, sample_daily_data):
        """Test that grid is properly configured"""
        fig = create_time_series_plot(sample_daily_data, 'steps', 'Daily')
        
        # Check x-axis grid
        assert fig.layout.xaxis.showgrid is True
        assert fig.layout.xaxis.gridcolor == 'lightgray'
        assert fig.layout.xaxis.griddash == 'dot'
        
        # Check y-axis grid
        assert fig.layout.yaxis.showgrid is True
        assert fig.layout.yaxis.gridcolor == 'lightgray'
        assert fig.layout.yaxis.griddash == 'dot'
    
    def test_unknown_metric_fallback(self, sample_daily_data):
        """Test that unknown metrics use default configuration"""
        # Add a custom metric not in the config
        sample_daily_data['custom_metric'] = np.random.randint(100, 200, size=len(sample_daily_data))
        
        fig = create_time_series_plot(sample_daily_data, 'custom_metric', 'Daily')
        
        # Should use default color
        assert fig.data[0].line.color == '#2ca02c'
        
        # Should still create valid figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_single_data_point(self):
        """Test with only one data point"""
        df = pd.DataFrame({
            'date': [datetime(2024, 1, 1)],
            'weight': [75.0]
        })
        
        fig = create_time_series_plot(df, 'weight', 'Daily')
        
        # Should create figure with just main line
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
    
    def test_all_null_values(self):
        """Test with all null values for a metric"""
        df = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'weight': [np.nan] * 10
        })
        
        fig = create_time_series_plot(df, 'weight', 'Daily')
        
        # Should create figure but with empty/null data
        assert isinstance(fig, go.Figure)
        # No trend line should be added
        trace_names = [trace.name for trace in fig.data]
        assert 'Trend' not in trace_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])