"""
ETL Package for Health Dashboard Data Transformation

This package provides tools for transforming raw health data into enriched datasets
with pre-computed features like moving averages and derived health metrics.
"""

from .transform_health_data import HealthDataETL, run_etl

__all__ = ['HealthDataETL', 'run_etl']
__version__ = '1.0.0'
