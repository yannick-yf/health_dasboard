"""
ETL Script for Health Dashboard Data Transformation

This script performs Extract, Transform, and Load operations on health data:
- Extract: Load raw health data from CSV
- Transform: Calculate moving averages and derived features
- Load: Save enriched data with pre-computed features

Features Generated:
- Moving averages (3, 5, 7 days) for: weight, steps, calories_consumed,
  calories_burned, sleep_min, workout_duration_min_tot
- Derived metrics: BMI, BMR, TDEE, energy_balance

Usage:
    python etl/transform_health_data.py

Or import and use programmatically:
    from etl.transform_health_data import run_etl
    run_etl()
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthDataETL:
    """ETL processor for health dashboard data"""

    def __init__(self,
                 input_csv: str = "data/health_data.csv",
                 output_csv: str = "data/health_data_enriched.csv",
                 personal_info_json: str = "data/personal_info.json"):
        """
        Initialize ETL processor

        Args:
            input_csv: Path to raw health data CSV
            output_csv: Path to save enriched data
            personal_info_json: Path to personal information JSON
        """
        # Get project root directory
        self.project_root = Path(__file__).parent.parent
        self.input_csv = self.project_root / input_csv
        self.output_csv = self.project_root / output_csv
        self.personal_info_json = self.project_root / personal_info_json

        # Personal info for derived calculations
        self.personal_info = None

        # Metrics to calculate moving averages for
        self.ma_metrics = [
            'weight',
            'steps',
            'calories_consumed',
            'calories_burned',
            'sleep_min',
            'workout_duration_min_tot'
        ]

        # Moving average windows
        self.ma_windows = [3, 5, 7]

    def extract(self) -> pd.DataFrame:
        """
        Extract: Load raw health data from CSV

        Returns:
            DataFrame with raw health data
        """
        logger.info(f"Extracting data from {self.input_csv}")

        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_csv}")

        # Load CSV with date parsing
        df = pd.read_csv(
            self.input_csv,
            parse_dates=['date'],
            dayfirst=True  # Format is dd/mm/yyyy
        )

        # Sort by date to ensure proper rolling window calculations
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Extracted {len(df)} records from {df['date'].min()} to {df['date'].max()}")

        return df

    def load_personal_info(self) -> Dict[str, Any]:
        """
        Load personal information for derived calculations

        Returns:
            Dictionary with personal information
        """
        logger.info(f"Loading personal info from {self.personal_info_json}")

        if not self.personal_info_json.exists():
            logger.warning(f"Personal info file not found: {self.personal_info_json}")
            # Return default values
            return {
                'age': 30,
                'height_cm': 175,
                'sex': 'Male',
                'body_fat_prct': 15.0,
                'activity_factor': 1.55  # Moderately active
            }

        with open(self.personal_info_json, 'r') as f:
            personal_info = json.load(f)

        # Add activity factor if not present (default: moderately active)
        if 'activity_factor' not in personal_info:
            personal_info['activity_factor'] = 1.55

        height_display = personal_info.get('height_cm', 175) / 100
        logger.info(f"Loaded personal info: age={personal_info.get('age')}, height={height_display:.2f}m")

        return personal_info

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform: Calculate moving averages and derived features

        Uses formulas from frontend/app.py and frontend/analytics.py:
        - Moving average: df[metric].rolling(window=X, min_periods=1).mean()
        - BMI: weight_kg / (height_m ** 2)
        - BMR: Mifflin-St Jeor equation
        - TDEE: BMR × activity_factor
        - Energy Balance: calories_consumed - calories_burned

        Args:
            df: DataFrame with raw health data

        Returns:
            DataFrame with enriched features
        """
        logger.info("Transforming data: calculating features")

        df_enriched = df.copy()

        # Load personal info for derived calculations
        self.personal_info = self.load_personal_info()

        # 1. Calculate Moving Averages
        logger.info("Calculating moving averages...")
        df_enriched = self._calculate_moving_averages(df_enriched)

        # 2. Calculate Derived Features
        logger.info("Calculating derived features...")
        df_enriched = self._calculate_derived_features(df_enriched)

        # 3. Add metadata columns
        df_enriched['etl_processed_at'] = datetime.now().isoformat()
        df_enriched['etl_version'] = '1.0.0'

        logger.info(f"Transformation complete. Added {len(df_enriched.columns) - len(df.columns)} new features")

        return df_enriched

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages for specified metrics and windows

        Formula from frontend/app.py:304:
            ma7 = filtered_df[metric].rolling(window=7, min_periods=1).mean()

        Args:
            df: DataFrame with health data

        Returns:
            DataFrame with moving average columns added
        """
        df_ma = df.copy()

        for metric in self.ma_metrics:
            if metric not in df_ma.columns:
                logger.warning(f"Metric {metric} not found in data, skipping")
                continue

            for window in self.ma_windows:
                col_name = f"{metric}_ma_{window}"

                # Calculate moving average with min_periods=1
                # This ensures we get values even for the first few rows
                df_ma[col_name] = df_ma[metric].rolling(
                    window=window,
                    min_periods=1
                ).mean()

                logger.debug(f"Calculated {col_name}")

        return df_ma

    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived health metrics

        Formulas from frontend/analytics.py:
        - BMI (line 30): weight_kg / (height_m ** 2)
        - BMR (line 50): Mifflin-St Jeor equation
        - TDEE (line 83): BMR × activity_factor
        - Energy Balance (line 99): calories_consumed - calories_burned

        Args:
            df: DataFrame with health data

        Returns:
            DataFrame with derived feature columns added
        """
        df_derived = df.copy()

        # Get personal info
        height_cm = self.personal_info.get('height_cm', 175)
        height_m = height_cm / 100
        age = self.personal_info.get('age', 30)
        sex = self.personal_info.get('sex', 'Male')
        activity_factor = self.personal_info.get('activity_factor', 1.55)

        # 1. BMI (Body Mass Index)
        # Formula: BMI = weight(kg) / height(m)²
        if 'weight' in df_derived.columns:
            df_derived['bmi'] = df_derived['weight'] / (height_m ** 2)
            logger.debug(f"Calculated BMI (height={height_m}m)")

        # 2. BMR (Basal Metabolic Rate) - Mifflin-St Jeor equation
        # Men: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age + 5
        # Women: BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age - 161
        if 'weight' in df_derived.columns:
            base_bmr = 10 * df_derived['weight'] + 6.25 * height_cm - 5 * age

            if sex.lower() in ['male', 'm', 'man']:
                df_derived['bmr'] = base_bmr + 5
            elif sex.lower() in ['female', 'f', 'woman']:
                df_derived['bmr'] = base_bmr - 161
            else:
                df_derived['bmr'] = base_bmr  # Default to neutral

            logger.debug(f"Calculated BMR (age={age}, sex={sex})")

        # 3. TDEE (Total Daily Energy Expenditure)
        # Formula: TDEE = BMR × activity_factor
        if 'bmr' in df_derived.columns:
            df_derived['tdee'] = df_derived['bmr'] * activity_factor
            logger.debug(f"Calculated TDEE (activity_factor={activity_factor})")

        # 4. Energy Balance
        # Formula: energy_balance = calories_consumed - calories_burned
        if 'calories_consumed' in df_derived.columns and 'calories_burned' in df_derived.columns:
            df_derived['energy_balance'] = df_derived['calories_consumed'] - df_derived['calories_burned']
            logger.debug("Calculated energy balance")

        # 5. Sleep hours (convenience conversion)
        if 'sleep_min' in df_derived.columns:
            df_derived['sleep_hours'] = df_derived['sleep_min'] / 60
            logger.debug("Calculated sleep hours")

        # 6. Workout hours (convenience conversion)
        if 'workout_duration_min_tot' in df_derived.columns:
            df_derived['workout_hours'] = df_derived['workout_duration_min_tot'] / 60
            logger.debug("Calculated workout hours")

        return df_derived

    def load(self, df: pd.DataFrame) -> None:
        """
        Load: Save enriched data to CSV

        Args:
            df: DataFrame with enriched features
        """
        logger.info(f"Loading enriched data to {self.output_csv}")

        # Create output directory if it doesn't exist
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(self.output_csv, index=False, date_format='%d/%m/%Y')

        logger.info(f"Successfully saved {len(df)} records with {len(df.columns)} columns")
        logger.info(f"Output file: {self.output_csv}")

    def run(self) -> pd.DataFrame:
        """
        Run the complete ETL pipeline

        Returns:
            DataFrame with enriched data
        """
        logger.info("=" * 60)
        logger.info("Starting Health Data ETL Pipeline")
        logger.info("=" * 60)

        try:
            # Extract
            df_raw = self.extract()

            # Transform
            df_enriched = self.transform(df_raw)

            # Load
            self.load(df_enriched)

            logger.info("=" * 60)
            logger.info("ETL Pipeline Completed Successfully")
            logger.info("=" * 60)

            # Print summary
            self._print_summary(df_raw, df_enriched)

            return df_enriched

        except Exception as e:
            logger.error(f"ETL Pipeline failed: {str(e)}", exc_info=True)
            raise

    def _print_summary(self, df_raw: pd.DataFrame, df_enriched: pd.DataFrame) -> None:
        """Print ETL summary statistics"""

        print("\n" + "=" * 60)
        print("ETL SUMMARY")
        print("=" * 60)
        print(f"Records processed: {len(df_enriched)}")
        print(f"Date range: {df_enriched['date'].min().strftime('%d/%m/%Y')} to {df_enriched['date'].max().strftime('%d/%m/%Y')}")
        print(f"Original columns: {len(df_raw.columns)}")
        print(f"Enriched columns: {len(df_enriched.columns)}")
        print(f"New features added: {len(df_enriched.columns) - len(df_raw.columns)}")

        print("\nNew Features:")
        new_cols = [col for col in df_enriched.columns if col not in df_raw.columns]
        for col in sorted(new_cols):
            print(f"  - {col}")

        print("\nSample Statistics:")
        if 'weight_ma_7' in df_enriched.columns:
            print(f"  Weight (7-day avg): {df_enriched['weight_ma_7'].iloc[-1]:.2f} kg")
        if 'bmi' in df_enriched.columns:
            print(f"  BMI: {df_enriched['bmi'].iloc[-1]:.2f}")
        if 'energy_balance' in df_enriched.columns:
            avg_balance = df_enriched['energy_balance'].mean()
            print(f"  Avg Energy Balance: {avg_balance:+.0f} kcal/day")

        print("=" * 60 + "\n")


def run_etl(input_csv: Optional[str] = None,
            output_csv: Optional[str] = None,
            personal_info_json: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to run ETL pipeline

    Args:
        input_csv: Path to input CSV (default: data/health_data.csv)
        output_csv: Path to output CSV (default: data/health_data_enriched.csv)
        personal_info_json: Path to personal info JSON (default: data/personal_info.json)

    Returns:
        DataFrame with enriched data
    """
    kwargs = {}
    if input_csv:
        kwargs['input_csv'] = input_csv
    if output_csv:
        kwargs['output_csv'] = output_csv
    if personal_info_json:
        kwargs['personal_info_json'] = personal_info_json

    etl = HealthDataETL(**kwargs)
    return etl.run()


if __name__ == "__main__":
    """Run ETL when script is executed directly"""

    # Parse command line arguments if needed
    import argparse

    parser = argparse.ArgumentParser(
        description='ETL pipeline for health dashboard data transformation'
    )
    parser.add_argument(
        '--input',
        default='data/health_data.csv',
        help='Path to input CSV file (default: data/health_data.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/health_data_enriched.csv',
        help='Path to output CSV file (default: data/health_data_enriched.csv)'
    )
    parser.add_argument(
        '--personal-info',
        default='data/personal_info.json',
        help='Path to personal info JSON (default: data/personal_info.json)'
    )

    args = parser.parse_args()

    # Run ETL
    run_etl(
        input_csv=args.input,
        output_csv=args.output,
        personal_info_json=args.personal_info
    )
