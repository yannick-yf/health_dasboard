"""
Example: Using Pre-computed Features from ETL Pipeline

This script demonstrates how to load and use the enriched health data
that contains pre-computed moving averages and derived metrics.

Run this after executing the ETL pipeline:
    poetry run python etl/transform_health_data.py
    poetry run python examples/use_enriched_data.py
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


def load_enriched_data():
    """Load the enriched health data with pre-computed features"""
    data_path = Path(__file__).parent.parent / 'data' / 'health_data_enriched.csv'

    df = pd.read_csv(
        data_path,
        parse_dates=['date'],
        dayfirst=True
    )

    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Date range: {df['date'].min().strftime('%d/%m/%Y')} to {df['date'].max().strftime('%d/%m/%Y')}")

    return df


def plot_weight_with_moving_averages(df):
    """
    Create a weight trend plot using pre-computed moving averages

    This replaces the need to calculate moving averages on the fly
    """
    fig = go.Figure()

    # Actual weight
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['weight'],
        mode='markers+lines',
        name='Weight',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    # Pre-computed 3-day moving average
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['weight_ma_3'],
        mode='lines',
        name='3-day Average',
        line=dict(dash='dash', width=2, color='#ff7f0e'),
        opacity=0.8
    ))

    # Pre-computed 5-day moving average
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['weight_ma_5'],
        mode='lines',
        name='5-day Average',
        line=dict(dash='dash', width=2, color='#2ca02c'),
        opacity=0.7
    ))

    # Pre-computed 7-day moving average
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['weight_ma_7'],
        mode='lines',
        name='7-day Average',
        line=dict(dash='dot', width=2, color='#d62728'),
        opacity=0.6
    ))

    fig.update_layout(
        title='Weight Trend with Pre-computed Moving Averages',
        xaxis_title='Date',
        yaxis_title='Weight (kg)',
        hovermode='x unified',
        template='plotly_white'
    )

    fig.show()
    print("Weight plot created successfully!")


def plot_energy_balance(df):
    """
    Plot energy balance using pre-computed values

    No need to calculate calories_consumed - calories_burned
    """
    fig = go.Figure()

    # Energy balance bars
    colors = ['red' if x < 0 else 'green' for x in df['energy_balance']]

    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['energy_balance'],
        name='Energy Balance',
        marker_color=colors,
        hovertemplate='Date: %{x}<br>Balance: %{y:+.0f} kcal<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title='Daily Energy Balance (Pre-computed)',
        xaxis_title='Date',
        yaxis_title='Energy Balance (kcal)',
        hovermode='x unified',
        template='plotly_white'
    )

    fig.show()
    print("Energy balance plot created successfully!")


def show_derived_metrics(df):
    """Display pre-computed derived metrics"""

    latest = df.iloc[-1]

    print("\n" + "="*60)
    print("LATEST HEALTH METRICS (Pre-computed by ETL)")
    print("="*60)
    print(f"Date: {latest['date'].strftime('%d/%m/%Y')}")
    print()
    print(f"Weight: {latest['weight']:.2f} kg")
    print(f"  - 3-day avg: {latest['weight_ma_3']:.2f} kg")
    print(f"  - 5-day avg: {latest['weight_ma_5']:.2f} kg")
    print(f"  - 7-day avg: {latest['weight_ma_7']:.2f} kg")
    print()
    print(f"Steps: {latest['steps']:.0f}")
    print(f"  - 3-day avg: {latest['steps_ma_3']:.0f}")
    print(f"  - 5-day avg: {latest['steps_ma_5']:.0f}")
    print(f"  - 7-day avg: {latest['steps_ma_7']:.0f}")
    print()
    print(f"BMI: {latest['bmi']:.2f}")
    print(f"BMR: {latest['bmr']:.0f} kcal/day")
    print(f"TDEE: {latest['tdee']:.0f} kcal/day")
    print(f"Energy Balance: {latest['energy_balance']:+.0f} kcal")
    print()
    print(f"Sleep: {latest['sleep_hours']:.2f} hours ({latest['sleep_min']:.0f} min)")
    print(f"  - 7-day avg: {latest['sleep_min_ma_7']/60:.2f} hours")
    print()
    print(f"Workout: {latest['workout_hours']:.2f} hours ({latest['workout_duration_min_tot']:.0f} min)")
    print(f"  - 7-day avg: {latest['workout_duration_min_tot_ma_7']/60:.2f} hours")
    print("="*60)
    print()


def compare_performance():
    """
    Compare the performance of using pre-computed vs on-the-fly calculations
    """
    import time

    df = load_enriched_data()

    print("\nPerformance Comparison:")
    print("-" * 60)

    # Method 1: Using pre-computed moving average
    start = time.time()
    weight_ma7_precomputed = df['weight_ma_7'].values
    time_precomputed = time.time() - start
    print(f"Pre-computed (ETL):     {time_precomputed*1000:.4f} ms")

    # Method 2: Computing on the fly
    start = time.time()
    weight_ma7_computed = df['weight'].rolling(window=7, min_periods=1).mean().values
    time_computed = time.time() - start
    print(f"On-the-fly calculation: {time_computed*1000:.4f} ms")

    speedup = time_computed / time_precomputed
    print(f"Speedup: {speedup:.1f}x faster")
    print("-" * 60)

    # Verify they're the same
    import numpy as np
    if np.allclose(weight_ma7_precomputed, weight_ma7_computed):
        print("✓ Results verified: Pre-computed matches on-the-fly calculation")
    else:
        print("✗ Warning: Results differ!")


def calculate_weekly_summary(df):
    """
    Calculate weekly summary using pre-computed features
    """
    # Add week column
    df['week'] = df['date'].dt.to_period('W')

    weekly = df.groupby('week').agg({
        'weight_ma_7': 'last',  # Use last 7-day average of the week
        'steps_ma_7': 'last',
        'calories_consumed_ma_7': 'last',
        'calories_burned_ma_7': 'last',
        'energy_balance': 'sum',  # Total weekly balance
        'sleep_hours': 'mean',
        'workout_hours': 'sum',
        'bmi': 'last',
        'tdee': 'last'
    }).round(2)

    print("\n" + "="*60)
    print("WEEKLY SUMMARY (Using Pre-computed Features)")
    print("="*60)
    print(weekly.to_string())
    print("="*60)
    print()


def main():
    """Main example function"""

    print("="*60)
    print("Health Dashboard - Using Enriched Data Example")
    print("="*60)
    print()

    # Load enriched data
    df = load_enriched_data()

    # Show derived metrics
    show_derived_metrics(df)

    # Calculate weekly summary
    calculate_weekly_summary(df)

    # Performance comparison
    compare_performance()

    # Create visualizations
    print("\nGenerating visualizations...")
    print("(Close the plot windows to continue)")

    try:
        plot_weight_with_moving_averages(df)
        plot_energy_balance(df)
    except Exception as e:
        print(f"Note: Could not display plots (expected in non-GUI environment): {e}")

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
