"""
Helper functions for calculating and formatting health metrics
"""

import pandas as pd
from scipy import stats

# ── Calorie target constants ───────────────────────────────────────────────────
_SURPLUS_TARGET      = 250   # kcal/day — target anabolic surplus for clean bulk
_KCAL_PER_KG_TISSUE  = 7700  # energy equivalent of 1 kg mixed lean+fat tissue
_STALL_THRESHOLD     = 0.06  # kg/wk — below this, treat weight as empirically flat
_TRAIN_OFFSET        = 100   # kcal above daily avg on training days
_REST_OFFSET         = 100   # kcal below daily avg on rest days


def _round50(x: float) -> int:
    """Round to nearest 50 kcal for clean target numbers."""
    return int(round(x / 50) * 50)


def compute_intake_targets(df, bulk_start_ts=None, reference_date=None) -> dict | None:
    """
    Compute recommended daily calorie targets based on the last 28 days of data.

    Algorithm
    ---------
    1. Compute the 4-week gain rate from the 7-day rolling weight average.
    2. Back-calculate empirical TDEE:
       - Stalling (|rate| < 0.06 kg/wk): consumed ≈ TDEE (weight flat → no surplus stored)
       - Gaining / losing: TDEE = consumed − (rate × 7700 kcal/kg ÷ 7 days)
    3. Target = empirical TDEE + 250 kcal surplus.
    4. Periodize: training days get +100 kcal above target, rest days −100.

    Parameters
    ----------
    df             : full health DataFrame (date, weight, calories_consumed, calories_burned)
    bulk_start_ts  : optional pd.Timestamp to restrict to bulk phase only
    reference_date : optional pd.Timestamp — when set, only data up to this date is used.
                     Enables historical reports that reflect what was known at that time.

    Returns
    -------
    dict with keys: empirical_tdee, daily_avg_target, training_day, rest_day,
                    gain_rate_4w, avg_consumed, avg_burned, status, change_from_current
    Returns None if there is insufficient data.
    """
    work_df = df.copy()
    if bulk_start_ts is not None:
        work_df = work_df[work_df['date'] >= bulk_start_ts]
    if reference_date is not None:
        work_df = work_df[work_df['date'] <= reference_date]
    recent = work_df.sort_values('date').tail(28)

    if len(recent) < 7:
        return None

    avg_consumed = recent['calories_consumed'].dropna().mean()
    avg_burned   = recent['calories_burned'].dropna().mean()

    if pd.isna(avg_consumed):
        return None

    # ── Gain rate from 7d rolling average ─────────────────────────────────────
    weight_ma = recent['weight'].rolling(7, min_periods=3).mean().dropna()
    if len(weight_ma) < 4:
        rate = None
    else:
        elapsed_weeks = (recent['date'].max() - recent['date'].min()).days / 7
        rate = (weight_ma.iloc[-1] - weight_ma.iloc[0]) / elapsed_weeks if elapsed_weeks > 0 else None

    # ── Empirical TDEE ─────────────────────────────────────────────────────────
    if rate is None or pd.isna(rate):
        empirical_tdee = avg_burned if not pd.isna(avg_burned) else avg_consumed
        status = "unknown"
    elif abs(rate) < _STALL_THRESHOLD:
        # Weight is flat → whatever was consumed is maintenance
        empirical_tdee = avg_consumed
        status = "stalling"
    else:
        # Portion of intake went to (or came from) tissue
        empirical_tdee = avg_consumed - (rate * _KCAL_PER_KG_TISSUE / 7)
        status = "gaining" if rate > 0 else "losing"

    daily_avg_target = empirical_tdee + _SURPLUS_TARGET

    return {
        "empirical_tdee":     round(empirical_tdee),
        "daily_avg_target":   round(daily_avg_target),
        "training_day":       _round50(daily_avg_target + _TRAIN_OFFSET),
        "rest_day":           _round50(daily_avg_target - _REST_OFFSET),
        "gain_rate_4w":       round(rate, 3) if rate is not None else None,
        "avg_consumed":       round(avg_consumed),
        "avg_burned":         round(avg_burned) if not pd.isna(avg_burned) else None,
        "status":             status,
        "change_from_current": round(daily_avg_target - avg_consumed),
    }


def calculate_weight_trend(df, days):
    """Calculate weight trend percentage over the last N days"""
    if len(df) < 2 or df['weight'].isna().all():
        return None

    weight_data = df[df['weight'].notna()].sort_values('date')
    if len(weight_data) < 2:
        return None

    recent_data = weight_data.tail(days) if len(weight_data) >= days else weight_data
    if len(recent_data) < 2:
        return None

    first_weight = recent_data['weight'].iloc[0]
    last_weight  = recent_data['weight'].iloc[-1]

    if first_weight == 0:
        return None

    return ((last_weight - first_weight) / first_weight) * 100


def calculate_weight_rate_kgperweek(df, days=7):
    """Calculate weight gain/loss rate in kg/week using linear regression over last N days."""
    weight_data = df[df['weight'].notna()].sort_values('date')
    if len(weight_data) < 2:
        return None

    recent = weight_data.tail(days) if len(weight_data) >= days else weight_data
    if len(recent) < 2:
        return None

    x = (recent['date'] - recent['date'].min()).dt.days.values
    y = recent['weight'].values
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope * 7  # kg per week


def format_sleep_time(minutes):
    """Convert minutes to hours and minutes format"""
    if pd.isna(minutes):
        return "N/A"
    hours = int(minutes // 60)
    mins  = int(minutes % 60)
    return f"{hours}h {mins}m"
