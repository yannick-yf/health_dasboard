"""
Weekly report generator — produces a structured data dict for any past week.

All calculations are scoped to data available up to week_end, ensuring historical
reports reflect what would have been known at that time.
"""

from datetime import date

import pandas as pd

from utils.metrics_helpers import compute_intake_targets
from utils.visualization_helpers import (
    create_weight_chart,
    create_energy_balance_chart,
    create_sleep_chart,
    create_steps_workout_chart,
)
from utils.bulk_constants import (
    BULK_START,
    GAIN_MIN,
    GAIN_MAX,
    GAIN_WARN_HIGH,
    SURPLUS_LO,
    SURPLUS_HI,
    SLEEP_MIN_H,
    NEAT_HIGH,
    SHORT_SESSION_WARN_MIN,
    classify_status,
    get_mesocycle_context,
)


def generate_weekly_report_data(df: pd.DataFrame, week_start: date) -> dict:
    """
    Generate all data needed to render a weekly health report.

    Parameters
    ----------
    df         : full health DataFrame with a parsed 'date' column (datetime64)
    week_start : first day of the target week (Monday)

    Returns
    -------
    dict with sections: meta, kpis, targets, signals, figures
    """
    week_start_ts = pd.Timestamp(week_start)
    week_end_ts   = week_start_ts + pd.Timedelta(days=6)

    # Scope: everything known up to week_end (historical accuracy)
    df_context = df[df["date"] <= week_end_ts].copy().sort_values("date").reset_index(drop=True)

    # Week slice: only the 7 target days
    df_week = df[
        (df["date"] >= week_start_ts) & (df["date"] <= week_end_ts)
    ].copy().sort_values("date").reset_index(drop=True)

    days_available = len(df_week)
    context_days   = len(df_context)

    # ── Bulk week number + mesocycle context ─────────────────────────────────
    bulk_week_num = (
        max(1, int((week_start_ts - BULK_START).days / 7) + 1)
        if week_start_ts >= BULK_START
        else None
    )
    mesocycle = get_mesocycle_context(week_end_ts)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    avg_weight = df_week["weight"].mean() if df_week["weight"].notna().any() else None

    prior_start = week_start_ts - pd.Timedelta(days=7)
    prior_end   = week_start_ts - pd.Timedelta(days=1)
    df_prior    = df[(df["date"] >= prior_start) & (df["date"] <= prior_end)]
    prior_avg_weight = df_prior["weight"].mean() if not df_prior.empty and df_prior["weight"].notna().any() else None

    weight_delta = (
        (avg_weight - prior_avg_weight)
        if (avg_weight is not None and not pd.isna(avg_weight) and
            prior_avg_weight is not None and not pd.isna(prior_avg_weight))
        else None
    )

    avg_sleep_h = (
        df_week["sleep_min"].mean() / 60
        if df_week["sleep_min"].notna().any()
        else None
    )
    avg_steps = df_week["steps"].mean() if df_week["steps"].notna().any() else None

    df_week = df_week.copy()
    df_week["surplus"] = df_week["calories_consumed"] - df_week["calories_burned"]
    avg_surplus = df_week["surplus"].mean() if df_week["surplus"].notna().any() else None

    workout_days = (
        int((df_week["workout_duration_min_tot"] > 30).sum())
        if df_week["workout_duration_min_tot"].notna().any()
        else None
    )

    # ── 4-week gain rate + 7-day MA delta ────────────────────────────────────
    df_ma = df_context.copy()
    df_ma["weight_7d_ma"] = df_ma["weight"].rolling(7, min_periods=3).mean()
    df_ma["gain_rate_4w"] = (df_ma["weight_7d_ma"] - df_ma["weight_7d_ma"].shift(28)) / 4.0

    rate_series = df_ma["gain_rate_4w"].dropna()
    gain_rate   = float(rate_series.iloc[-1]) if len(rate_series) else None

    # 7-day MA at end of this week vs end of prior week — the "pulse" signal
    ma_at_week_end  = df_ma.loc[df_ma["date"] <= week_end_ts,  "weight_7d_ma"].dropna()
    ma_at_prior_end = df_ma.loc[df_ma["date"] <  week_start_ts, "weight_7d_ma"].dropna()
    weekly_ma_delta = (
        float(ma_at_week_end.iloc[-1]) - float(ma_at_prior_end.iloc[-1])
        if len(ma_at_week_end) and len(ma_at_prior_end)
        else None
    )

    status_label, status_color = classify_status(gain_rate)

    # ── Calorie targets (data scoped to context = up to week_end) ────────────
    targets = compute_intake_targets(
        df_context,
        bulk_start_ts=BULK_START,
        reference_date=week_end_ts,
    )

    # ── Data confidence note ──────────────────────────────────────────────────
    data_note = None
    if context_days < 14:
        data_note = (
            f"{context_days} days of data available — empirical TDEE estimate "
            "may be less reliable (ideal: 28+ days)"
        )
    elif days_available < 4:
        data_note = f"Only {days_available}/7 days have data for this week"

    # ── Signals ───────────────────────────────────────────────────────────────
    signals = _compute_signals(df_week, gain_rate, avg_surplus, avg_sleep_h)

    # ── Charts ────────────────────────────────────────────────────────────────
    # Weight: 4-week window ending at week_end
    weight_window_start = week_end_ts - pd.Timedelta(days=27)
    df_weight_window    = df_context[df_context["date"] >= weight_window_start].copy()
    fig_weight  = create_weight_chart(df_weight_window) if not df_weight_window.empty else None
    fig_energy  = create_energy_balance_chart(df_week)  if days_available > 0 else None
    fig_sleep   = create_sleep_chart(df_week)           if days_available > 0 else None
    fig_steps   = create_steps_workout_chart(df_week)   if days_available > 0 else None

    return {
        # Meta
        "week_start":     week_start_ts,
        "week_end":       week_end_ts,
        "bulk_week_num":  bulk_week_num,
        "mesocycle":      mesocycle,
        "days_available": days_available,
        "context_days":   context_days,
        "data_note":      data_note,
        # Status
        "status_label":     status_label,
        "status_color":     status_color,
        # KPIs
        "avg_weight":       avg_weight,
        "weight_delta":     weight_delta,
        "weekly_ma_delta":  weekly_ma_delta,
        "avg_sleep_h":   avg_sleep_h,
        "avg_steps":     avg_steps,
        "avg_surplus":   avg_surplus,
        "workout_days":  workout_days,
        "gain_rate":     gain_rate,
        # Calorie targets dict (from compute_intake_targets)
        "targets":       targets,
        # Rule-based signals
        "signals":       signals,
        # Plotly figures
        "fig_weight":    fig_weight,
        "fig_energy":    fig_energy,
        "fig_sleep":     fig_sleep,
        "fig_steps":     fig_steps,
    }


def _compute_signals(df_week: pd.DataFrame, gain_rate, avg_surplus, avg_sleep_h) -> list[dict]:
    """Return a list of {type, msg} signal dicts. type ∈ {'ok', 'warn', 'info'}."""
    signals = []

    # Gain rate
    if gain_rate is not None and not pd.isna(gain_rate):
        if gain_rate > GAIN_WARN_HIGH:
            signals.append({"type": "warn", "msg": f"Gaining too fast ({gain_rate:+.2f} kg/wk > {GAIN_WARN_HIGH} limit)"})
        elif gain_rate > GAIN_MAX:
            signals.append({"type": "warn", "msg": f"Slightly fast ({gain_rate:+.2f} kg/wk, target {GAIN_MIN}–{GAIN_MAX})"})
        elif gain_rate >= GAIN_MIN:
            signals.append({"type": "ok",   "msg": f"Gain rate on target ({gain_rate:+.2f} kg/wk)"})
        elif gain_rate >= 0:
            signals.append({"type": "warn", "msg": f"Gaining slowly ({gain_rate:+.2f} kg/wk, target {GAIN_MIN}–{GAIN_MAX})"})
        else:
            signals.append({"type": "warn", "msg": f"Weight trending down ({gain_rate:+.2f} kg/wk)"})

    # Average surplus
    if avg_surplus is not None and not pd.isna(avg_surplus):
        if avg_surplus > SURPLUS_HI + 200:
            signals.append({"type": "warn", "msg": f"Surplus too high ({avg_surplus:.0f} kcal/day avg, target {SURPLUS_LO}–{SURPLUS_HI})"})
        elif avg_surplus < SURPLUS_LO:
            signals.append({"type": "warn", "msg": f"Surplus too low ({avg_surplus:.0f} kcal/day avg, target {SURPLUS_LO}–{SURPLUS_HI})"})
        else:
            signals.append({"type": "ok",   "msg": f"Surplus in target range ({avg_surplus:.0f} kcal/day avg)"})

    # Days with surplus > 300 kcal
    if "surplus" in df_week.columns:
        high_days = int((df_week["surplus"] > SURPLUS_HI).sum())
        if high_days >= 3:
            signals.append({"type": "warn", "msg": f"Surplus above {SURPLUS_HI} kcal on {high_days} days this week"})

    # Sleep
    if avg_sleep_h is not None and not pd.isna(avg_sleep_h):
        if "sleep_min" in df_week.columns:
            below_nights = int((df_week["sleep_min"].dropna() / 60 < SLEEP_MIN_H).sum())
            if below_nights >= 4:
                signals.append({"type": "warn", "msg": f"{below_nights}/7 nights below {SLEEP_MIN_H:.0f}h sleep"})
            elif avg_sleep_h >= 7.5:
                signals.append({"type": "ok",   "msg": f"Good sleep average ({avg_sleep_h:.1f}h/night)"})
            else:
                signals.append({"type": "info", "msg": f"Sleep avg {avg_sleep_h:.1f}h (target 7.5h)"})

    # NEAT
    if "steps" in df_week.columns and df_week["steps"].notna().any():
        high_neat_days = int((df_week["steps"] >= NEAT_HIGH).sum())
        if high_neat_days >= 4:
            signals.append({
                "type": "info",
                "msg": f"High NEAT: {high_neat_days}/7 days above {NEAT_HIGH:,} steps — may offset surplus",
            })

    # Outlier calorie day (> median + 800 kcal)
    cal_col = df_week["calories_consumed"].dropna()
    if len(cal_col) >= 3:
        median_cal = cal_col.median()
        outlier_rows = df_week[df_week["calories_consumed"] > median_cal + 800]
        for _, row in outlier_rows.iterrows():
            delta = int(row["calories_consumed"] - median_cal)
            day_str = pd.Timestamp(row["date"]).strftime("%b %d")
            signals.append({
                "type": "info",
                "msg": (
                    f"{day_str}: {int(row['calories_consumed']):,} kcal consumed "
                    f"(+{delta} above median) — may inflate weekly average"
                ),
            })

    # Short training session (> 0 but < SHORT_SESSION_WARN_MIN minutes)
    if "workout_duration_min_tot" in df_week.columns:
        short_sessions = df_week[
            (df_week["workout_duration_min_tot"] > 0) &
            (df_week["workout_duration_min_tot"] < SHORT_SESSION_WARN_MIN)
        ]
        for _, row in short_sessions.iterrows():
            day_str = pd.Timestamp(row["date"]).strftime("%b %d")
            signals.append({
                "type": "warn",
                "msg": (
                    f"{day_str}: {int(row['workout_duration_min_tot'])}min workout — "
                    "unusually short, verify session was complete"
                ),
            })

    return signals
