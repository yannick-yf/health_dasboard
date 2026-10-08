"""Weekly WADP review, scoped to a selected Monday–Sunday week."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from utils.wadp import BMR_BASE_KCAL, CURRENT_DEFICIT_KCAL

TRAINING_CSV = Path(__file__).resolve().parents[2] / "data" / "training_log.csv"


def _mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _window_mean(df, column, end, days):
    values = df.loc[(df["date"] > end - pd.Timedelta(days=days)) &
                    (df["date"] <= end), column]
    return _mean(values)


def _week_deficit(df):
    move = pd.to_numeric(df.get("move_kcal"), errors="coerce")
    intake = pd.to_numeric(df.get("calories_consumed"), errors="coerce")
    valid = move.notna() & (move > 0) & intake.notna() & (intake > 0)
    return (BMR_BASE_KCAL + move[valid] - intake[valid]), int(valid.sum())


def _session_count(start, end, training_csv):
    if not training_csv.exists():
        return None
    log = pd.read_csv(training_csv, usecols=["date", "session"], dtype=str)
    log["date"] = pd.to_datetime(log["date"], format="%d/%m/%Y", errors="coerce")
    week = log[(log["date"] >= start) & (log["date"] <= end)]
    return len(week.drop_duplicates(["date", "session"]))


def _trend_chart(context):
    recent = context.tail(28)
    fig = go.Figure()
    for column, label, color, axis in [("weight", "Weight", "#a78bfa", "y"),
                                        ("waist_cm", "Waist", "#14b8a6", "y2")]:
        values = pd.to_numeric(recent[column], errors="coerce")
        fig.add_trace(go.Scatter(x=recent["date"], y=values, name=label,
                                 mode="lines+markers", line_color=color, yaxis=axis))
    fig.update_layout(title="Weight and waist (last 28 records)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#f0f2f6", height=370,
                      yaxis=dict(title="Weight (kg)"),
                      yaxis2=dict(title="Waist (cm)", overlaying="y", side="right"))
    return fig


def _deficit_chart(week):
    deficit, _ = _week_deficit(week)
    valid = week.loc[deficit.index]
    fig = go.Figure(go.Bar(x=valid["date"], y=deficit,
                           marker_color=["#10b981" if v >= 0 else "#f97316" for v in deficit],
                           name="Nominal WADP deficit"))
    fig.add_hline(y=CURRENT_DEFICIT_KCAL, line_dash="dot", line_color="#a78bfa",
                  annotation_text=f"Current target {CURRENT_DEFICIT_KCAL}")
    fig.update_layout(title="Observed WADP deficit (kcal/day)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#f0f2f6", height=350)
    return fig


def generate_weekly_report_data(df: pd.DataFrame, week_start, training_csv=TRAINING_CSV) -> dict:
    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    context = df[df["date"] <= end].copy().sort_values("date")
    week = context[context["date"] >= start]
    prior = context[(context["date"] >= start - pd.Timedelta(days=7)) &
                    (context["date"] < start)]
    deficit, deficit_days = _week_deficit(week)
    prior_deficit, prior_deficit_days = _week_deficit(prior)
    avg_deficit = _mean(deficit)
    prior_avg_deficit = _mean(prior_deficit)
    weight_ma7 = _window_mean(context, "weight", end, 7)
    prior_weight_ma7 = _window_mean(context, "weight", start - pd.Timedelta(days=1), 7)
    waist_ma7 = _window_mean(context, "waist_cm", end, 7)
    prior_waist_ma7 = _window_mean(context, "waist_cm", start - pd.Timedelta(days=1), 7)
    waist_ma14 = _window_mean(context, "waist_cm", end, 14)
    prior_waist_ma14 = _window_mean(context, "waist_cm", start - pd.Timedelta(days=1), 14)
    signals = []
    if deficit_days < len(week):
        signals.append(f"WADP average uses {deficit_days}/{len(week)} days with both Move and intake.")
    if len(week) < 7:
        signals.append(f"Only {len(week)}/7 days have health records.")
    if not week.empty and _mean(week["waist_cm"]) is not None:
        signals.append("Travel measurements carried forward in the CSV cannot be identified automatically; check original readings before changing the deficit.")
    return {
        "week_start": start, "week_end": end, "days_available": len(week),
        "deficit_days": deficit_days, "avg_deficit": avg_deficit,
        "prior_deficit_days": prior_deficit_days, "prior_avg_deficit": prior_avg_deficit,
        "weight_ma7": weight_ma7, "prior_weight_ma7": prior_weight_ma7,
        "waist_ma7": waist_ma7, "prior_waist_ma7": prior_waist_ma7,
        "waist_ma14": waist_ma14, "prior_waist_ma14": prior_waist_ma14,
        "sessions": _session_count(start, end, Path(training_csv)),
        "prior_sessions": _session_count(start - pd.Timedelta(days=7), start - pd.Timedelta(days=1), Path(training_csv)),
        "avg_sleep_h": _mean(week["sleep_min"]) / 60 if _mean(week["sleep_min"]) is not None else None,
        "avg_steps": _mean(week["steps"]), "signals": signals,
        "fig_trend": _trend_chart(context), "fig_deficit": _deficit_chart(week),
    }
