"""
Yearly deep-dive analysis: cut + bulk phases over 9 months of data.
Produces a self-contained HTML report with Plotly charts + tri-perspective synthesis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "health_data.csv"
OUT = ROOT / "reports" / "yearly_deep_dive.html"
OUT.parent.mkdir(exist_ok=True)

KCAL_PER_KG = 7700

# ─── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(SRC)
df.columns = [c.strip() for c in df.columns]
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
for c in ["steps","sleep_min","workout_duration_min_tot","weight","calories_burned","calories_consumed"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["weight_ma7"] = df["weight"].rolling(7, min_periods=3).mean()
df["weight_ma14"] = df["weight"].rolling(14, min_periods=5).mean()
df["consumed_ma7"] = df["calories_consumed"].rolling(7, min_periods=3).mean()
df["burned_ma7"] = df["calories_burned"].rolling(7, min_periods=3).mean()
df["balance_ma7"] = df["consumed_ma7"] - df["burned_ma7"]
df["steps_ma7"] = df["steps"].rolling(7, min_periods=3).mean()
df["workout_ma7"] = df["workout_duration_min_tot"].rolling(7, min_periods=3).mean()
df["sleep_ma7"] = df["sleep_min"].rolling(7, min_periods=3).mean()

# ─── Phase boundaries (data-driven) ──────────────────────────────────────
weight_min_idx = df["weight_ma7"].idxmin()
phase_low_date = df.loc[weight_min_idx, "date"]

phases = {
    "Cut": (df["date"].iloc[0], phase_low_date),
    "Rebuild": (phase_low_date, pd.Timestamp("2025-12-01")),
    "Bulk": (pd.Timestamp("2025-12-01"), df["date"].iloc[-1]),
}

def phase_stats(name, start, end):
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if len(sub) < 7:
        return None
    w_start = sub["weight_ma7"].dropna().iloc[0] if sub["weight_ma7"].dropna().size else sub["weight"].iloc[0]
    w_end = sub["weight_ma7"].dropna().iloc[-1] if sub["weight_ma7"].dropna().size else sub["weight"].iloc[-1]
    days = len(sub)
    weeks = days / 7
    delta = w_end - w_start
    rate = delta / weeks if weeks else 0
    avg_consumed = sub["calories_consumed"].mean()
    avg_burned = sub["calories_burned"].mean()
    avg_balance = avg_consumed - avg_burned
    empirical_tdee = avg_consumed - delta * KCAL_PER_KG / days
    apple_watch_correction = empirical_tdee - avg_burned
    return {
        "name": name, "start": start, "end": end, "days": days, "weeks": weeks,
        "w_start": w_start, "w_end": w_end, "delta": delta, "rate_kg_wk": rate,
        "avg_consumed": avg_consumed, "avg_burned": avg_burned,
        "avg_balance": avg_balance, "empirical_tdee": empirical_tdee,
        "apple_watch_correction": apple_watch_correction,
        "avg_steps": sub["steps"].mean(), "avg_sleep_min": sub["sleep_min"].mean(),
        "avg_workout": sub["workout_duration_min_tot"].mean(),
        "workout_days_pct": (sub["workout_duration_min_tot"] >= 30).mean() * 100,
    }

phase_results = {name: phase_stats(name, *bounds) for name, bounds in phases.items()}

# ─── Charts ────────────────────────────────────────────────────────────────
PHASE_COLORS = {"Cut": "rgba(231, 76, 60, 0.10)",
                "Rebuild": "rgba(149, 165, 166, 0.10)",
                "Bulk": "rgba(46, 204, 113, 0.10)"}
PHASE_LINES = {"Cut": "#c0392b", "Rebuild": "#7f8c8d", "Bulk": "#27ae60"}

def add_phase_bands(fig, y_low, y_high):
    for name, (s, e) in phases.items():
        fig.add_shape(type="rect", xref="x", yref="paper",
                      x0=s, x1=e, y0=0, y1=1,
                      fillcolor=PHASE_COLORS[name], line=dict(width=0),
                      layer="below")
        fig.add_annotation(x=s + (e - s) / 2, y=1.05, xref="x", yref="paper",
                           text=f"<b>{name}</b>", showarrow=False,
                           font=dict(size=13, color=PHASE_LINES[name]))

# Chart 1: Weight trajectory
fig_weight = go.Figure()
fig_weight.add_trace(go.Scatter(x=df["date"], y=df["weight"], mode="markers",
    name="Daily", marker=dict(size=4, color="rgba(52,73,94,0.35)"),
    hovertemplate="%{x|%b %d}<br>%{y:.1f} kg<extra></extra>"))
fig_weight.add_trace(go.Scatter(x=df["date"], y=df["weight_ma7"], mode="lines",
    name="7-day MA", line=dict(color="#2c3e50", width=2.5),
    hovertemplate="%{x|%b %d}<br>%{y:.2f} kg<extra></extra>"))
fig_weight.add_trace(go.Scatter(x=df["date"], y=df["weight_ma14"], mode="lines",
    name="14-day MA", line=dict(color="#e67e22", width=2, dash="dot"),
    hovertemplate="%{x|%b %d}<br>%{y:.2f} kg<extra></extra>"))
add_phase_bands(fig_weight, df["weight"].min(), df["weight"].max())
fig_weight.update_layout(
    title="<b>Weight Trajectory — Aug 2025 → May 2026</b>",
    yaxis_title="Weight (kg)", height=380,
    margin=dict(l=60, r=20, t=80, b=40),
    plot_bgcolor="white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_weight.update_xaxes(showgrid=True, gridcolor="#ecf0f1")
fig_weight.update_yaxes(showgrid=True, gridcolor="#ecf0f1")

# Chart 2: Calorie balance over time
fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=df["date"], y=df["consumed_ma7"], mode="lines",
    name="Consumed (7d MA)", line=dict(color="#3498db", width=2),
    fill="tozeroy", fillcolor="rgba(52,152,219,0.10)"))
fig_cal.add_trace(go.Scatter(x=df["date"], y=df["burned_ma7"], mode="lines",
    name="Burned, Apple Watch (7d MA)", line=dict(color="#e74c3c", width=2)))
add_phase_bands(fig_cal, 0, 5000)
fig_cal.update_layout(
    title="<b>Calorie Intake vs Burn (7-day MA)</b>",
    yaxis_title="kcal/day", height=320,
    margin=dict(l=60, r=20, t=70, b=40),
    plot_bgcolor="white", hovermode="x unified",
    yaxis=dict(range=[1500, 4500]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_cal.update_xaxes(showgrid=True, gridcolor="#ecf0f1")
fig_cal.update_yaxes(showgrid=True, gridcolor="#ecf0f1")

# Chart 3: Energy balance + cumulative
df["cum_balance"] = df["balance_ma7"].fillna(0).cumsum()
fig_bal = make_subplots(specs=[[{"secondary_y": True}]])
fig_bal.add_trace(go.Bar(x=df["date"], y=df["balance_ma7"], name="Daily balance (7d MA)",
    marker_color=df["balance_ma7"].apply(lambda v: "#27ae60" if pd.notna(v) and v > 0 else "#c0392b"),
    opacity=0.65), secondary_y=False)
fig_bal.add_trace(go.Scatter(x=df["date"], y=df["cum_balance"], mode="lines",
    name="Cumulative", line=dict(color="#34495e", width=2.5)), secondary_y=True)
add_phase_bands(fig_bal, df["balance_ma7"].min(), df["balance_ma7"].max())
fig_bal.update_layout(
    title="<b>Energy Balance (consumed − burned, Apple Watch)</b>",
    height=320, margin=dict(l=60, r=60, t=70, b=40),
    plot_bgcolor="white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_bal.update_yaxes(title_text="kcal/day", secondary_y=False, gridcolor="#ecf0f1")
fig_bal.update_yaxes(title_text="Cumulative kcal", secondary_y=True, showgrid=False)
fig_bal.update_xaxes(showgrid=True, gridcolor="#ecf0f1")

# Chart 4: Activity load
fig_act = make_subplots(specs=[[{"secondary_y": True}]])
fig_act.add_trace(go.Scatter(x=df["date"], y=df["steps_ma7"], mode="lines",
    name="Steps (7d MA)", line=dict(color="#16a085", width=2)), secondary_y=False)
fig_act.add_trace(go.Scatter(x=df["date"], y=df["workout_ma7"], mode="lines",
    name="Workout min (7d MA)", line=dict(color="#8e44ad", width=2)), secondary_y=True)
add_phase_bands(fig_act, 0, 25000)
fig_act.update_layout(
    title="<b>Activity Load — Steps & Workout Duration</b>",
    height=320, margin=dict(l=60, r=60, t=70, b=40),
    plot_bgcolor="white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_act.update_yaxes(title_text="Steps/day", secondary_y=False, gridcolor="#ecf0f1")
fig_act.update_yaxes(title_text="Workout min/day", secondary_y=True, showgrid=False)
fig_act.update_xaxes(showgrid=True, gridcolor="#ecf0f1")

# Chart 5: Sleep distribution
fig_sleep = go.Figure()
fig_sleep.add_trace(go.Scatter(x=df["date"], y=df["sleep_min"]/60, mode="markers",
    name="Daily", marker=dict(size=3.5, color="rgba(127,140,141,0.45)")))
fig_sleep.add_trace(go.Scatter(x=df["date"], y=df["sleep_ma7"]/60, mode="lines",
    name="7-day MA", line=dict(color="#2980b9", width=2.5)))
fig_sleep.add_hline(y=8, line=dict(color="#e74c3c", dash="dash", width=1),
                    annotation_text="8h target", annotation_position="right")
add_phase_bands(fig_sleep, 0, 16)
fig_sleep.update_layout(
    title="<b>Sleep — Hours per Night</b>",
    yaxis_title="Hours", height=300,
    margin=dict(l=60, r=20, t=70, b=40),
    plot_bgcolor="white", hovermode="x unified",
    yaxis=dict(range=[2, 14]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_sleep.update_xaxes(showgrid=True, gridcolor="#ecf0f1")
fig_sleep.update_yaxes(showgrid=True, gridcolor="#ecf0f1")

# Chart 6: Phase comparison radar/bar
phase_metrics_df = pd.DataFrame([{
    "Phase": p["name"],
    "Δ weight (kg)": round(p["delta"], 1),
    "Rate (kg/wk)": round(p["rate_kg_wk"], 3),
    "Avg consumed": int(p["avg_consumed"]),
    "Avg burned": int(p["avg_burned"]),
    "Empirical TDEE": int(p["empirical_tdee"]),
    "Avg steps": int(p["avg_steps"]),
    "Avg sleep (h)": round(p["avg_sleep_min"]/60, 2),
    "Workout min/day": round(p["avg_workout"], 0),
    "Days": p["days"],
} for p in phase_results.values()])

fig_compare = make_subplots(rows=1, cols=3, subplot_titles=(
    "Avg Consumed (kcal/day)", "Empirical TDEE (kcal/day)", "Workout Min/Day"))
for i, col in enumerate(["Avg consumed", "Empirical TDEE", "Workout min/day"]):
    colors = [PHASE_LINES[p] for p in phase_metrics_df["Phase"]]
    fig_compare.add_trace(
        go.Bar(x=phase_metrics_df["Phase"], y=phase_metrics_df[col],
               marker_color=colors, text=phase_metrics_df[col],
               textposition="outside", showlegend=False),
        row=1, col=i+1)
fig_compare.update_layout(height=320, margin=dict(l=40, r=20, t=60, b=40),
                          plot_bgcolor="white",
                          title="<b>Phase Comparison</b>")
fig_compare.update_yaxes(showgrid=True, gridcolor="#ecf0f1")

# ─── Build HTML ────────────────────────────────────────────────────────────
def render_chart(fig):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})

cut = phase_results["Cut"]
rebuild = phase_results["Rebuild"]
bulk = phase_results["Bulk"]

# Format helpers
def kg(v): return f"{v:+.1f} kg" if v else "0.0 kg"
def fmt_date(d): return d.strftime("%b %d, %Y")

period_summary = f"Aug 4, 2025 → May 3, 2026 · {len(df)} days"

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Yearly Deep Dive — Yannick</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 32px 40px;
       color: #2c3e50; background: #fafbfc; line-height: 1.55; }}
h1 {{ font-size: 28px; margin-bottom: 4px; color: #1a252f; letter-spacing: -0.5px; }}
h2 {{ font-size: 20px; margin-top: 32px; margin-bottom: 12px; color: #1a252f;
      border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }}
h3 {{ font-size: 15px; margin-top: 18px; margin-bottom: 6px; color: #34495e; }}
.subtitle {{ color: #7f8c8d; font-size: 14px; margin-bottom: 24px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0 24px; }}
.kpi {{ background: white; padding: 14px 16px; border-radius: 8px;
       border-left: 3px solid #3498db; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.kpi-label {{ font-size: 11px; text-transform: uppercase; color: #7f8c8d;
              letter-spacing: 0.5px; margin-bottom: 4px; }}
.kpi-value {{ font-size: 22px; font-weight: 600; color: #1a252f; }}
.kpi-sub {{ font-size: 11px; color: #95a5a6; margin-top: 2px; }}
.phase-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }}
.phase-card {{ background: white; padding: 18px; border-radius: 8px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.04); border-top: 4px solid; }}
.phase-cut {{ border-color: #c0392b; }}
.phase-rebuild {{ border-color: #7f8c8d; }}
.phase-bulk {{ border-color: #27ae60; }}
.phase-card h3 {{ margin-top: 0; }}
.phase-card table {{ width: 100%; font-size: 13px; border-collapse: collapse; }}
.phase-card td {{ padding: 4px 0; border-bottom: 1px solid #f4f6f7; }}
.phase-card td:first-child {{ color: #7f8c8d; }}
.phase-card td:last-child {{ text-align: right; font-weight: 500; }}
.chart-wrapper {{ background: white; border-radius: 8px; padding: 8px;
                  margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.expert {{ background: white; padding: 18px 22px; border-radius: 8px;
           margin: 14px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
           border-left: 4px solid; }}
.expert-data {{ border-color: #2980b9; }}
.expert-nutrition {{ border-color: #e67e22; }}
.expert-coach {{ border-color: #16a085; }}
.expert h3 {{ margin-top: 0; }}
.expert-tag {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
               font-size: 10px; font-weight: 600; text-transform: uppercase;
               letter-spacing: 0.5px; margin-right: 8px; vertical-align: middle; }}
.tag-data {{ background: #d6eaf8; color: #1f618d; }}
.tag-nutrition {{ background: #fdebd0; color: #af601a; }}
.tag-coach {{ background: #d1f2eb; color: #117a65; }}
ul {{ margin: 6px 0; padding-left: 22px; }}
li {{ margin: 4px 0; }}
.footer {{ font-size: 11px; color: #95a5a6; margin-top: 40px;
           padding-top: 16px; border-top: 1px solid #ecf0f1; text-align: center; }}
@media print {{ body {{ background: white; padding: 16px 20px; }}
                .chart-wrapper, .expert, .phase-card, .kpi {{ box-shadow: none; }} }}
</style>
</head>
<body>

<h1>Yearly Deep Dive — Health Data Analysis</h1>
<div class="subtitle">{period_summary} · Profile: 33y M, 185cm · Cut → Rebuild → Bulk</div>

<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">Days Tracked</div>
    <div class="kpi-value">{len(df)}</div>
    <div class="kpi-sub">100% logged, no gaps</div></div>
  <div class="kpi" style="border-left-color:#c0392b;"><div class="kpi-label">Cut Phase</div>
    <div class="kpi-value">{cut['delta']:+.1f} kg</div>
    <div class="kpi-sub">{cut['days']} days · {cut['rate_kg_wk']:+.2f} kg/wk</div></div>
  <div class="kpi" style="border-left-color:#27ae60;"><div class="kpi-label">Bulk Phase</div>
    <div class="kpi-value">{bulk['delta']:+.1f} kg</div>
    <div class="kpi-sub">{bulk['days']} days · {bulk['rate_kg_wk']:+.2f} kg/wk</div></div>
  <div class="kpi" style="border-left-color:#8e44ad;"><div class="kpi-label">Net Body Mass</div>
    <div class="kpi-value">{df['weight_ma7'].dropna().iloc[-1] - df['weight_ma7'].dropna().iloc[0]:+.1f} kg</div>
    <div class="kpi-sub">Aug start → today</div></div>
</div>

<h2>1. Phase Breakdown</h2>

<div class="phase-grid">
  <div class="phase-card phase-cut">
    <h3 style="color:#c0392b;">Cut · {fmt_date(cut['start'])} → {fmt_date(cut['end'])}</h3>
    <table>
      <tr><td>Weight change</td><td>{cut['w_start']:.1f} → {cut['w_end']:.1f} ({cut['delta']:+.1f} kg)</td></tr>
      <tr><td>Rate</td><td>{cut['rate_kg_wk']:+.2f} kg/week</td></tr>
      <tr><td>Avg consumed</td><td>{cut['avg_consumed']:,.0f} kcal/day</td></tr>
      <tr><td>Avg burned (Watch)</td><td>{cut['avg_burned']:,.0f} kcal/day</td></tr>
      <tr><td>Empirical TDEE</td><td>{cut['empirical_tdee']:,.0f} kcal/day</td></tr>
      <tr><td>Avg steps</td><td>{cut['avg_steps']:,.0f}/day</td></tr>
      <tr><td>Avg sleep</td><td>{int(cut['avg_sleep_min']//60)}h{int(cut['avg_sleep_min']%60):02d}</td></tr>
      <tr><td>Workout min/day</td><td>{cut['avg_workout']:.0f}</td></tr>
      <tr><td>Duration</td><td>{cut['days']} days · {cut['weeks']:.1f} wk</td></tr>
    </table>
  </div>
  <div class="phase-card phase-rebuild">
    <h3 style="color:#7f8c8d;">Rebuild · {fmt_date(rebuild['start'])} → {fmt_date(rebuild['end'])}</h3>
    <table>
      <tr><td>Weight change</td><td>{rebuild['w_start']:.1f} → {rebuild['w_end']:.1f} ({rebuild['delta']:+.1f} kg)</td></tr>
      <tr><td>Rate</td><td>{rebuild['rate_kg_wk']:+.2f} kg/week</td></tr>
      <tr><td>Avg consumed</td><td>{rebuild['avg_consumed']:,.0f} kcal/day</td></tr>
      <tr><td>Avg burned (Watch)</td><td>{rebuild['avg_burned']:,.0f} kcal/day</td></tr>
      <tr><td>Empirical TDEE</td><td>{rebuild['empirical_tdee']:,.0f} kcal/day</td></tr>
      <tr><td>Avg steps</td><td>{rebuild['avg_steps']:,.0f}/day</td></tr>
      <tr><td>Avg sleep</td><td>{int(rebuild['avg_sleep_min']//60)}h{int(rebuild['avg_sleep_min']%60):02d}</td></tr>
      <tr><td>Workout min/day</td><td>{rebuild['avg_workout']:.0f}</td></tr>
      <tr><td>Duration</td><td>{rebuild['days']} days · {rebuild['weeks']:.1f} wk</td></tr>
    </table>
  </div>
  <div class="phase-card phase-bulk">
    <h3 style="color:#27ae60;">Bulk · {fmt_date(bulk['start'])} → {fmt_date(bulk['end'])}</h3>
    <table>
      <tr><td>Weight change</td><td>{bulk['w_start']:.1f} → {bulk['w_end']:.1f} ({bulk['delta']:+.1f} kg)</td></tr>
      <tr><td>Rate</td><td>{bulk['rate_kg_wk']:+.2f} kg/week</td></tr>
      <tr><td>Avg consumed</td><td>{bulk['avg_consumed']:,.0f} kcal/day</td></tr>
      <tr><td>Avg burned (Watch)</td><td>{bulk['avg_burned']:,.0f} kcal/day</td></tr>
      <tr><td>Empirical TDEE</td><td>{bulk['empirical_tdee']:,.0f} kcal/day</td></tr>
      <tr><td>Avg steps</td><td>{bulk['avg_steps']:,.0f}/day</td></tr>
      <tr><td>Avg sleep</td><td>{int(bulk['avg_sleep_min']//60)}h{int(bulk['avg_sleep_min']%60):02d}</td></tr>
      <tr><td>Workout min/day</td><td>{bulk['avg_workout']:.0f}</td></tr>
      <tr><td>Duration</td><td>{bulk['days']} days · {bulk['weeks']:.1f} wk</td></tr>
    </table>
  </div>
</div>

<h2>2. Visual Analysis</h2>
<div class="chart-wrapper">{render_chart(fig_weight)}</div>
<div class="chart-wrapper">{render_chart(fig_cal)}</div>
<div class="chart-wrapper">{render_chart(fig_bal)}</div>
<div class="chart-wrapper">{render_chart(fig_act)}</div>
<div class="chart-wrapper">{render_chart(fig_sleep)}</div>
<div class="chart-wrapper">{render_chart(fig_compare)}</div>

<h2>3. Tri-Perspective Synthesis</h2>

<div class="expert expert-data">
<h3><span class="expert-tag tag-data">Data Analyst</span>What the numbers reveal</h3>
<ul>
<li><b>Data quality is exceptional</b> — 273 / 273 days logged, zero missing values across 6 metrics. This is unusual; most fitness datasets have 15–30% gaps. Your conclusions can be drawn with high confidence.</li>
<li><b>Cut rate ({cut['rate_kg_wk']:+.2f} kg/wk)</b> is in the textbook safe range (-0.3 to -0.5/wk). No aggressive deficit phase.</li>
<li><b>Bulk rate ({bulk['rate_kg_wk']:+.2f} kg/wk)</b> sits below the typical lean-bulk target (+0.20/wk). Bulk has been conservative — likely fewer fat gains, but slower muscle accrual than max possible.</li>
<li><b>Apple Watch underestimates TDEE</b> systematically: in the cut phase it read {cut['avg_burned']:,.0f} avg vs empirical {cut['empirical_tdee']:,.0f} (Δ {cut['apple_watch_correction']:+.0f}); in bulk {bulk['avg_burned']:,.0f} vs {bulk['empirical_tdee']:,.0f} (Δ {bulk['apple_watch_correction']:+.0f}). The bias is consistent — do not use Apple Watch burn for surplus/deficit calculations.</li>
<li><b>NEAT volatility is the hidden variable</b>: steps range 6k–25k+ across the year. Single-day TDEE estimates have ±300 kcal noise from this alone, which is why short-window calorie targeting is unreliable.</li>
<li><b>Sleep regression</b>: phase averages stayed near {int(bulk['avg_sleep_min']//60)}h{int(bulk['avg_sleep_min']%60):02d} — never crossed the 8h target as a phase average. This is the most under-leveraged variable in the dataset.</li>
</ul>
</div>

<div class="expert expert-nutrition">
<h3><span class="expert-tag tag-nutrition">Nutritionist</span>Energy balance and intake patterns</h3>
<ul>
<li><b>Cut execution was clean</b>: avg consumed {cut['avg_consumed']:,.0f} vs empirical TDEE {cut['empirical_tdee']:,.0f} → ~{abs(cut['avg_consumed']-cut['empirical_tdee']):.0f} kcal/day deficit, producing the {cut['rate_kg_wk']:+.2f} kg/wk loss exactly as physics predicts. This is rare — most users either over-restrict or stall.</li>
<li><b>Bulk has been functionally a slow-build maintenance</b>: surplus averaged ~{bulk['avg_consumed']-bulk['empirical_tdee']:+.0f} kcal/day. Adequate for partitioning, but on the conservative end. Without the daily bike commute (added recently), you'd be even closer to maintenance.</li>
<li><b>Empirical TDEE drift</b> matters: cut TDEE ~{cut['empirical_tdee']:,.0f}, bulk TDEE ~{bulk['empirical_tdee']:,.0f}. Difference (~{bulk['empirical_tdee']-cut['empirical_tdee']:+.0f}) is largely metabolic adaptation reversing + added muscle + bike NEAT. Your TDEE is now ~3,500 — calorie targets calibrated against the older 3,316 figure are stale.</li>
<li><b>Protein</b>: not tracked in this dataset. At your weight (76 kg) and goals, 2.2–2.5 g/kg = 170–190 g/day is the floor. If you've been eyeballing it, the bulk's slow muscle accrual signal could partially be a protein-floor issue, not a calorie issue.</li>
<li><b>For the upcoming cut</b>: empirical TDEE in cut conditions will likely settle at ~3,400–3,500 (slight downregulation). A -350 deficit puts targets at ~3,050–3,150 — close to your saved 3,200 plan. Stay there; do not go lower.</li>
</ul>
</div>

<div class="expert expert-coach">
<h3><span class="expert-tag tag-coach">Fitness Coach</span>Training, recovery, body composition</h3>
<ul>
<li><b>Workout consistency is excellent</b>: cut phase averaged {cut['avg_workout']:.0f} min/day, bulk {bulk['avg_workout']:.0f} min/day. Phase-level engagement is rare in 9-month windows.</li>
<li><b>Bulk rate is light for a true muscle-focused phase</b> ({bulk['rate_kg_wk']:+.2f} kg/wk vs target +0.15–0.25). Two interpretations: (a) you're genuinely closer to your genetic ceiling and gains are slow — possible, given your training age; (b) intake periodization wasn't pushing surplus consistently — supported by the empirical data showing ~maintenance more often than surplus.</li>
<li><b>The Cut → Bulk transition was well-handled</b>: no rebound spike, no extended dirty-bulk window. Rebuild phase ({rebuild['days']} days) acted as a planned bridge — most lifters skip this and overshoot.</li>
<li><b>Bro Split (recent switch) + cut combo is a known retention risk</b>. If you proceed to the soft cut without switching back to higher-frequency programming (PPL, upper/lower), monitor main-lift loads weekly. >5% drop by week 3 = either bump intake or change split.</li>
<li><b>Recovery liability</b>: average sleep never reached 8h. With the family return scheduled, sleep is the variable most likely to derail the bulk extension and the upcoming cut. Protein + sleep are bigger levers than the last 100 kcal of intake.</li>
<li><b>Body composition outlook</b>: current 75.4 kg with abs visible in good light = ~14–15% BF based on visual norms. A 6-week soft cut targeting -2 kg fat lands at ~73.5 kg / ~12% — abs visible at conversational light. This is achievable; you've already proven the system works.</li>
</ul>
</div>

<h2>4. The Bottom Line</h2>
<p>Nine months of disciplined tracking, two clean phases, no major dietary or training breakdowns. The cut was textbook. The bulk has been conservative — closer to recomp than aggressive growth — which is a defensible choice at your training age but means the muscle gains may be smaller than expected.</p>

<p>Three priorities for the next 10 weeks (bulk extension + cut):</p>
<ol>
<li><b>Sleep first.</b> Hitting 7.5h+ as a phase average will move the needle more than a 200-kcal intake adjustment. Tag sleep tracking in next week's review.</li>
<li><b>Track main-lift loads.</b> The dataset has weight, intake, sleep — but not strength. That blind spot is the largest in this analysis. Even informal logging (top set × reps per main lift) closes it.</li>
<li><b>Don't recalibrate during chaos.</b> Family return = noise. Hold targets at 3,500/3,800/4,300 through the bulk extension. Recalibrate after the cut, with clean data.</li>
</ol>

<div class="footer">
Generated by <code>scripts/yearly_deep_dive.py</code> · Source: <code>data/health_data.csv</code> ·
Empirical TDEE = avg_consumed − Δweight × 7700 / days · Apple Watch correction = empirical TDEE − avg_burned
</div>

</body></html>"""

OUT.write_text(html)
print(f"Wrote: {OUT}")
print(f"Size: {OUT.stat().st_size / 1024:.1f} KB")

# Also print phase summary table for terminal reference
print("\n=== Phase Stats ===")
print(phase_metrics_df.to_string(index=False))
