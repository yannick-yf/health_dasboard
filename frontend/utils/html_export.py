"""
HTML report renderer — converts a report_data dict into a self-contained HTML string.

Charts are embedded via Plotly CDN (requires internet to view).
"""

import pandas as pd
import plotly.io as pio


def render_report_html(report_data: dict) -> str:
    """Build a self-contained dark-themed HTML report. Returns the full HTML string."""
    week_start  = report_data["week_start"]
    week_end    = report_data["week_end"]
    week_label  = f"{week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}"
    bulk_week   = f"Bulk Week {report_data['bulk_week_num']}" if report_data.get("bulk_week_num") else ""

    status_label = report_data["status_label"]
    status_color = report_data["status_color"]

    # ── KPI formatting ────────────────────────────────────────────────────────
    avg_weight   = report_data.get("avg_weight")
    weight_delta = report_data.get("weight_delta")
    avg_sleep_h  = report_data.get("avg_sleep_h")
    avg_steps    = report_data.get("avg_steps")
    avg_surplus  = report_data.get("avg_surplus")
    workout_days = report_data.get("workout_days")

    def _fmt(val, fmt, fallback="N/A"):
        return fmt.format(val) if val is not None and not (isinstance(val, float) and pd.isna(val)) else fallback

    weight_str       = _fmt(avg_weight,   "{:.2f} kg")
    weight_delta_str = _fmt(weight_delta, "{:+.2f} kg", "")
    sleep_str        = _fmt(avg_sleep_h,  "{:.1f}h")
    steps_str        = (f"{avg_steps / 1000:.1f}k" if avg_steps is not None else "N/A")
    surplus_str      = _fmt(avg_surplus,  "{:+.0f} kcal")
    workouts_str     = f"{workout_days}/7" if workout_days is not None else "N/A"

    # ── Charts ────────────────────────────────────────────────────────────────
    def _chart_html(fig, first=False):
        if fig is None:
            return '<p class="no-data">No data available for this period</p>'
        return pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn" if first else False,
            config={"displayModeBar": False, "responsive": True},
        )

    chart_weight = _chart_html(report_data.get("fig_weight"), first=True)
    chart_energy = _chart_html(report_data.get("fig_energy"))
    chart_sleep  = _chart_html(report_data.get("fig_sleep"))
    chart_steps  = _chart_html(report_data.get("fig_steps"))

    # ── Signals ───────────────────────────────────────────────────────────────
    signals_html = _build_signals_html(report_data.get("signals", []))

    # ── Targets ───────────────────────────────────────────────────────────────
    next_week_start = week_end + pd.Timedelta(days=1)
    next_week_end   = week_end + pd.Timedelta(days=7)
    next_week_label = f"{next_week_start.strftime('%b %d')} – {next_week_end.strftime('%b %d')}"
    targets_html = _build_targets_html(report_data.get("targets"), next_week_label)

    # ── Data note ─────────────────────────────────────────────────────────────
    data_note_html = ""
    if report_data.get("data_note"):
        data_note_html = f'<div class="data-note">ℹ️ {report_data["data_note"]}</div>'

    gain_rate_str = (
        f"{report_data['gain_rate']:+.2f} kg/wk"
        if report_data.get("gain_rate") is not None
        else "—"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Weekly Health Report — {week_label}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0f172a;
      color: #f0f2f6;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 15px;
      line-height: 1.5;
      padding: 2rem;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}

    /* Header */
    .report-header {{
      background: linear-gradient(135deg, #1e293b 0%, {status_color}22 100%);
      border-left: 5px solid {status_color};
      border-radius: 12px;
      padding: 1.5rem 2rem;
      margin-bottom: 1.5rem;
    }}
    .report-header h1 {{ font-size: 1.1rem; color: #9ca3af; text-transform: uppercase;
                         letter-spacing: .08em; font-weight: 600; margin-bottom: .3rem; }}
    .report-header h2 {{ font-size: 1.75rem; font-weight: 700; color: #f0f2f6; }}
    .header-meta {{
      display: flex; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;
    }}
    .header-pill {{
      background: rgba(255,255,255,0.06);
      border-radius: 20px;
      padding: .3rem .9rem;
      font-size: .85rem;
      color: #cbd5e1;
    }}
    .status-badge {{
      background: {status_color}22;
      border: 1px solid {status_color};
      border-radius: 20px;
      padding: .3rem .9rem;
      font-size: .85rem;
      font-weight: 700;
      color: {status_color};
    }}

    /* Data note */
    .data-note {{
      background: #1e3a5f;
      border-left: 3px solid #3b82f6;
      border-radius: 6px;
      padding: .6rem 1rem;
      font-size: .85rem;
      color: #93c5fd;
      margin-bottom: 1.2rem;
    }}

    /* Section */
    .section {{
      background: #1e293b;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .section h3 {{
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: .07em;
      color: #9ca3af;
      font-weight: 600;
      margin-bottom: 1.2rem;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      padding-bottom: .7rem;
    }}

    /* KPI snapshot */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 1rem;
    }}
    .kpi-card {{
      background: #0f172a;
      border-radius: 10px;
      padding: 1rem;
      text-align: center;
    }}
    .kpi-label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; color: #9ca3af; }}
    .kpi-value {{ font-size: 1.5rem; font-weight: 700; color: #f0f2f6; margin: .3rem 0; }}
    .kpi-delta {{ font-size: .75rem; color: #6b7280; }}

    /* Two-column charts */
    .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
    @media (max-width: 700px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}

    /* Signals */
    .signal {{ display: flex; align-items: flex-start; gap: .6rem; padding: .5rem .8rem;
               border-radius: 8px; margin-bottom: .5rem; font-size: .88rem; }}
    .signal.ok   {{ background: rgba(16,185,129,.12); color: #6ee7b7; }}
    .signal.warn {{ background: rgba(245,158,11,.12);  color: #fcd34d; }}
    .signal.info {{ background: rgba(99,102,241,.12);  color: #a5b4fc; }}
    .signal-icon {{ flex-shrink: 0; margin-top: .1rem; }}

    /* Targets */
    .targets-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 1rem;
    }}
    .target-card {{
      background: #0f172a;
      border-radius: 10px;
      padding: 1rem;
      text-align: center;
    }}
    .target-label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; color: #9ca3af; }}
    .target-value {{ font-size: 1.6rem; font-weight: 700; margin: .4rem 0; }}
    .target-unit  {{ font-size: .72rem; color: #6b7280; }}

    /* Misc */
    .no-data {{ color: #6b7280; font-style: italic; text-align: center; padding: 1.5rem 0; }}
    footer {{ text-align: center; color: #475569; font-size: .8rem; margin-top: 2rem; padding-top: 1rem;
              border-top: 1px solid rgba(255,255,255,0.06); }}
  </style>
</head>
<body>
<div class="container">

  <!-- ── Header ── -->
  <div class="report-header">
    <h1>Weekly Health Report</h1>
    <h2>Week of {week_label}</h2>
    <div class="header-meta">
      {f'<span class="header-pill">{bulk_week}</span>' if bulk_week else ''}
      <span class="status-badge">{status_label}</span>
      <span class="header-pill">4-week rate: {gain_rate_str}</span>
      <span class="header-pill">{report_data.get('days_available', 0)}/7 days of data</span>
    </div>
  </div>

  {data_note_html}

  <!-- ── Snapshot KPIs ── -->
  <div class="section">
    <h3>Snapshot</h3>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Avg Weight</div>
        <div class="kpi-value">{weight_str}</div>
        <div class="kpi-delta">{weight_delta_str}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Sleep</div>
        <div class="kpi-value">{sleep_str}</div>
        <div class="kpi-delta">target 7.5h</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Steps</div>
        <div class="kpi-value">{steps_str}</div>
        <div class="kpi-delta">NEAT variable</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Surplus</div>
        <div class="kpi-value">{surplus_str}</div>
        <div class="kpi-delta">target +200–300</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Workouts</div>
        <div class="kpi-value">{workouts_str}</div>
        <div class="kpi-delta">days &gt;30 min</div>
      </div>
    </div>
  </div>

  <!-- ── Body Composition ── -->
  <div class="section">
    <h3>Body Composition — 4-Week Window</h3>
    {chart_weight}
  </div>

  <!-- ── Energy Balance ── -->
  <div class="section">
    <h3>Energy Balance</h3>
    {chart_energy}
  </div>

  <!-- ── Recovery & Activity ── -->
  <div class="section">
    <h3>Recovery &amp; Activity</h3>
    <div class="chart-row">
      <div>{chart_sleep}</div>
      <div>{chart_steps}</div>
    </div>
  </div>

  <!-- ── Signals ── -->
  <div class="section">
    <h3>Signals &amp; Observations</h3>
    {signals_html}
  </div>

  <!-- ── Next Week Targets ── -->
  {targets_html}

  <footer>Generated by Health Tracker · Data up to {week_end.strftime('%b %d, %Y')}</footer>

</div>
</body>
</html>"""


def _build_signals_html(signals: list) -> str:
    if not signals:
        return '<p class="no-data">No signals computed — insufficient data.</p>'
    rows = []
    icon_map = {"ok": "✅", "warn": "⚠️", "info": "ℹ️"}
    for s in signals:
        icon = icon_map.get(s["type"], "•")
        rows.append(
            f'<div class="signal {s["type"]}">'
            f'<span class="signal-icon">{icon}</span>'
            f'<span>{s["msg"]}</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _build_targets_html(targets: dict | None, next_week_label: str) -> str:
    if targets is None:
        return (
            '<div class="section">'
            '<h3>Plan for Next Week</h3>'
            '<p class="no-data">Insufficient data to compute calorie targets.</p>'
            '</div>'
        )

    train_day   = targets.get("training_day", "—")
    rest_day    = targets.get("rest_day", "—")
    daily_avg   = targets.get("daily_avg_target", "—")
    emp_tdee    = targets.get("empirical_tdee", "—")
    change      = targets.get("change_from_current")
    change_str  = f"{change:+,} vs current" if change is not None else ""

    return f"""
  <div class="section">
    <h3>Plan for Week of {next_week_label}</h3>
    <div class="targets-grid">
      <div class="target-card" style="border-top: 3px solid #6366f1;">
        <div class="target-label">🏋️ Training Day</div>
        <div class="target-value" style="color:#6366f1;">{train_day:,}</div>
        <div class="target-unit">kcal</div>
      </div>
      <div class="target-card" style="border-top: 3px solid #94a3b8;">
        <div class="target-label">🛋️ Rest Day</div>
        <div class="target-value" style="color:#94a3b8;">{rest_day:,}</div>
        <div class="target-unit">kcal</div>
      </div>
      <div class="target-card" style="border-top: 3px solid #10b981;">
        <div class="target-label">📊 Daily Avg Target</div>
        <div class="target-value" style="color:#10b981;">{daily_avg:,}</div>
        <div class="target-unit">{change_str}</div>
      </div>
      <div class="target-card" style="border-top: 3px solid #475569;">
        <div class="target-label">⚗️ Empirical TDEE</div>
        <div class="target-value" style="color:#cbd5e1;">{emp_tdee:,}</div>
        <div class="target-unit">kcal/day · 28-day window</div>
      </div>
    </div>
  </div>"""
