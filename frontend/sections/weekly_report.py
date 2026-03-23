"""
Weekly Report page — generate an interactive HTML report for any past week.
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from utils.report_generator import generate_weekly_report_data
from utils.html_export import render_report_html
from utils.bulk_constants import BULK_START, GAIN_MIN, GAIN_MAX


def _snap_to_monday(d: date) -> date:
    """Return the Monday of the week containing d."""
    return d - timedelta(days=d.weekday())


def render(df: pd.DataFrame):
    st.title("📋 Weekly Report")
    st.caption("One-click review of your last or current week. All metrics are scoped to data available up to that week's end.")

    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
        return

    today = date.today()
    min_date = df["date"].min().date() if not df.empty else date(2025, 11, 5)

    last_week_start = _snap_to_monday(today - timedelta(days=7))
    this_week_start = _snap_to_monday(today)

    # ── Quick-select buttons ──────────────────────────────────────────────────
    # Initialise session state to last completed week on first load
    if "report_week_start" not in st.session_state:
        st.session_state["report_week_start"] = last_week_start

    last_week_end = last_week_start + timedelta(days=6)
    this_week_end = this_week_start + timedelta(days=6)

    # Count available days for labelling the This Week button
    this_week_days = len(df[
        (df["date"] >= pd.Timestamp(this_week_start)) &
        (df["date"] <= pd.Timestamp(today))
    ])

    col_last, col_this, col_spacer = st.columns([2, 2, 3])
    with col_last:
        last_label = f"⬅ Last Week  ({last_week_start.strftime('%b %d')}–{last_week_end.strftime('%b %d')})"
        if st.button(last_label, use_container_width=True, type="primary"
                     if st.session_state["report_week_start"] == last_week_start else "secondary"):
            st.session_state["report_week_start"] = last_week_start
            st.rerun()
    with col_this:
        this_label = f"This Week ➡  ({this_week_start.strftime('%b %d')}–, {this_week_days}d)"
        if st.button(this_label, use_container_width=True, type="primary"
                     if st.session_state["report_week_start"] == this_week_start else "secondary"):
            st.session_state["report_week_start"] = this_week_start
            st.rerun()

    # ── Historical lookup expander ────────────────────────────────────────────
    with st.expander("🗂 Browse a past week"):
        selected_date = st.date_input(
            "Pick any date in the target week:",
            value=st.session_state["report_week_start"],
            min_value=min_date,
            max_value=today,
            help="The report will cover Monday–Sunday of the week containing this date.",
            key="history_date_picker",
        )
        if st.button("Load this week", key="load_history"):
            st.session_state["report_week_start"] = _snap_to_monday(selected_date)
            st.rerun()

    week_start = st.session_state["report_week_start"]
    week_end   = week_start + timedelta(days=6)

    # ── Validation ────────────────────────────────────────────────────────────
    df_week_check = df[
        (df["date"] >= pd.Timestamp(week_start)) & (df["date"] <= pd.Timestamp(week_end))
    ]
    days_available = len(df_week_check)

    if days_available == 0:
        st.error("No data found for this week. Choose a different week or add data first.")
        return

    if week_end > today:
        st.info(
            f"**Week in progress** — {days_available}/7 days complete. "
            "Calorie targets are valid; body composition metrics are directional."
        )
    elif days_available < 4:
        st.warning(
            f"Only {days_available}/7 days have data for this week. "
            "The report will be generated but some metrics may be incomplete."
        )

    # ── Generate ──────────────────────────────────────────────────────────────
    with st.spinner("Generating report…"):
        report = generate_weekly_report_data(df, week_start)

    # ── Inline preview ────────────────────────────────────────────────────────
    _render_status_banner(report)
    _render_kpi_row(report)
    _render_charts(report)
    _render_signals(report)
    _render_targets_preview(report)

    # ── Download button ───────────────────────────────────────────────────────
    st.markdown("---")
    html_str  = render_report_html(report)
    html_bytes = html_str.encode("utf-8")
    filename  = f"health_report_{week_start.strftime('%Y-%m-%d')}.html"

    st.download_button(
        label="⬇️ Download HTML Report",
        data=html_bytes,
        file_name=filename,
        mime="text/html",
        help="Opens in any browser. Charts are interactive. Requires internet for Plotly CDN.",
    )


# ── Inline section renderers ─────────────────────────────────────────────────

def _render_status_banner(report: dict):
    status_label = report["status_label"]
    status_color = report["status_color"]
    bulk_week    = f"Bulk Week {report['bulk_week_num']}" if report.get("bulk_week_num") else "Pre-bulk"
    gain_str     = f"{report['gain_rate']:+.2f} kg/wk" if report.get("gain_rate") is not None else "—"
    ma_delta     = report.get("weekly_ma_delta")
    ma_delta_str = f"{ma_delta:+.2f} kg" if ma_delta is not None else "—"
    ma_delta_color = "#ef4444" if (ma_delta or 0) > 0.25 else "#10b981" if (ma_delta or 0) < 0 else "#f0f2f6"
    week_start   = report["week_start"]
    week_end     = report["week_end"]
    data_note    = report.get("data_note")

    meso = report.get("mesocycle")
    meso_pill = (
        f'<span style="color:#a78bfa; font-size:.75rem; font-weight:600;">{meso["label"]}</span>'
        if meso else ""
    )
    meso_sep = ' <span style="color:#475569;">·</span> ' if meso else ""

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, {status_color}22 100%);
            border-left: 4px solid {status_color};
            border-radius: 10px; padding: 1rem 1.5rem; margin-bottom: 1rem;
        ">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:.5rem;">
            <div>
              <div style="color:#9ca3af; font-size:.75rem; text-transform:uppercase; letter-spacing:.05em;">
                {bulk_week}{meso_sep}{meso_pill}
                <span style="color:#475569;"> · </span>
                {week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}
              </div>
              <div style="color:{status_color}; font-size:1.15rem; font-weight:700; margin-top:.2rem;">
                {status_label}
              </div>
            </div>
            <div style="display:flex; gap:2rem;">
              <div style="text-align:center;">
                <div style="color:#9ca3af; font-size:.7rem;">4-WEEK RATE</div>
                <div style="color:#f0f2f6; font-size:1.1rem; font-weight:700;">{gain_str}</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#9ca3af; font-size:.7rem;">THIS WEEK (7d MA)</div>
                <div style="color:{ma_delta_color}; font-size:1.1rem; font-weight:700;">{ma_delta_str}</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#9ca3af; font-size:.7rem;">DATA COVERAGE</div>
                <div style="color:#f0f2f6; font-size:1.1rem; font-weight:700;">{report['days_available']}/7 days</div>
              </div>
            </div>
          </div>
          {f'<div style="margin-top:.8rem; font-size:.8rem; color:#93c5fd;">ℹ️ {data_note}</div>' if data_note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_row(report: dict):
    avg_weight   = report.get("avg_weight")
    weight_delta = report.get("weight_delta")
    avg_sleep_h  = report.get("avg_sleep_h")
    avg_steps    = report.get("avg_steps")
    avg_surplus  = report.get("avg_surplus")
    workout_days = report.get("workout_days")

    cols = st.columns(5)
    cols[0].metric(
        "Avg Weight",
        f"{avg_weight:.2f} kg" if avg_weight is not None and not pd.isna(avg_weight) else "N/A",
        delta=f"{weight_delta:+.2f} kg vs prior wk" if weight_delta is not None and not pd.isna(weight_delta) else None,
    )
    cols[1].metric(
        "Avg Sleep",
        f"{avg_sleep_h:.1f}h" if avg_sleep_h is not None else "N/A",
        delta="target 7.5h",
        delta_color="off",
    )
    cols[2].metric(
        "Avg Steps",
        f"{avg_steps / 1000:.1f}k" if avg_steps is not None else "N/A",
    )
    cols[3].metric(
        "Avg Surplus",
        f"{avg_surplus:+.0f} kcal" if avg_surplus is not None and not pd.isna(avg_surplus) else "N/A",
        delta="target +200–300 kcal",
        delta_color="off",
    )
    cols[4].metric(
        "Workouts",
        f"{workout_days}/7" if workout_days is not None else "N/A",
        help="Days with >30 min workout logged",
    )


def _render_charts(report: dict):
    st.subheader("Body Composition — 4-Week Window")
    if report.get("fig_weight"):
        st.plotly_chart(report["fig_weight"], use_container_width=True)
    else:
        st.info("Not enough data for the weight chart.")

    st.subheader("Energy Balance")
    if report.get("fig_energy"):
        st.plotly_chart(report["fig_energy"], use_container_width=True)
    else:
        st.info("No energy balance data for this week.")

    st.subheader("Recovery & Activity")
    col_a, col_b = st.columns(2)
    with col_a:
        if report.get("fig_sleep"):
            st.plotly_chart(report["fig_sleep"], use_container_width=True)
        else:
            st.info("No sleep data.")
    with col_b:
        if report.get("fig_steps"):
            st.plotly_chart(report["fig_steps"], use_container_width=True)
        else:
            st.info("No steps/workout data.")


def _render_signals(report: dict):
    signals = report.get("signals", [])
    if not signals:
        return

    n_warn = sum(1 for s in signals if s["type"] == "warn")
    n_ok   = sum(1 for s in signals if s["type"] == "ok")
    icon   = "⚠️" if n_warn else "✅"
    label  = f"{icon} {n_warn} warning(s) — {n_ok} green signal(s)"

    with st.expander(label, expanded=bool(n_warn)):
        for s in signals:
            if s["type"] == "ok":
                st.success(s["msg"])
            elif s["type"] == "warn":
                st.warning(s["msg"])
            else:
                st.info(s["msg"])


def _render_targets_preview(report: dict):
    targets = report.get("targets")
    week_end = report["week_end"]
    next_week_start = week_end + pd.Timedelta(days=1)
    next_week_end   = week_end + pd.Timedelta(days=7)

    st.subheader(
        f"Plan for Week of {next_week_start.strftime('%b %d')} – {next_week_end.strftime('%b %d')}"
    )

    if targets is None:
        st.info("Insufficient data to compute calorie targets for next week.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""
        <div style="background:#1e293b; border:1px solid #6366f1; border-radius:10px;
                    padding:1rem; text-align:center;">
          <div style="color:#9ca3af; font-size:.72rem; text-transform:uppercase;">🏋️ Training Day</div>
          <div style="color:#6366f1; font-size:1.6rem; font-weight:700; margin:.3rem 0;">
            {targets['training_day']:,}
          </div>
          <div style="color:#6b7280; font-size:.7rem;">kcal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"""
        <div style="background:#1e293b; border:1px solid #94a3b8; border-radius:10px;
                    padding:1rem; text-align:center;">
          <div style="color:#9ca3af; font-size:.72rem; text-transform:uppercase;">🛋️ Rest Day</div>
          <div style="color:#94a3b8; font-size:1.6rem; font-weight:700; margin:.3rem 0;">
            {targets['rest_day']:,}
          </div>
          <div style="color:#6b7280; font-size:.7rem;">kcal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    change = targets.get("change_from_current", 0)
    col3.markdown(
        f"""
        <div style="background:#1e293b; border:1px solid #10b981; border-radius:10px;
                    padding:1rem; text-align:center;">
          <div style="color:#9ca3af; font-size:.72rem; text-transform:uppercase;">📊 Daily Avg</div>
          <div style="color:#10b981; font-size:1.6rem; font-weight:700; margin:.3rem 0;">
            {targets['daily_avg_target']:,}
          </div>
          <div style="color:#6b7280; font-size:.7rem;">{change:+,} vs current</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col4.markdown(
        f"""
        <div style="background:#1e293b; border:1px solid #475569; border-radius:10px;
                    padding:1rem; text-align:center;">
          <div style="color:#9ca3af; font-size:.72rem; text-transform:uppercase;">⚗️ Empirical TDEE</div>
          <div style="color:#cbd5e1; font-size:1.6rem; font-weight:700; margin:.3rem 0;">
            {targets['empirical_tdee']:,}
          </div>
          <div style="color:#6b7280; font-size:.7rem;">kcal/day · 28-day window</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
