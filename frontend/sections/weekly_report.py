"""Interactive weekly review using the current WADP protocol."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from utils.report_generator import generate_weekly_report_data
from utils.html_export import render_report_html
from utils.wadp import CURRENT_DEFICIT_KCAL


def _snap_to_monday(day):
    return day - timedelta(days=day.weekday())


def _metric(value, unit, decimals=1):
    return f"{value:.{decimals}f} {unit}" if value is not None else "N/A"


def _delta(value, prior, unit, decimals=1):
    return f"{value - prior:+.{decimals}f} {unit} vs prior week" if value is not None and prior is not None else None


def render(df: pd.DataFrame):
    st.title("📋 Weekly Report")
    st.caption("Monday–Sunday review · observed WADP = 2,000 + Move − intake. Positive means deficit.")
    if df.empty:
        st.warning("No health data available.")
        return

    today = date.today()
    this_week = _snap_to_monday(today)
    last_week = this_week - timedelta(days=7)
    if "report_week_start" not in st.session_state:
        st.session_state.report_week_start = last_week
    left, right = st.columns(2)
    with left:
        if st.button("Last week", use_container_width=True):
            st.session_state.report_week_start = last_week
            st.rerun()
    with right:
        if st.button("This week", use_container_width=True):
            st.session_state.report_week_start = this_week
            st.rerun()
    with st.expander("Browse a past week"):
        selected = st.date_input("Date in target week", value=st.session_state.report_week_start,
                                 min_value=df["date"].min().date(), max_value=today)
        if st.button("Load this week"):
            st.session_state.report_week_start = _snap_to_monday(selected)
            st.rerun()

    report = generate_weekly_report_data(df, st.session_state.report_week_start)
    if not report["days_available"]:
        st.warning("No health records for that week.")
        return
    st.subheader(f"{report['week_start']:%b %d}–{report['week_end']:%b %d, %Y}")
    st.caption(f"{report['days_available']}/7 health days · current target deficit {CURRENT_DEFICIT_KCAL} kcal/day. The target is a relative dial; body and strength trends decide changes.")

    cols = st.columns(4)
    cols[0].metric("Observed WADP deficit", _metric(report["avg_deficit"], "kcal/day", 0),
                   _delta(report["avg_deficit"], report["prior_avg_deficit"], "kcal/day", 0),
                   help=f"{report['deficit_days']} days with Move and intake; prior week {report['prior_deficit_days']} days")
    cols[1].metric("Weight 7-day average", _metric(report["weight_ma7"], "kg", 2),
                   _delta(report["weight_ma7"], report["prior_weight_ma7"], "kg", 2))
    cols[2].metric("Waist 7-day average", _metric(report["waist_ma7"], "cm", 2),
                   _delta(report["waist_ma7"], report["prior_waist_ma7"], "cm", 2))
    cols[3].metric("Waist 14-day average", _metric(report["waist_ma14"], "cm", 2),
                   _delta(report["waist_ma14"], report["prior_waist_ma14"], "cm", 2))
    cols = st.columns(3)
    cols[0].metric("Training sessions", str(report["sessions"]) if report["sessions"] is not None else "N/A",
                   f"Prior week: {report['prior_sessions']}" if report["prior_sessions"] is not None else None,
                   help="Distinct date and session pairs in training_log.csv")
    cols[1].metric("Average sleep", _metric(report["avg_sleep_h"], "h"))
    cols[2].metric("Average steps", f"{report['avg_steps']:,.0f}" if report["avg_steps"] is not None else "N/A")
    for note in report["signals"]:
        st.info(note)
    st.plotly_chart(report["fig_trend"], use_container_width=True)
    st.plotly_chart(report["fig_deficit"], use_container_width=True)
    st.download_button("⬇️ Download HTML Report", render_report_html(report).encode("utf-8"),
                       file_name=f"health_report_{report['week_start']:%Y-%m-%d}.html", mime="text/html")
