"""Training views shared by the Analytics Dashboard and the Deep Dive.

Everything comes from data/training_log.csv through utils/training_metrics.py: cycles, hard
sets per muscle, venue-aware progress against each exercise's own past, and a weekly table that
puts training next to the health data.
"""

import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import training_metrics as tm

GREEN, GREY, ORANGE, ACCENT = "#10b981", "#64748b", "#f97316", "#667eea"
# Explicit colours so the charts read the same whatever the Streamlit theme is.
PALETTE = [
    "#667eea",
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#06b6d4",
    "#ec4899",
    "#84cc16",
    "#f97316",
    "#14b8a6",
    "#a855f7",
    "#eab308",
]
VENUE_COLORS = {
    "ToTheLimitGym": "#667eea",
    "Basic Fit": "#f59e0b",
    "Home": "#10b981",
    "Hotel": "#ec4899",
    "Other": "#06b6d4",
    "Unrecorded": "#94a3b8",
}
STATUS_LABELS = {"PR": "🏆 PR", "up": "▲ up", "flat": "= flat", "down": "▼ down", "new": "🆕 new"}


def _load_log():
    base = os.path.dirname(st.session_state.get("csv_path", "data/health_data.csv"))
    return tm.load_training_log(os.path.join(base, "training_log.csv"))


def _bodyweight(health):
    return health.dropna(subset=["weight"]).drop_duplicates("date").set_index("date")["weight"]


def _style(fig, height=330, **layout):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f2f6",
        legend=dict(orientation="h", y=-0.2),
        **layout,
    )
    fig.update_xaxes(automargin=True, gridcolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(automargin=True, gridcolor="rgba(148,163,184,0.2)")
    return fig


def _progress_view(table):
    shown = table.assign(Status=table["Status"].map(STATUS_LABELS))
    return shown[
        ["Exercise", "Compared", "Best set", "e1RM", "Previous e1RM", "Change %", "Status", "Note"]
    ]


def _week_ticks(fig, weeks):
    """Label the axis with the Monday each week starts on, under its bar."""
    fig.update_xaxes(tickmode="array", tickvals=list(weeks), ticktext=[f"{d:%d %b}" for d in weeks])
    return fig


def _hard_sets_chart(weekly):
    fig = go.Figure()
    for i, muscle in enumerate(weekly.sum().sort_values(ascending=False).index):
        fig.add_trace(
            go.Bar(
                x=weekly.index,
                y=weekly[muscle],
                name=muscle,
                marker_color=PALETTE[i % len(PALETTE)],
            )
        )
    return _week_ticks(
        _style(
            fig,
            title="Hard sets per week by muscle",
            barmode="stack",
            yaxis_title="hard sets (RIR ≤ 3)",
        ),
        weekly.index,
    )


def _progress_chart(progress):
    fig = go.Figure()
    for column, color in [("Up", GREEN), ("Flat", GREY), ("Down", ORANGE)]:
        fig.add_trace(go.Bar(x=progress.index, y=progress[column], name=column, marker_color=color))
    return _week_ticks(
        _style(
            fig,
            title="Lifts up / flat / down vs their previous time",
            barmode="stack",
            yaxis_title="exercises",
        ),
        progress.index,
    )


def render_dashboard_block(health, start_date, end_date):
    """Compact training block for the selected time period of the Analytics Dashboard."""
    st.subheader("🏋️ Training")
    log = _load_log()
    if log is None:
        st.caption("Training log unavailable.")
        return
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    window = log[(log["dt"] >= start) & (log["dt"] <= end)]
    if window.empty:
        st.info("No training sessions in this period.")
        return

    sessions = tm.session_table(log)
    history = tm.best_sets(log, _bodyweight(health))
    weeks = max((end - max(start, log["dt"].min())).days + 1, 7) / 7
    progress = tm.weekly_progress(history)
    in_period = progress[(progress.index >= start - pd.Timedelta(days=6)) & (progress.index <= end)]
    compared = int(in_period[["Up", "Flat", "Down"]].sum().sum())
    prs = tm.pr_events(history, start, end)

    cols = st.columns(5)
    cols[0].metric(
        "Sessions",
        len(tm.session_table(window)),
        f"{len(tm.session_table(window)) / weeks:.1f} per week",
    )
    cols[1].metric("Hard sets per week", f"{len(tm.hard_sets(window)) / weeks:.0f}")
    cols[2].metric("🏆 PRs", len(prs))
    cols[3].metric(
        "Lifts that went up",
        f"{in_period['Up'].sum() / compared * 100:.0f}%" if compared else "N/A",
        help="Share of exercises done in this period that beat their previous time",
    )
    cycles = tm.cycles_in_range(sessions, start, end)
    cols[4].metric("Cycles completed", sum(1 for c in cycles if c["complete"]))
    for cycle in cycles[-2:]:
        st.markdown(tm.describe_cycle(cycle))

    left, right = st.columns(2)
    weekly = tm.weekly_hard_sets(log, start, end)
    if not weekly.empty:
        left.plotly_chart(_hard_sets_chart(weekly), use_container_width=True)
    if not in_period.empty:
        right.plotly_chart(_progress_chart(in_period), use_container_width=True)

    st.caption("Weeks run Monday–Sunday; the first and last week of the period can be partial.")
    with st.expander("PRs and progress in this period"):
        if prs.empty:
            st.caption("No PRs in this period.")
        else:
            st.markdown("**🏆 PRs** (best estimated 1RM ever for that exercise and gym type)")
            st.dataframe(
                prs.assign(Date=prs["Date"].dt.strftime("%d %b")).drop(columns="Muscle"),
                hide_index=True,
                use_container_width=True,
            )
        change = tm.period_progress(history, start, end)
        if not change.empty:
            st.markdown("**Best of the period vs the last time before it**")
            st.dataframe(_progress_view(change), hide_index=True, use_container_width=True)
        st.caption(
            "Free weights are compared across gyms; machines and cables only within the "
            "same gym. Abs are left out."
        )


def _heatmap(weekly):
    order = weekly.sum().sort_values(ascending=False).index
    fig = go.Figure(
        go.Heatmap(
            z=weekly[order].T.values,
            x=[f"{d:%d %b}" for d in weekly.index],
            y=list(order),
            colorscale="Viridis",
            text=weekly[order].T.values,
            texttemplate="%{text}",
            hovertemplate="%{y}, week of %{x}: %{z} hard sets<extra></extra>",
        )
    )
    return _style(fig, height=380, title="Hard sets per muscle, last 12 weeks")


def _exercise_chart(series, history, prs, label):
    fig = go.Figure()
    for venue, g in series.groupby("Venue"):
        color = VENUE_COLORS.get(venue, ACCENT)
        fig.add_trace(
            go.Scatter(
                x=g["dt"],
                y=g["e1RM"],
                mode="lines+markers",
                name=venue,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=8),
                text=g["Sets"],
                hovertemplate="%{x|%d %b}<br>%{text}<br>e1RM %{y:.1f}<extra></extra>",
            )
        )
    if not prs.empty:
        fig.add_trace(
            go.Scatter(
                x=prs["Date"],
                y=prs["e1RM"],
                mode="markers",
                name="PR",
                marker=dict(symbol="star", size=14, color="#facc15"),
            )
        )
    return _style(fig, title=f"{label} — estimated 1RM", yaxis_title="e1RM (kg)")


def render_deep_dive(health):
    """Fuller training view for the Deep Dive: cycles, muscles, exercises, weekly overview."""
    st.markdown("---")
    st.subheader("🏋️ Training deep dive")
    log = _load_log()
    if log is None:
        st.caption("Training log unavailable.")
        return
    sessions = tm.session_table(log)
    history = tm.best_sets(log, _bodyweight(health))
    tab_cycles, tab_muscles, tab_exercises, tab_weeks = st.tabs(
        ["🔁 Cycles", "💪 Muscles", "📈 Exercises", "🗓️ Weekly overview"]
    )

    with tab_cycles:
        cycles = tm.cycle_history(sessions)
        current = tm.cycles_in_range(sessions, sessions["dt"].max(), sessions["dt"].max())
        for cycle in current:
            st.markdown(tm.describe_cycle(cycle))
        shown = cycles.assign(
            Start=cycles["Start"].dt.strftime("%d %b"), End=cycles["End"].dt.strftime("%d %b")
        )
        st.dataframe(shown.iloc[::-1], hide_index=True, use_container_width=True)
        st.caption(
            "A cycle is one pass through the five sessions: a repeated session or a "
            "complete cycle starts the next one. Numbers follow the training notes."
        )

    with tab_muscles:
        fresh = tm.muscle_freshness(log, date.today())
        st.markdown("**Freshness** — days since each muscle was last trained")
        st.dataframe(
            fresh.assign(**{"Last trained": fresh["Last trained"].dt.strftime("%d %b")}),
            hide_index=True,
            use_container_width=True,
        )
        weekly = tm.weekly_hard_sets(log).tail(12)
        if not weekly.empty:
            st.plotly_chart(_heatmap(weekly), use_container_width=True)
        st.caption("Hard sets = working sets at RIR 3 or lower, or with no RIR logged.")

    with tab_exercises:
        options = tm.series_options(history)
        if not options:
            st.info("No weighted exercises logged yet.")
        else:
            label = st.selectbox("Exercise", [o[0] for o in options], key="training_exercise")
            _, exercise, scope = next(o for o in options if o[0] == label)
            best = history[(history["exercise"] == exercise) & (history["scope"] == scope)]
            sets = tm.series_sessions(log, exercise, scope)
            series = (
                sets.drop(columns="Best e1RM")
                .merge(best[["dt", "e1rm"]], on="dt")
                .rename(columns={"e1rm": "e1RM"})
            )
            prs = tm.pr_events(best)
            if len(series) >= 2:
                st.plotly_chart(
                    _exercise_chart(series, history, prs, label), use_container_width=True
                )
            else:
                st.info("Only one session so far; log it again to draw a trend.")
            st.dataframe(
                series.assign(
                    Date=series["dt"].dt.strftime("%d %b %Y"), e1RM=series["e1RM"].round(1)
                )[["Date", "Venue", "Sets", "e1RM"]].iloc[::-1],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Free weights are compared across gyms; machines and cables only within "
                "one gym, so each gym is its own line for machines."
            )

    with tab_weeks:
        overview = tm.weekly_overview(log, health)
        fig = go.Figure(
            go.Bar(x=overview.index, y=overview["Hard sets"], name="Hard sets", marker_color=ACCENT)
        )
        fig.add_trace(
            go.Scatter(
                x=overview.index,
                y=overview["Lifts up %"],
                name="Lifts up %",
                yaxis="y2",
                mode="lines+markers",
                line_color=GREEN,
            )
        )
        st.plotly_chart(
            _style(
                fig,
                title="Hard sets and share of lifts that went up",
                yaxis_title="hard sets",
                yaxis2=dict(title="lifts up %", overlaying="y", side="right", range=[0, 100]),
            ),
            use_container_width=True,
        )
        shown = overview.iloc[::-1].copy()
        shown.index = [f"{d:%d %b}" for d in shown.index]
        st.dataframe(shown, use_container_width=True)
        st.caption(
            "Training next to the health data for each Monday week. Deficit is the "
            "nominal WADP value (2,000 + Move − intake) on days with both logged."
        )
