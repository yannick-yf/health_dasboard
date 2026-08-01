"""
Deep Dive — Bulk Journey Tracker

Evidence-based metrics for tracking a clean bulk (lean mass gain) phase.
Research: Ripped Body, Helms et al., PMC10620361, PMC6680710, Levine 2004, Lamon 2021.
"""

import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils.tdee_calculator import apply_to_dataframe
from utils.visualization_helpers import _DARK_LAYOUT
from utils.metrics_helpers import compute_intake_targets
from utils.bodyfat_estimators import estimate_body_fat
from utils.deep_dive_helpers import (
    load_personal_info,
    calculate_enhanced_metrics,
    render_daily_trends,
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
    classify_status,
)

# ── Deep Dive-specific constants ──────────────────────────────────────────────
TARGET_WEIGHT = 76.0    # kg — goal weight for this phase
START_BF_PCT  = 13.0    # % body fat at bulk start (from personal_info)
P_RATIO_LEAN  = 0.80    # lean gain fraction at ~13% BF + target rate
P_RATIO_FAST  = 0.65    # lean fraction if gaining excessively fast
SLEEP_TGT_H   = 7.5     # hours — target sleep (above the shared SLEEP_MIN_H floor)
BF_CEILING    = 17.0    # % BF — end-bulk signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dark_base() -> dict:
    """_DARK_LAYOUT without xaxis/yaxis keys — safe for make_subplots figures."""
    return {k: v for k, v in _DARK_LAYOUT.items() if k not in ("xaxis", "yaxis")}


# ── Entry point ───────────────────────────────────────────────────────────────

def render(df):
    """Render the Bulk Journey Tracker."""
    st.title("🏋️ Bulk Journey Tracker")

    if df.empty:
        st.warning("No data available. Please add records in the Data Entry page.")
        return

    personal_info = load_personal_info()
    has_complete = all(
        k in personal_info and personal_info[k]
        for k in ("birth_date", "height_cm", "sex")
    )
    has_bf = bool(personal_info.get("body_fat_prct"))

    df_e = calculate_enhanced_metrics(df, personal_info, has_complete, has_bf)

    # Shared derived columns — computed once, used by all sections
    df_e["surplus"] = df_e["calories_consumed"] - df_e["calories_burned"]
    df_e["weight_7d_ma"] = df_e["weight"].rolling(7, min_periods=3).mean()
    df_e["is_training"] = df_e["workout_duration_min_tot"] > 30
    df_e["sleep_h"] = df_e["sleep_min"] / 60
    df_e["recovery_risk"] = (df_e["sleep_min"] < 360) & (df_e["workout_duration_min_tot"] > 45)

    # Waist MAs — waist signal is noisier than weight, so we surface both 7d (responsive)
    # and 14d (decision-grade) windows. min_periods=3/5 lets us start showing the line
    # as soon as there are enough points without requiring a full window.
    if "waist_cm" in df_e.columns:
        df_e["waist_7d_ma"] = df_e["waist_cm"].rolling(7, min_periods=3).mean()
        df_e["waist_14d_ma"] = df_e["waist_cm"].rolling(14, min_periods=5).mean()
    else:
        df_e["waist_cm"] = pd.NA
        df_e["waist_7d_ma"] = pd.NA
        df_e["waist_14d_ma"] = pd.NA

    # 4-week rolling gain rate: kg change in rolling avg over 28 days / 4 weeks
    df_e["gain_rate_4w"] = (df_e["weight_7d_ma"] - df_e["weight_7d_ma"].shift(28)) / 4.0

    # Bulk phase slice
    bulk = df_e[df_e["date"] >= BULK_START].copy().reset_index(drop=True)
    if len(bulk) < 7:
        st.info(
            f"Not enough bulk-phase data yet (fewer than 7 days since {BULK_START.strftime('%B %d, %Y')}). "
            "Showing all program data below."
        )
        bulk = df_e.copy().reset_index(drop=True)

    bulk["days_elapsed"] = (bulk["date"] - BULK_START).dt.days.clip(lower=0)
    bulk["week_start"] = bulk["date"].dt.to_period("W").dt.start_time

    # ── Render sections ───────────────────────────────────────────────────────
    _render_status_banner(bulk)
    _render_kpi_cards(bulk)
    _render_weekly_targets(bulk)
    _render_warning_signals(bulk)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_chart_weight_trajectory(bulk), use_container_width=True)
    with col_b:
        st.plotly_chart(_chart_lean_fat_estimate(bulk, personal_info), use_container_width=True)

    # Waist + BF estimates — only rendered if any waist data exists in the bulk window
    if bulk["waist_cm"].notna().any():
        _render_waist_section(bulk, personal_info)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(_chart_surplus_split(bulk), use_container_width=True)
    with col_d:
        st.plotly_chart(_chart_sleep_recovery(bulk), use_container_width=True)

    _render_neat_surplus(bulk)

    st.markdown("---")
    _render_tdee_section(bulk)

    render_daily_trends(df_e, has_complete, has_bf)


# ── Section 1: Status Banner ──────────────────────────────────────────────────

def _render_status_banner(bulk):
    rate_series = bulk["gain_rate_4w"].dropna()
    latest_rate = float(rate_series.iloc[-1]) if len(rate_series) else None
    status_label, status_color = classify_status(latest_rate)

    current_w = bulk["weight"].dropna().iloc[-1] if bulk["weight"].notna().any() else None
    weeks_in = int((bulk["date"].max() - BULK_START).days / 7)

    if current_w and latest_rate and latest_rate > 0.01:
        remaining_kg = max(0, TARGET_WEIGHT - current_w)
        weeks_to = f"~{remaining_kg / latest_rate:.0f} wks to {TARGET_WEIGHT} kg" if remaining_kg > 0 else "Target reached 🎯"
    else:
        weeks_to = "—"

    rate_str = f"{latest_rate:+.2f} kg/wk" if latest_rate is not None else "—"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, {status_color}22 100%);
            border-left: 4px solid {status_color};
            padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">
                        Bulk Phase · Week {weeks_in}
                    </div>
                    <div style="color:{status_color}; font-size:1.15rem; font-weight:700; margin-top:0.2rem;">
                        {status_label}
                    </div>
                </div>
                <div style="display:flex; gap:2rem;">
                    <div style="text-align:center;">
                        <div style="color:#9ca3af; font-size:0.7rem;">4-WEEK RATE</div>
                        <div style="color:#f0f2f6; font-size:1.1rem; font-weight:700;">{rate_str}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#9ca3af; font-size:0.7rem;">TARGET</div>
                        <div style="color:#f0f2f6; font-size:1.1rem; font-weight:700;">{weeks_to}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Section 2: KPI Cards ──────────────────────────────────────────────────────

def _render_kpi_cards(bulk):
    st.subheader("📋 Bulk Phase KPIs")

    # 7d avg weight + delta vs prior 7d
    curr_7d = bulk["weight"].tail(7).mean()
    prev_7d = bulk["weight"].tail(14).head(7).mean()
    weight_delta = f"{curr_7d - prev_7d:+.2f} kg vs prior wk" if not pd.isna(prev_7d) else None

    # 4-week gain rate
    rate_series = bulk["gain_rate_4w"].dropna()
    latest_rate = float(rate_series.iloc[-1]) if len(rate_series) else None

    # Total gained since bulk start (rolling avg vs first rolling avg)
    start_rw = bulk["weight_7d_ma"].dropna().iloc[0] if bulk["weight_7d_ma"].notna().any() else None
    curr_rw = bulk["weight_7d_ma"].dropna().iloc[-1] if bulk["weight_7d_ma"].notna().any() else None
    total_gained = (curr_rw - start_rw) if (start_rw and curr_rw) else None

    # 7-day mean surplus
    mean_surplus_7d = bulk["surplus"].tail(7).mean()

    # Sleep score (% nights ≥7h, whole bulk)
    sleep_score = (bulk["sleep_h"] >= SLEEP_MIN_H).mean() * 100 if bulk["sleep_h"].notna().any() else None

    # Training consistency
    training_pct = bulk["is_training"].mean() * 100 if not bulk.empty else None

    cols = st.columns(6)
    cols[0].metric(
        "7d Avg Weight",
        f"{curr_7d:.2f} kg" if not pd.isna(curr_7d) else "N/A",
        delta=weight_delta,
        help="7-day rolling mean — filters water & glycogen noise",
    )
    cols[1].metric(
        "4w Gain Rate",
        f"{latest_rate:+.2f} kg/wk" if latest_rate is not None else "N/A",
        delta=f"target: +{GAIN_MIN}–{GAIN_MAX} kg/wk",
        delta_color="off",
        help="Change in 7d rolling avg over 28 days ÷ 4. Target: 0.12–0.18 kg/wk.",
    )
    cols[2].metric(
        "Total Gained",
        f"{total_gained:+.1f} kg" if total_gained is not None else "N/A",
        help=f"Cumulative since {BULK_START.strftime('%b %d, %Y')} (based on rolling avg)",
    )
    cols[3].metric(
        "7d Mean Surplus",
        f"{mean_surplus_7d:+.0f} kcal" if not pd.isna(mean_surplus_7d) else "N/A",
        delta=f"target: +{SURPLUS_LO}–{SURPLUS_HI} kcal",
        delta_color="inverse",
        help="Average daily surplus over last 7 days. Target: 200–300 kcal/day.",
    )
    cols[4].metric(
        "Sleep Score",
        f"{sleep_score:.0f}%" if sleep_score is not None else "N/A",
        help=f"% of nights with ≥{SLEEP_MIN_H:.0f}h sleep since bulk start",
    )
    cols[5].metric(
        "Training Days",
        f"{training_pct:.0f}%" if training_pct is not None else "N/A",
        help="% of bulk days with workout > 30 min",
    )


# ── Section 2b: Weekly Calorie Targets ───────────────────────────────────────

def _render_weekly_targets(bulk):
    """
    Auto-updating calorie target table.

    Rule: empirical TDEE is back-calculated from the last 28 days of weight trend
    and average calorie intake. Target = empirical TDEE + 250 kcal surplus,
    periodized ±100 kcal for training vs rest days.
    """
    targets = compute_intake_targets(bulk, bulk_start_ts=BULK_START)
    if targets is None:
        return

    rate     = targets["gain_rate_4w"]
    status   = targets["status"]
    change   = targets["change_from_current"]

    # Status-aware explanation
    if status == "stalling":
        explanation = (
            f"Weight flat ({rate:+.3f} kg/wk over 28 days) → "
            f"empirical TDEE ≈ avg consumed ({targets['avg_consumed']:,} kcal). "
            f"Increase by **{abs(change):,} kcal/day** to hit the +250 kcal surplus target."
        )
        banner_color = "#f59e0b"
    elif status == "gaining" and rate is not None and rate >= GAIN_MIN:
        explanation = (
            f"On track ({rate:+.3f} kg/wk) → "
            f"estimated TDEE {targets['empirical_tdee']:,} kcal. "
            "Maintain current intake."
        )
        banner_color = "#10b981"
    elif status == "gaining" and rate is not None and rate < GAIN_MIN:
        explanation = (
            f"Gaining slowly ({rate:+.3f} kg/wk, target {GAIN_MIN}–{GAIN_MAX}) → "
            f"add **{abs(change):,} kcal/day** to reach the target surplus."
        )
        banner_color = "#f59e0b"
    elif status == "gaining" and rate is not None and rate > GAIN_WARN_HIGH:
        explanation = (
            f"Gaining too fast ({rate:+.3f} kg/wk > {GAIN_WARN_HIGH} limit) → "
            f"reduce by **{abs(change):,} kcal/day** to limit fat accretion."
        )
        banner_color = "#ef4444"
    else:
        explanation = f"Based on last 28 days. Estimated TDEE: {targets['empirical_tdee']:,} kcal/day."
        banner_color = "#667eea"

    st.subheader("🍽️ This Week's Calorie Targets")
    st.caption(explanation)

    col_train, col_rest, col_avg, col_tdee = st.columns(4)

    col_train.markdown(
        f"""
        <div style="
            background:#1e293b; border:1px solid #6366f1;
            border-radius:10px; padding:1rem; text-align:center;
        ">
            <div style="color:#9ca3af; font-size:0.72rem; text-transform:uppercase; letter-spacing:.05em;">
                🏋️ Training Day
            </div>
            <div style="color:#6366f1; font-size:1.6rem; font-weight:700; margin:.3rem 0;">
                {targets['training_day']:,}
            </div>
            <div style="color:#6b7280; font-size:0.7rem;">kcal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_rest.markdown(
        f"""
        <div style="
            background:#1e293b; border:1px solid #94a3b8;
            border-radius:10px; padding:1rem; text-align:center;
        ">
            <div style="color:#9ca3af; font-size:0.72rem; text-transform:uppercase; letter-spacing:.05em;">
                🛋️ Rest Day
            </div>
            <div style="color:#94a3b8; font-size:1.6rem; font-weight:700; margin:.3rem 0;">
                {targets['rest_day']:,}
            </div>
            <div style="color:#6b7280; font-size:0.7rem;">kcal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_avg.markdown(
        f"""
        <div style="
            background:#1e293b; border:1px solid {banner_color};
            border-radius:10px; padding:1rem; text-align:center;
        ">
            <div style="color:#9ca3af; font-size:0.72rem; text-transform:uppercase; letter-spacing:.05em;">
                📊 Daily Avg Target
            </div>
            <div style="color:{banner_color}; font-size:1.6rem; font-weight:700; margin:.3rem 0;">
                {targets['daily_avg_target']:,}
            </div>
            <div style="color:#6b7280; font-size:0.7rem;">
                {'+' if change >= 0 else ''}{change:,} vs current
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_tdee.markdown(
        f"""
        <div style="
            background:#1e293b; border:1px solid #475569;
            border-radius:10px; padding:1rem; text-align:center;
        ">
            <div style="color:#9ca3af; font-size:0.72rem; text-transform:uppercase; letter-spacing:.05em;">
                ⚗️ Empirical TDEE
            </div>
            <div style="color:#cbd5e1; font-size:1.6rem; font-weight:700; margin:.3rem 0;">
                {targets['empirical_tdee']:,}
            </div>
            <div style="color:#6b7280; font-size:0.7rem;">kcal/day (28-day avg)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacing


# ── Section 3: Warning Signals ────────────────────────────────────────────────

def _render_warning_signals(bulk):
    rate_series = bulk["gain_rate_4w"].dropna()
    latest_rate = float(rate_series.iloc[-1]) if len(rate_series) else None
    surplus_7d = bulk["surplus"].tail(7).mean()
    sleep_14d_pct = (bulk["sleep_h"].tail(14) >= SLEEP_MIN_H).mean() * 100
    neat_7d_pct_high = (bulk["steps"].tail(7) >= NEAT_HIGH).mean()
    risk_days_14 = int(bulk["recovery_risk"].tail(14).sum())

    # Estimated current BF% from rolling avg
    if bulk["weight_7d_ma"].notna().any():
        start_rw = bulk["weight_7d_ma"].dropna().iloc[0]
        curr_rw = bulk["weight_7d_ma"].dropna().iloc[-1]
        cum_gain = max(0.0, curr_rw - start_rw)
        p = P_RATIO_FAST if (latest_rate and latest_rate > GAIN_WARN_HIGH) else P_RATIO_LEAN
        fat_gained = cum_gain * (1 - p)
        start_fat = start_rw * (START_BF_PCT / 100) if start_rw else 0
        est_bf = (start_fat + fat_gained) / curr_rw * 100 if curr_rw else START_BF_PCT
    else:
        est_bf = START_BF_PCT

    warnings, oks = [], []

    # Rate checks
    if latest_rate is not None:
        if latest_rate > GAIN_WARN_HIGH:
            warnings.append(
                f"⚡ **Gaining too fast** ({latest_rate:+.2f} kg/wk > {GAIN_WARN_HIGH} limit) — "
                "P-ratio likely degraded to ~0.65 (fat-dominant gain). Cut 150–200 kcal/day."
            )
        elif latest_rate < 0:
            warnings.append(
                f"📉 **Weight trending down** ({latest_rate:+.2f} kg/wk) — check calorie logging or "
                "audit NEAT. Add 150–200 kcal/day if underreporting is ruled out."
            )
        elif latest_rate < GAIN_MIN:
            warnings.append(
                f"🐢 **Gaining slowly** ({latest_rate:+.2f} kg/wk vs target {GAIN_MIN}–{GAIN_MAX}) — "
                "consider adding 100–150 kcal/day, prioritising pre/post-workout carbs."
            )
        else:
            oks.append(f"✅ Gain rate on target ({latest_rate:+.2f} kg/wk)")

    # Surplus checks
    if not pd.isna(surplus_7d):
        if surplus_7d > SURPLUS_HI + 200:
            warnings.append(
                f"🍕 **Surplus too high** ({surplus_7d:.0f} kcal/day, target {SURPLUS_LO}–{SURPLUS_HI}) — "
                "excess surplus beyond muscle protein synthesis capacity stores as fat."
            )
        elif surplus_7d < SURPLUS_LO:
            warnings.append(
                f"🔋 **Surplus too low** ({surplus_7d:.0f} kcal/day) — sub-optimal MPS environment. "
                "Add 100–200 kcal on training days via carbohydrates."
            )
        else:
            oks.append(f"✅ Surplus in target range ({surplus_7d:.0f} kcal/day)")

    # Sleep
    if sleep_14d_pct < 50:
        warnings.append(
            f"😴 **Poor sleep** ({sleep_14d_pct:.0f}% of last 14 nights ≥7h) — "
            "MPS is reduced ~18% on sleep-deprived nights (Lamon et al. 2021). Prioritise sleep over training volume."
        )
    elif sleep_14d_pct >= 80:
        oks.append(f"✅ Sleep quality good ({sleep_14d_pct:.0f}% nights ≥7h)")

    # NEAT compensation
    if neat_7d_pct_high >= 0.5 and latest_rate is not None and latest_rate < GAIN_MIN:
        warnings.append(
            f"🦶 **NEAT compensation likely** ({neat_7d_pct_high*100:.0f}% of recent days >12k steps) — "
            "high daily movement can offset 200–400 kcal of your planned surplus. "
            "Add 200 kcal on high-step days (>12k)."
        )

    # Recovery risk
    if risk_days_14 >= 2:
        warnings.append(
            f"⚠️ **{risk_days_14} recovery risk days** in last 14 (sleep <6h + workout >45 min same day) — "
            "cortisol elevation impairs MPS for 18–24h post-workout."
        )

    # BF ceiling
    if est_bf >= BF_CEILING - 1:
        warnings.append(
            f"📊 **Approaching bulk ceiling** (est. BF ~{est_bf:.1f}%, ceiling {BF_CEILING}%) — "
            "consider planning a maintenance or cut phase."
        )
    elif est_bf < BF_CEILING - 3:
        oks.append(f"✅ Body fat well within bulk ceiling (est. ~{est_bf:.1f}%)")

    if not warnings and not oks:
        return

    n_warn = len(warnings)
    icon = "⚠️" if warnings else "✅"
    label = f"{icon} {n_warn} warning(s) — {len(oks)} green signal(s)"

    with st.expander(label, expanded=bool(warnings)):
        for w in warnings:
            st.warning(w)
        for o in oks:
            st.success(o)


# ── Chart 1: Weight Trajectory ────────────────────────────────────────────────

def _chart_weight_trajectory(bulk):
    data = bulk[bulk["weight"].notna()].copy()
    if data.empty:
        return go.Figure()

    start_w = data["weight"].iloc[0]
    days = data["days_elapsed"]
    corridor_lo = start_w + days / 7 * GAIN_MIN
    corridor_hi = start_w + days / 7 * GAIN_MAX

    fig = go.Figure()

    # Shaded target corridor
    fig.add_trace(go.Scatter(
        x=pd.concat([data["date"], data["date"].iloc[::-1]]),
        y=pd.concat([corridor_hi, corridor_lo.iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(16,185,129,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Target corridor",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Raw weight dots
    fig.add_trace(go.Scatter(
        x=data["date"], y=data["weight"],
        mode="markers",
        marker=dict(color="rgba(148,163,184,0.5)", size=4),
        name="Daily weight",
        hovertemplate="%{x|%b %d}: %{y:.2f} kg<extra></extra>",
    ))

    # 7-day rolling average
    fig.add_trace(go.Scatter(
        x=bulk["date"], y=bulk["weight_7d_ma"],
        mode="lines",
        line=dict(color="#667eea", width=2.5),
        name="7d avg",
        hovertemplate="Avg: %{y:.2f} kg<extra></extra>",
    ))

    # 4-week forward projection from last valid MA point
    ma_valid = bulk[bulk["weight_7d_ma"].notna()]
    rate_series = bulk["gain_rate_4w"].dropna()
    if len(ma_valid) >= 7 and len(rate_series):
        last_date = ma_valid["date"].iloc[-1]
        last_ma = ma_valid["weight_7d_ma"].iloc[-1]
        latest_rate = float(rate_series.iloc[-1])
        proj_dates = pd.date_range(last_date, periods=29, freq="D")
        proj_y = [min(last_ma + (i / 7) * latest_rate, TARGET_WEIGHT + 1) for i in range(29)]
        fig.add_trace(go.Scatter(
            x=proj_dates, y=proj_y,
            mode="lines",
            line=dict(color="#f59e0b", width=1.5, dash="dashdot"),
            name="Projection",
            hovertemplate="Proj: %{y:.2f} kg<extra></extra>",
        ))

    # Target weight reference
    fig.add_hline(
        y=TARGET_WEIGHT, line_dash="dot", line_color="#10b981",
        annotation_text=f"Target {TARGET_WEIGHT} kg",
        annotation_font_color="#10b981",
        annotation_position="top left",
    )

    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Weight Trajectory", x=0.5))
    fig.update_yaxes(title_text="Weight (kg)")
    return fig


# ── Chart 2: Lean vs Fat Estimate (Stacked Area) ──────────────────────────────

def _chart_lean_fat_estimate(bulk, personal_info):
    """Cut-aware body-composition tracker.

    Derives fat mass & lean mass each day from the SAME waist-based BF% consensus
    used in the "Waist & Body Fat Estimates" section (RFM / Deurenberg / YMCA), then
    plots how each has CHANGED since the window start. This answers the recomp
    question directly: fat line falling + lean line flat = losing fat while holding
    muscle. Replaces the old bulk-only P-ratio projection, which went inert during a
    cut (it partitioned weight *gains* only, so it froze at the start assumption).

    Uses smoothed inputs (weight & waist 7-day MAs) so daily water noise doesn't
    swing the composition read.
    """
    height_cm = personal_info.get("height_cm")
    age = personal_info.get("age")
    sex = personal_info.get("sex", "Male")

    d = bulk[["date", "weight_7d_ma", "waist_7d_ma"]].copy()
    d = d[d["weight_7d_ma"].notna() & d["waist_7d_ma"].notna()]

    if d.empty or not (height_cm and age):
        fig = go.Figure()
        fig.add_annotation(
            text="Needs waist + weight history<br>(and height / age in personal_info)",
            showarrow=False, font=dict(color="#94a3b8", size=13),
        )
        base = _dark_base()
        base["title"] = dict(text="Lean vs Fat Mass Change", x=0.5)
        fig.update_layout(**base)
        return fig

    def _row_bf(r):
        est = estimate_body_fat(
            waist_cm=float(r["waist_7d_ma"]),
            weight_kg=float(r["weight_7d_ma"]),
            height_cm=float(height_cm),
            age_yrs=int(age),
            sex=sex,
        )
        return est["consensus"]

    d["bf_pct"] = d.apply(_row_bf, axis=1)
    d = d[d["bf_pct"].notna()]
    if len(d) < 3:
        return go.Figure()

    d["fat_mass"] = d["weight_7d_ma"] * d["bf_pct"] / 100.0
    d["lean_mass"] = d["weight_7d_ma"] - d["fat_mass"]
    d["fat_delta"] = d["fat_mass"] - d["fat_mass"].iloc[0]
    d["lean_delta"] = d["lean_mass"] - d["lean_mass"].iloc[0]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=d["date"], y=d["fat_delta"],
        mode="lines", name="Δ Fat mass",
        line=dict(color="#ef4444", width=2.5),
        hovertemplate="Δ Fat: %{y:+.2f} kg<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=d["date"], y=d["lean_delta"],
        mode="lines", name="Δ Lean mass",
        line=dict(color="#6366f1", width=2.5),
        hovertemplate="Δ Lean: %{y:+.2f} kg<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=d["date"], y=d["bf_pct"],
        mode="lines", name="Est. BF%",
        line=dict(color="#f59e0b", width=1.6, dash="dash"),
        hovertemplate="BF: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(
        y=0, secondary_y=False, line_dash="dot",
        line_color="rgba(255,255,255,0.25)",
    )

    # Recomp verdict badge: net fat & lean change over the window
    fat_chg = float(d["fat_delta"].iloc[-1])
    lean_chg = float(d["lean_delta"].iloc[-1])
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, align="left",
        text=f"Net: Fat {fat_chg:+.1f} kg · Lean {lean_chg:+.1f} kg",
        font=dict(color="#cbd5e1", size=12),
        bgcolor="rgba(15,23,42,0.65)", borderpad=4,
    )

    base = _dark_base()
    base["title"] = dict(text="Lean vs Fat Mass Change (from waist BF%)", x=0.5)
    fig.update_layout(**base)
    fig.update_xaxes(**_DARK_LAYOUT["xaxis"], title_text="Date")
    fig.update_yaxes(**_DARK_LAYOUT["yaxis"], title_text="Δ mass (kg)", secondary_y=False)
    fig.update_yaxes(title_text="Est. BF%", secondary_y=True, showgrid=False)
    return fig


# ── Chart 3: Surplus — Training vs Rest Days ─────────────────────────────────

def _chart_surplus_split(bulk):
    data = bulk[bulk["surplus"].notna()].copy()
    if data.empty:
        return go.Figure()

    def _week_surplus(group):
        train = group.loc[group["is_training"], "surplus"].mean()
        rest = group.loc[~group["is_training"], "surplus"].mean()
        return pd.Series({"train": train, "rest": rest})

    weekly = data.groupby("week_start").apply(_week_surplus).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly["week_start"], y=weekly["train"],
        name="Training days", marker_color="#6366f1",
        hovertemplate="Train: %{y:.0f} kcal<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=weekly["week_start"], y=weekly["rest"],
        name="Rest days", marker_color="#94a3b8",
        hovertemplate="Rest: %{y:.0f} kcal<extra></extra>",
    ))

    # Target surplus band
    fig.add_hrect(
        y0=SURPLUS_LO, y1=SURPLUS_HI,
        fillcolor="rgba(16,185,129,0.08)", line_width=0,
        annotation_text=f"+{SURPLUS_LO}–{SURPLUS_HI} target",
        annotation_position="top right",
        annotation_font_color="#10b981",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)")

    fig.update_layout(
        **_DARK_LAYOUT,
        barmode="group",
        title=dict(text="Surplus: Training vs Rest Days", x=0.5),
    )
    fig.update_yaxes(title_text="Avg Daily Surplus (kcal)")
    return fig


# ── Chart 4: Sleep Adequacy & Recovery Risk ───────────────────────────────────

def _chart_sleep_recovery(bulk):
    data = bulk.copy()

    weekly = data.groupby("week_start").agg(
        sleep_pct=("sleep_h", lambda x: (x >= SLEEP_MIN_H).mean() * 100),
        risk_days=("recovery_risk", "sum"),
    ).reset_index()

    bar_colors = [
        "#10b981" if v >= 70 else "#f59e0b" if v >= 50 else "#ef4444"
        for v in weekly["sleep_pct"]
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=weekly["week_start"], y=weekly["sleep_pct"],
        name=f"Nights ≥{SLEEP_MIN_H:.0f}h (%)",
        marker_color=bar_colors,
        hovertemplate="%{y:.0f}% nights ≥7h<extra></extra>",
    ), secondary_y=False)

    if weekly["risk_days"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=weekly["week_start"], y=weekly["risk_days"],
            mode="lines+markers", name="Recovery risk days",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=7, symbol="x", color="#ef4444"),
            hovertemplate="%{y} risk days<extra></extra>",
        ), secondary_y=True)

    fig.add_hline(
        y=70, line_dash="dot", line_color="#10b981",
        annotation_text="70% floor",
        annotation_font_color="#10b981",
    )

    base = _dark_base()
    base["title"] = dict(text="Sleep Adequacy & Recovery Risk", x=0.5)
    fig.update_layout(**base)
    fig.update_xaxes(**_DARK_LAYOUT["xaxis"], title_text="Week")
    fig.update_yaxes(**_DARK_LAYOUT["yaxis"], title_text="Nights ≥7h (%)", range=[0, 105], secondary_y=False)
    fig.update_yaxes(title_text="Risk days", showgrid=False, rangemode="tozero", secondary_y=True)
    return fig


# ── Chart 5: NEAT × Surplus Interaction ──────────────────────────────────────

def _chart_neat_surplus(bulk):
    scatter = bulk[bulk["steps"].notna() & bulk["surplus"].notna()].copy()
    if len(scatter) < 5:
        return go.Figure()

    # Color by 7-day weight trend
    trend = scatter["weight_7d_ma"].diff(7)

    def _color(v):
        if pd.isna(v):
            return "#94a3b8"
        if v > 0.10:
            return "#10b981"
        if v < -0.10:
            return "#ef4444"
        return "#f59e0b"

    scatter["pt_color"] = trend.apply(_color)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scatter["steps"], y=scatter["surplus"],
        mode="markers",
        marker=dict(
            color=scatter["pt_color"],
            size=7, opacity=0.75,
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
        ),
        customdata=scatter["date"].dt.strftime("%b %d"),
        hovertemplate="%{customdata}<br>Steps: %{x:,}<br>Surplus: %{y:+.0f} kcal<extra></extra>",
        name="Daily data",
    ))

    # NEAT high threshold
    bulk_ts = pd.Timestamp(NEAT_HIGH)  # just for annotation; x is numeric steps
    fig.add_vline(
        x=NEAT_HIGH, line_dash="dot", line_color="rgba(245,158,11,0.55)",
        annotation_text="High NEAT (12k)",
        annotation_position="top right",
        annotation_font_color="#f59e0b",
    )
    # Target surplus band
    fig.add_hrect(
        y0=SURPLUS_LO, y1=SURPLUS_HI,
        fillcolor="rgba(16,185,129,0.08)", line_width=0,
    )
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.15)")
    fig.add_hline(
        y=SURPLUS_LO, line_dash="dot", line_color="rgba(16,185,129,0.4)",
        annotation_text=f"+{SURPLUS_LO}", annotation_position="top left",
        annotation_font_color="#10b981",
    )
    fig.add_hline(
        y=SURPLUS_HI, line_dash="dot", line_color="rgba(16,185,129,0.4)",
        annotation_text=f"+{SURPLUS_HI}", annotation_position="top left",
        annotation_font_color="#10b981",
    )

    layout = dict(_DARK_LAYOUT)
    layout["hovermode"] = "closest"
    layout["title"] = dict(text="NEAT vs Surplus — do high-step days kill your bulk?", x=0.5)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Daily Steps")
    fig.update_yaxes(title_text="Daily Surplus (kcal)")
    return fig


def _render_neat_surplus(bulk):
    fig = _chart_neat_surplus(bulk)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Point color: 🟢 gaining (7d avg rising) · 🟡 stable · 🔴 losing. "
            "Green band = target surplus zone (+200–300 kcal). "
            "Points right of the dashed line likely had NEAT offset their caloric surplus."
        )


# ── Section: TDEE Comparison ──────────────────────────────────────────────────

def _render_tdee_section(bulk):
    st.subheader("⚗️ TDEE: Apple Watch vs Science")

    available_weeks = sorted(bulk["week_start"].unique(), reverse=True)
    if not len(available_weeks):
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        week_options = [
            f"{w.strftime('%b %d')} – {(w + pd.Timedelta(days=6)).strftime('%b %d')}"
            for w in available_weeks
        ]
        idx = st.selectbox(
            "Select week",
            range(len(week_options)),
            format_func=lambda x: week_options[x],
        )
    sel_week = available_weeks[idx]
    week_data = bulk[bulk["week_start"] == sel_week].copy()

    _render_tdee_comparison(week_data)


def _render_tdee_comparison(week_data):
    try:
        df_tdee = apply_to_dataframe(week_data)
    except Exception as e:
        st.warning(f"Could not compute science TDEE: {e}")
        return

    valid = df_tdee[df_tdee["tdee_science"].notna() & df_tdee["calories_burned"].notna()].copy()
    if valid.empty:
        st.info("Not enough data to compute science TDEE for this week.")
        return

    mean_watch = valid["calories_burned"].mean()
    mean_science = valid["tdee_science"].mean()
    delta = mean_science - mean_watch

    c1, c2 = st.columns(2)
    c1.metric("Apple Watch TDEE (mean)", f"{mean_watch:.0f} kcal/day")
    c2.metric(
        "Science TDEE (mean)",
        f"{mean_science:.0f} kcal/day",
        delta=f"{delta:+.0f} kcal vs Watch",
    )

    if abs(delta) > 50 and "tdee_delta" in valid.columns and valid["tdee_delta"].notna().any():
        direction = "underestimates" if delta > 0 else "overestimates"
        dominant = valid.loc[valid["tdee_delta"].abs().idxmax(), "date"].strftime("%a %b %d")
        st.caption(
            f"Watch {direction} by ~{abs(delta):.0f} kcal on average. "
            f"Largest gap: {dominant}."
        )

    components = [
        ("tdee_bmr", "BMR", "#6366f1"),
        ("tdee_eat_gym", "EAT Gym", "#f97316"),
        ("tdee_eat_bike", "EAT Bike", "#14b8a6"),
        ("tdee_neat", "NEAT", "#22c55e"),
        ("tdee_tef", "TEF", "#a78bfa"),
    ]

    fig = go.Figure()
    for col_name, label, color in components:
        if col_name in valid.columns:
            fig.add_trace(go.Bar(
                x=valid["date"], y=valid[col_name],
                name=label, marker_color=color,
                hovertemplate=f"{label}: %{{y:.0f}} kcal<extra></extra>",
            ))
    fig.add_trace(go.Scatter(
        x=valid["date"], y=valid["calories_burned"],
        mode="lines+markers", name="Apple Watch",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        marker=dict(size=6),
        hovertemplate="Watch: %{y:.0f} kcal<extra></extra>",
    ))

    layout = dict(_DARK_LAYOUT)
    layout["barmode"] = "stack"
    layout["title"] = dict(text="Daily TDEE Breakdown", x=0.5)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="kcal")
    st.plotly_chart(fig, use_container_width=True)


# ── Section: Waist & Body Fat Estimates ───────────────────────────────────────

def _render_waist_section(bulk, personal_info):
    """Waist trajectory chart + BF % estimate cards.

    Only rendered when at least one waist measurement exists in the bulk window.
    BF estimates use the latest non-null waist + the latest non-null weight,
    paired with height/age/sex from personal_info.json.
    """
    st.markdown("---")
    st.subheader("📏 Waist & Body Fat Estimates")

    col_chart, col_cards = st.columns([2, 1])

    with col_chart:
        st.plotly_chart(_chart_waist_trajectory(bulk), use_container_width=True)

    with col_cards:
        _render_bf_estimate_cards(bulk, personal_info)


def _chart_waist_trajectory(bulk):
    data = bulk[bulk["waist_cm"].notna()].copy()
    if data.empty:
        return go.Figure()

    fig = go.Figure()

    # Daily measurements (dots)
    fig.add_trace(go.Scatter(
        x=data["date"], y=data["waist_cm"],
        mode="markers",
        marker=dict(color="rgba(148,163,184,0.55)", size=5),
        name="Daily",
        hovertemplate="%{x|%b %d}: %{y:.1f} cm<extra></extra>",
    ))

    # 7-day MA — responsive signal
    fig.add_trace(go.Scatter(
        x=bulk["date"], y=bulk["waist_7d_ma"],
        mode="lines",
        line=dict(color="#67e8f9", width=2),
        name="7d MA",
        hovertemplate="7d MA: %{y:.2f} cm<extra></extra>",
    ))

    # 14-day MA — decision-grade signal (bolder)
    fig.add_trace(go.Scatter(
        x=bulk["date"], y=bulk["waist_14d_ma"],
        mode="lines",
        line=dict(color="#10b981", width=3),
        name="14d MA",
        hovertemplate="14d MA: %{y:.2f} cm<extra></extra>",
    ))

    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Waist Trajectory", x=0.5))
    fig.update_yaxes(title_text="Waist (cm)")
    return fig


def _render_bf_estimate_cards(bulk, personal_info):
    """Show RFM / Deurenberg / YMCA / Consensus cards using latest waist + weight."""
    # Use latest non-null waist + latest non-null weight (they may be on different days)
    latest_waist_row = bulk[bulk["waist_cm"].notna()].tail(1)
    latest_weight_row = bulk[bulk["weight"].notna()].tail(1)
    if latest_waist_row.empty or latest_weight_row.empty:
        st.info("Need at least one waist + one weight reading for BF estimates.")
        return

    waist_cm = float(latest_waist_row["waist_cm"].iloc[0])
    weight_kg = float(latest_weight_row["weight"].iloc[0])
    waist_date = latest_waist_row["date"].iloc[0]
    weight_date = latest_weight_row["date"].iloc[0]

    height_cm = personal_info.get("height_cm")
    age = personal_info.get("age")
    sex = personal_info.get("sex", "Male")
    if not (height_cm and age):
        st.warning("Personal info missing — set height + age in personal_info.json to enable BF estimates.")
        return

    est = estimate_body_fat(
        waist_cm=waist_cm,
        weight_kg=weight_kg,
        height_cm=float(height_cm),
        age_yrs=int(age),
        sex=sex,
    )

    def _card(label, value, sub, accent):
        if value is None:
            value_str = "—"
        else:
            value_str = f"{value:.1f}%"
        st.markdown(
            f"""
            <div style="
                background:#0f172a; border-left:3px solid {accent};
                border-radius:6px; padding:0.6rem 0.85rem; margin-bottom:0.5rem;
            ">
                <div style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
                <div style="color:#f0f2f6; font-size:1.4rem; font-weight:700;">{value_str}</div>
                <div style="color:#64748b; font-size:0.7rem;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _card("Consensus", est["consensus"], "mean of 3 formulas (bold)", "#10b981")
    _card("RFM (Woolcott 2018)", est["rfm"], "height + waist", "#67e8f9")
    _card("Deurenberg (1991)", est["deurenberg"], "BMI + age", "#a78bfa")
    _card("YMCA (Wallace 1991)", est["ymca"], "waist + weight", "#f59e0b")

    # Mass breakdown
    if est["lean_mass_kg"] is not None and est["fat_mass_kg"] is not None:
        st.markdown(
            f"""
            <div style="
                background:#1e293b; border-radius:6px; padding:0.6rem 0.85rem; margin-top:0.5rem;
                font-size:0.75rem; color:#94a3b8; line-height:1.5;
            ">
                <b style="color:#e2e8f0;">Mass split</b> (from consensus):<br>
                Lean: <b style="color:#10b981;">{est['lean_mass_kg']:.1f} kg</b> ·
                Fat: <b style="color:#f59e0b;">{est['fat_mass_kg']:.1f} kg</b><br>
                <span style="color:#64748b;">
                    Waist {waist_cm:.1f} cm ({waist_date.strftime('%b %d')}) ·
                    Weight {weight_kg:.1f} kg ({weight_date.strftime('%b %d')})
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
