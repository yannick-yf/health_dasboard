"""
Training Log page — session history, weekly volume, per-exercise progression.

Reads `data/training_log.csv` (populated by the phone app export + the merge
script scripts/merge_tracker_export.py). Everything here is computed from real
logged sessions — NO mocked/sample data. If the log is empty, the page says so.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ACCENT = "#667eea"
ACCENT2 = "#f59e0b"

# Ordered keyword → muscle rules. First match wins, so order matters
# (e.g. "overhead triceps" must hit Triceps before Front delts; "wrist curl"
# must hit Forearms before Biceps). This classifies REAL exercise names; it is
# metadata, not fabricated data.
MUSCLE_RULES = [
    (["jackknife", "crunch", "russian twist", "sit-up", "situp", "plank",
      "abdo", "leg raise", "levé de jambe", "leve de jambe", "toes-to-bar",
      "dead bug", "wood chop", "pallof"], "Abs"),
    (["mollet", "calf", "calv"], "Calves"),
    (["wrist", "avant-bras", "forearm"], "Forearms"),
    (["triceps"], "Triceps"),
    (["face pull", "rear delt", "oiseau", "reverse pec", "arrière épaule",
      "arriere epaule"], "Rear delts"),
    (["élévation latérale", "elevation laterale", "élévations latérales",
      "elevations laterales", "lateral raise", "side delt"], "Side delts"),
    (["ohp", "militaire", "shoulder press", "overhead press"], "Front delts"),
    # Leg rules MUST come before the biceps "curl" rule — "leg curl" contains "curl".
    (["leg curl", "ischio", "leg-curl"], "Hamstrings"),
    (["deadlift", "soulevé de terre", "souleve de terre", "sdt", "roumain",
      "romanian", "pull-through", "pull through"], "Hamstrings"),
    (["leg extension"], "Quads"),
    (["squat", "hack", "bulgarian", "fente", "lunge", "presse", "leg press"], "Quads"),
    (["curl"], "Biceps"),
    (["shrug", "trapèze", "trapeze"], "Traps"),
    (["couché", "couche", "incliné", "incline", "écarté", "ecarte", "dips",
      "pec", "bench", "fly"], "Chest"),
    (["tirage", "rowing", "row", "traction", "pull-up", "pull up", "pull-over",
      "pullover", "pulldown"], "Back"),
]


def _muscle(ex):
    e = str(ex).lower()
    for kws, m in MUSCLE_RULES:
        if any(k in e for k in kws):
            return m
    return "Other"


def _epley(w, reps):
    if pd.isna(w) or pd.isna(reps) or w <= 0:
        return None
    return w * (1 + reps / 30)


def _load_training():
    """Training log lives beside health_data.csv."""
    base = os.path.dirname(st.session_state.get("csv_path", "data/health_data.csv"))
    path = os.path.join(base, "training_log.csv")
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path)
    if t.empty:
        return t
    t["dt"] = pd.to_datetime(t["date"], format="%d/%m/%Y", errors="coerce")
    for c in ["weight", "reps", "rir", "set_number", "exercise_order"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["muscle"] = t["exercise"].apply(_muscle)
    return t


def render(df):
    st.title("🏋️ Training Log")

    t = _load_training()
    if t is None:
        st.warning("No `data/training_log.csv` found. Log sessions in the phone app, "
                   "export the JSON to iCloud, and run `scripts/merge_tracker_export.py`.")
        return
    if t.empty:
        st.info("Training log is empty. Nothing to show yet.")
        return

    sess = (t[["date", "session", "dt"]].drop_duplicates()
            .sort_values("dt").reset_index(drop=True))
    st.caption(f"**{len(sess)}** sessions logged · "
               f"{sess['dt'].min():%d %b %Y} → {sess['dt'].max():%d %b %Y} · "
               f"{t.shape[0]} total sets")

    tab_hist, tab_vol, tab_prog = st.tabs(
        ["📋 Session history", "📊 Weekly volume", "📈 Exercise progression"])

    with tab_hist:
        _render_history(t, sess)
    with tab_vol:
        _render_volume(t)
    with tab_prog:
        _render_progression(t)


def _render_history(t, sess):
    labels = [f"{r.date} · {r.session}" for r in sess.itertuples()]
    pick = st.selectbox("Session", labels[::-1])  # newest first
    d, s = pick.split(" · ", 1)
    block = t[(t["date"] == d) & (t["session"] == s)].sort_values(["exercise_order", "set_number"])

    n_sets = len(block)
    tonnage = float((block["weight"].fillna(0) * block["reps"].fillna(0)).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Exercises", block["exercise"].nunique())
    c2.metric("Working sets", n_sets)
    c3.metric("Tonnage (kg·reps)", f"{tonnage:,.0f}")

    for ex, g in block.groupby("exercise", sort=False):
        tgt = g["target"].iloc[0]
        note = g["note"].iloc[0]
        head = f"**{ex}**  ·  target {tgt}"
        if isinstance(note, str) and note.strip():
            head += f"  ·  _{note}_"
        st.markdown(head)
        rows = []
        for r in g.itertuples():
            w = "BW" if (pd.isna(r.weight) or r.weight == 0) else f"{r.weight:g} kg"
            rir = "" if pd.isna(r.rir) else f"{r.rir:g}"
            rows.append({"Set": int(r.set_number), "Weight": w,
                         "Reps": "" if pd.isna(r.reps) else int(r.reps), "RIR": rir})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_volume(t):
    st.subheader("Sets per muscle per week")
    st.caption("One row = one working set. Week starts Monday.")
    tt = t.dropna(subset=["dt"]).copy()
    tt["week"] = tt["dt"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    pivot = (tt.groupby(["week", "muscle"]).size()
             .unstack(fill_value=0).sort_index())
    if pivot.empty:
        st.info("Not enough data yet.")
        return

    # Most-recent week as a bar chart
    latest = pivot.index.max()
    wk = pivot.loc[latest].sort_values(ascending=False)
    wk = wk[wk > 0]
    fig = go.Figure(go.Bar(x=wk.index.tolist(), y=wk.values.tolist(),
                           marker_color=ACCENT))
    fig.update_layout(title=f"Week of {latest:%d %b %Y}", height=320,
                      margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_title="working sets")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**All logged weeks** (sets per muscle)")
    disp = pivot.copy()
    disp.index = [f"{d:%d %b}" for d in disp.index]
    st.dataframe(disp, use_container_width=True)


def _exercise_series(g):
    """Ordered [{dt,val,label}] of best set per session for one exercise, + is_bw flag."""
    is_bw = g["weight"].fillna(0).le(0).all()
    out = []
    for d, day in g.groupby("dt"):
        if is_bw:
            v = day["reps"].max()
            if pd.isna(v):
                continue
            lab = f"{int(v)} reps"
        else:
            e1 = day.apply(lambda r: _epley(r["weight"], r["reps"]), axis=1)
            if not e1.notna().any():
                continue
            v = e1.max()
            b = day.loc[e1.idxmax()]
            lab = f"{b['weight']:g}kg×{int(b['reps'])} (~{v:.0f})"
        out.append({"dt": d, "val": float(v), "label": lab})
    out.sort(key=lambda o: o["dt"])
    return out, is_bw


def _render_progression(t):
    st.subheader("Progress overview")
    st.caption("Best set each session — **e1RM (kg)** for weighted lifts, "
               "**top-set reps** for bodyweight. Δ = change vs your previous session of that lift.")

    rows = []
    for ex, g in t.groupby("exercise"):
        series, is_bw = _exercise_series(g)
        if not series:
            continue
        latest, n = series[-1], len(series)
        unit = "reps" if is_bw else "kg"
        if n >= 2:
            d = latest["val"] - series[-2]["val"]
            arrow = "▲" if d > 0.5 else ("▼" if d < -0.5 else "=")
            delta = f"{arrow} {d:+.0f} {unit}"
        else:
            delta = "🆕 new"
        rows.append({"Muscle": g["muscle"].iloc[0], "Exercise": ex, "Sess": n,
                     "Last": latest["dt"].strftime("%d %b"),
                     "Best set": latest["label"], "Δ vs prev": delta})
    if not rows:
        st.info("No numeric data yet.")
        return
    ov = pd.DataFrame(rows).sort_values(["Muscle", "Exercise"]).reset_index(drop=True)
    st.dataframe(ov, hide_index=True, use_container_width=True)
    st.caption("Most lifts show 🆕 until you repeat the session — trends fill in as you log more.")

    st.markdown("---")
    st.subheader("Drill-down")
    # default the picker to an exercise that actually has a trend, if any
    multi = [ex for ex, g in t.groupby("exercise") if len(_exercise_series(g)[0]) >= 2]
    opts = t["exercise"].drop_duplicates().tolist()
    idx = opts.index(multi[0]) if multi else 0
    ex = st.selectbox("Exercise", opts, index=idx)
    g = t[t["exercise"] == ex].copy()
    series, is_bw = _exercise_series(g)
    metric = "Top-set reps (BW)" if is_bw else "Estimated 1RM (kg)"
    if len(series) < 2:
        one = series[0] if series else None
        st.info(f"Only one session for **{ex}** so far"
                + (f" ({one['dt']:%d %b}: {one['label']})." if one else ".")
                + " Log it again to draw a trend.")
        return
    prog = pd.DataFrame(series)
    fig = go.Figure(go.Scatter(x=prog["dt"], y=prog["val"], mode="lines+markers",
                               line=dict(color=ACCENT, width=2), marker=dict(size=8),
                               text=prog["label"],
                               hovertemplate="%{x|%d %b}<br>%{text}<extra></extra>"))
    fig.update_layout(title=f"{ex} — {metric}", height=340,
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title=metric)
    st.plotly_chart(fig, use_container_width=True)
    first, last = series[0], series[-1]
    unit = "reps" if is_bw else "kg e1RM"
    st.caption(f"{first['dt']:%d %b} → {last['dt']:%d %b}: "
               f"{first['val']:.0f} → {last['val']:.0f} {unit} ({last['val'] - first['val']:+.0f})")
