"""Training-log analytics: cycles, venue-aware progression and weekly volume.

Comparison rules come from docs/context/02_training_program.md: barbell and dumbbell loads
compare across venues, machine and cable loads only within one venue, bench counts only from
the unassisted baseline, and e1RM uses the Epley formula.
"""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_FAMILIES = ["Upper A", "Lower A", "Upper B", "Upper C", "Lower B"]

# Cycles are one pass through the five sessions. A cycle starts at the first session, when a
# session repeats within it, or once it holds all five. These dates are the cycles Yannick's
# notes started by hand (4-day trial, incomplete cycle, re-sequenced week), and these sessions
# are TTL equipment tests inside cycle 11 that never start a cycle of their own.
KNOWN_CYCLE_STARTS = [date(2026, 9, 14), date(2026, 9, 19), date(2026, 9, 25)]
EXTRA_SESSION_DATES = [date(2026, 9, 29), date(2026, 9, 30)]

VENUE_ALIASES = {"BF Saint-Louis": "Basic Fit", "Basic Fit Paris - no hack squat": "Basic Fit"}
BENCH_UNASSISTED_FROM = pd.Timestamp("2026-09-10")
HARD_SET_MAX_RIR = 3  # sets at RIR 0-3, or with no RIR logged, count as hard sets
FLAT_BAND = 0.01  # changes within ±1% e1RM read as flat

FREE_WEIGHT_EXERCISES = {
    "Élévations latérales",
    "Standing OHP barbell",
    "Développé couché barre",
    "Back squat barre",
    "Développé incliné haltères",
    "Romanian deadlift barre",
    "Écarté haltère pec",
    "Curl incliné haltères",
    "Shrug barre",
    "Bulgarian split squat",
    "Reverse curl EZ",
    "Hammer curl",
    "Dips (lestés ou BW)",
    "Pull-ups (BW ou lestés)",
}
BODYWEIGHT_ADDED = {"Dips (lestés ou BW)", "Pull-ups (BW ou lestés)"}

# Ordered keyword → muscle rules. First match wins, so order matters
# (e.g. "overhead triceps" must hit Triceps before Front delts; "wrist curl"
# must hit Forearms before Biceps). This classifies REAL exercise names; it is
# metadata, not fabricated data.
MUSCLE_RULES = [
    (
        [
            "jackknife",
            "crunch",
            "russian twist",
            "sit-up",
            "situp",
            "plank",
            "abdo",
            "leg raise",
            "levé de jambe",
            "leve de jambe",
            "toes-to-bar",
            "dead bug",
            "wood chop",
            "pallof",
        ],
        "Abs",
    ),
    (["mollet", "calf", "calv"], "Calves"),
    (["glute"], "Glutes"),
    (["wrist", "avant-bras", "forearm"], "Forearms"),
    (["triceps"], "Triceps"),
    (
        ["face pull", "rear delt", "oiseau", "reverse pec", "arrière épaule", "arriere epaule"],
        "Rear delts",
    ),
    (
        [
            "élévation latérale",
            "elevation laterale",
            "élévations latérales",
            "elevations laterales",
            "lateral raise",
            "side delt",
        ],
        "Side delts",
    ),
    (["ohp", "militaire", "shoulder press", "overhead press"], "Front delts"),
    # Leg rules MUST come before the biceps "curl" rule — "leg curl" contains "curl".
    (["leg curl", "ischio", "leg-curl"], "Hamstrings"),
    (
        [
            "deadlift",
            "soulevé de terre",
            "souleve de terre",
            "sdt",
            "roumain",
            "romanian",
            "pull-through",
            "pull through",
        ],
        "Hamstrings",
    ),
    (["leg extension"], "Quads"),
    (["squat", "hack", "bulgarian", "fente", "lunge", "presse", "leg press"], "Quads"),
    (["curl"], "Biceps"),
    (["shrug", "trapèze", "trapeze"], "Traps"),
    (
        [
            "couché",
            "couche",
            "incliné",
            "incline",
            "écarté",
            "ecarte",
            "dips",
            "pec",
            "bench",
            "fly",
        ],
        "Chest",
    ),
    (
        [
            "tirage",
            "rowing",
            "row",
            "traction",
            "pull-up",
            "pull up",
            "pull-over",
            "pullover",
            "pulldown",
        ],
        "Back",
    ),
]

# Changes that look like regressions but are not like-for-like (shown as a note in the report).
CAVEATS = {
    ("Preacher curl", "ToTheLimitGym"): "Load convention was not recorded on the earlier dates",
    ("Kneeling leg curl iso-lateral (HS)", "ToTheLimitGym"): "Earlier set was an equipment test",
}

REQUIRED_COLUMNS = {"date", "session", "exercise", "weight", "reps", "rir", "note"}


def muscle_of(exercise):
    name = str(exercise).lower()
    for keywords, muscle in MUSCLE_RULES:
        if any(keyword in name for keyword in keywords):
            return muscle
    return "Other"


def load_training_log(path):
    """Read training_log.csv with parsed dates, session family, venue and muscle, or None."""
    path = Path(path)
    if not path.exists():
        return None
    t = pd.read_csv(path)
    if t.empty or not REQUIRED_COLUMNS <= set(t.columns):
        return None
    t["dt"] = pd.to_datetime(t["date"], format="%d/%m/%Y", errors="coerce")
    for column in ["weight", "reps", "rir"]:
        t[column] = pd.to_numeric(t[column], errors="coerce")
    t["family"] = t["session"].str.replace(r"\s*·\s*TTL$", "", regex=True).str.strip()
    venue = t["note"].fillna("").str.extract(r"^\[([^\]]+)\]")[0].fillna("Unrecorded")
    t["venue"] = venue.replace(VENUE_ALIASES)
    t["muscle"] = t["exercise"].apply(muscle_of)
    return t.dropna(subset=["dt"])


def session_table(t):
    """One row per logged session, oldest first, with its venue and number of sets."""
    if t is None or t.empty:
        return pd.DataFrame(columns=["dt", "date", "session", "family", "venue", "sets"])
    rows = []
    for (dt, session), g in t.groupby(["dt", "session"], sort=True):
        rows.append(
            {
                "dt": dt,
                "date": g["date"].iloc[0],
                "session": session,
                "family": g["family"].iloc[0],
                "venue": g["venue"].mode().iloc[0],
                "sets": len(g),
            }
        )
    return pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)


def assign_cycles(sessions):
    """Add `cycle` (1-based) and `extra` (repeat of a session already in its cycle) columns."""
    starts = {pd.Timestamp(d) for d in KNOWN_CYCLE_STARTS}
    extras = {pd.Timestamp(d) for d in EXTRA_SESSION_DATES}
    out = sessions.copy()
    cycle, seen, numbers, flags = 0, set(), [], []
    for row in out.itertuples():
        complete = set(SESSION_FAMILIES) <= seen
        extra_day = row.dt in extras
        if not seen or row.dt in starts or (not extra_day and (row.family in seen or complete)):
            cycle, seen = cycle + 1, set()
        flags.append(row.family in seen)
        seen.add(row.family)
        numbers.append(cycle)
    out["cycle"] = numbers
    out["extra"] = flags
    return out


def cycles_in_range(sessions, start, end):
    """Summaries of every cycle that has a session between start and end (inclusive)."""
    if sessions.empty:
        return []
    cycled = assign_cycles(sessions)
    in_range = cycled[(cycled["dt"] >= pd.Timestamp(start)) & (cycled["dt"] <= pd.Timestamp(end))]
    summaries = []
    for number in sorted(in_range["cycle"].unique()):
        g = cycled[cycled["cycle"] == number]
        done = [f for f in SESSION_FAMILIES if f in set(g["family"])]
        summaries.append(
            {
                "number": int(number),
                "start": g["dt"].min(),
                "end": g["dt"].max(),
                "sessions": len(g),
                "done": done,
                "missing": [f for f in SESSION_FAMILIES if f not in done],
                "complete": len(done) == len(SESSION_FAMILIES),
                "extras": int(g["extra"].sum()),
            }
        )
    return summaries


def _scope(exercise, venue, note):
    """Free weights compare everywhere; machines and cables only within one venue."""
    if exercise == "Élévations latérales" and (
        venue == "ToTheLimitGym" or re.search("machine", str(note), re.I)
    ):
        return venue  # at TTL the lateral raise is the Flame Sport machine, not dumbbells
    if exercise == "Standing OHP barbell" and venue == "Home":
        return venue  # home OHP is often seated
    return "All venues" if exercise in FREE_WEIGHT_EXERCISES else venue


def with_scope(t):
    """Copy of the log with the `scope` each set is compared within."""
    out = t.copy()
    out["scope"] = [_scope(e, v, n) for e, v, n in zip(out["exercise"], out["venue"], out["note"])]
    return out


def best_sets(t, bodyweight=None):
    """Best estimated 1RM per exercise, comparison scope and day. Abs are left out."""
    s = t[t["reps"].gt(0) & t["muscle"].ne("Abs")].copy()
    s = s[~((s["exercise"] == "Développé couché barre") & (s["dt"] < BENCH_UNASSISTED_FROM))]
    if s.empty:
        return pd.DataFrame(columns=["exercise", "scope", "dt", "e1rm", "top", "muscle"])
    s = with_scope(s)
    load = s["weight"].fillna(0)
    if bodyweight is not None and len(bodyweight):
        bw = bodyweight.dropna()
        bw = bw[~bw.index.duplicated()].sort_index()
        body = bw.reindex(s["dt"].values, method="ffill").fillna(0).values
    else:
        body = np.zeros(len(s))
    s["load"] = np.where(s["exercise"].isin(BODYWEIGHT_ADDED), load + body, load)
    s = s[s["load"] > 0]
    s["e1rm"] = s["load"] * (1 + s["reps"] / 30)
    s["top"] = [f"{w:g} kg × {r:g}" for w, r in zip(s["weight"].fillna(0), s["reps"])]
    idx = s.groupby(["exercise", "scope", "dt"])["e1rm"].idxmax()
    return s.loc[idx, ["exercise", "scope", "dt", "e1rm", "top", "muscle"]].reset_index(drop=True)


def hard_sets(t):
    """Working sets at RIR 3 or lower, or with no RIR logged."""
    return t[t["reps"].gt(0) & (t["rir"].isna() | (t["rir"] <= HARD_SET_MAX_RIR))]


def _volume_table(t, start):
    week = hard_sets(t[(t["dt"] >= start) & (t["dt"] <= start + pd.Timedelta(days=6))])
    prior = hard_sets(t[(t["dt"] >= start - pd.Timedelta(days=7)) & (t["dt"] < start)])
    first = t["dt"].min()
    weeks_back = max(0, min(4, (start - first).days // 7))
    prev = hard_sets(t[(t["dt"] >= start - pd.Timedelta(weeks=4)) & (t["dt"] < start)])
    table = pd.DataFrame(
        {
            "This week": week.groupby("muscle").size(),
            "Prior week": prior.groupby("muscle").size(),
            "Prior 4-wk avg": prev.groupby("muscle").size() / weeks_back if weeks_back else np.nan,
        }
    ).fillna({"This week": 0, "Prior week": 0})
    table = table[(table["This week"] > 0) | (table["Prior week"] > 0)]
    return table.sort_values(["This week", "Prior week"], ascending=False).round(1)


def _progress_table(history, start, end):
    rows = []
    for (exercise, scope), g in history.groupby(["exercise", "scope"]):
        week = g[(g["dt"] >= start) & (g["dt"] <= end)]
        if week.empty:
            continue
        best = week.loc[week["e1rm"].idxmax()]
        earlier = g[g["dt"] < start].sort_values("dt")
        if earlier.empty:
            status, change, prev = "new", np.nan, np.nan
        else:
            prev = earlier["e1rm"].iloc[-1]
            change = best["e1rm"] / prev - 1
            if len(earlier) >= 3 and best["e1rm"] > earlier["e1rm"].max() * (1 + FLAT_BAND / 2):
                status = "PR"
            elif change > FLAT_BAND:
                status = "up"
            elif change < -FLAT_BAND:
                status = "down"
            else:
                status = "flat"
        rows.append(
            {
                "Exercise": exercise,
                "Compared": scope,
                "Best set": best["top"],
                "e1RM": round(best["e1rm"], 1),
                "Previous e1RM": round(prev, 1) if prev == prev else np.nan,
                "Change %": round(change * 100, 1) if change == change else np.nan,
                "Status": status,
                "Muscle": best["muscle"],
                "Note": CAVEATS.get((exercise, scope), ""),
            }
        )
    order = {"PR": 0, "up": 1, "flat": 2, "down": 3, "new": 4}
    table = pd.DataFrame(
        rows,
        columns=[
            "Exercise",
            "Compared",
            "Best set",
            "e1RM",
            "Previous e1RM",
            "Change %",
            "Status",
            "Muscle",
            "Note",
        ],
    )
    if table.empty:
        return table
    return table.sort_values(by="Status", key=lambda s: s.map(order), kind="stable").reset_index(
        drop=True
    )


def weekly_training_summary(t, week_start, bodyweight=None):
    """Training view of one Monday–Sunday week: sessions, cycles, hard sets and progress."""
    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    upto = t[t["dt"] <= end]
    sessions = session_table(upto)
    week_sessions = assign_cycles(sessions) if not sessions.empty else sessions
    if not week_sessions.empty:
        week_sessions = week_sessions[(week_sessions["dt"] >= start) & (week_sessions["dt"] <= end)]
    progress = _progress_table(best_sets(upto, bodyweight), start, end)
    counts = progress["Status"].value_counts().to_dict() if not progress.empty else {}
    volume = _volume_table(upto, start)
    return {
        "week_start": start,
        "week_end": end,
        "sessions": week_sessions.reset_index(drop=True),
        "cycles": cycles_in_range(sessions, start, end),
        "volume": volume,
        "hard_sets": int(volume["This week"].sum()) if not volume.empty else 0,
        "prior_hard_sets": int(volume["Prior week"].sum()) if not volume.empty else 0,
        "progress": progress,
        "counts": {k: int(v) for k, v in counts.items()},
    }


def describe_cycle(cycle):
    """One markdown line describing a cycle summary from `cycles_in_range`."""
    status = "complete" if cycle["complete"] else f"{len(cycle['done'])}/5 so far"
    line = (
        f"**Cycle {cycle['number']}** · {cycle['start']:%b %d}–{cycle['end']:%b %d} · "
        f"{cycle['sessions']} sessions ({status})"
    )
    if cycle["extras"]:
        line += f" · {cycle['extras']} extra"
    if cycle["missing"]:
        line += " · still to do: " + ", ".join(cycle["missing"])
    return line


def cycle_history(sessions):
    """Every cycle, oldest first: dates, sessions, completeness and venues."""
    if sessions.empty:
        return pd.DataFrame(
            columns=["Cycle", "Start", "End", "Sessions", "Complete", "Extra", "Missing", "Venues"]
        )
    cycled = assign_cycles(sessions)
    rows = []
    for number, g in cycled.groupby("cycle"):
        done = {f for f in g["family"]}
        rows.append(
            {
                "Cycle": int(number),
                "Start": g["dt"].min(),
                "End": g["dt"].max(),
                "Sessions": len(g),
                "Complete": set(SESSION_FAMILIES) <= done,
                "Extra": int(g["extra"].sum()),
                "Missing": ", ".join(f for f in SESSION_FAMILIES if f not in done),
                "Venues": ", ".join(sorted(set(g["venue"]))),
            }
        )
    return pd.DataFrame(rows)


def weekly_progress(history):
    """Per Monday week: how many of the lifts done went up, stayed flat or went down against the
    previous time each was done, and the median change. Every lift is compared only with its own
    past, so new exercises and new gyms cannot distort it (they count as new, not as changes)."""
    columns = ["Up", "Flat", "Down", "Up share %", "Median change %"]
    if history.empty:
        return pd.DataFrame(columns=columns)
    h = history.sort_values(["exercise", "scope", "dt"]).copy()
    h["change"] = h["e1rm"] / h.groupby(["exercise", "scope"])["e1rm"].shift(1) - 1
    h = h.dropna(subset=["change"])
    h["week"] = h["dt"].dt.to_period("W-SUN").dt.start_time
    grouped = h.groupby("week")["change"]
    out = pd.DataFrame(
        {
            "Up": grouped.apply(lambda c: int((c > FLAT_BAND).sum())),
            "Flat": grouped.apply(lambda c: int(c.abs().le(FLAT_BAND).sum())),
            "Down": grouped.apply(lambda c: int((c < -FLAT_BAND).sum())),
            "Median change %": (grouped.median() * 100).round(1),
        }
    )
    out["Up share %"] = (out["Up"] / out[["Up", "Flat", "Down"]].sum(axis=1) * 100).round(0)
    return out[columns]


def pr_events(history, start=None, end=None):
    """Sessions that beat every earlier best of an exercise (needs 3 earlier sessions)."""
    rows = []
    for (exercise, scope), g in history.sort_values("dt").groupby(["exercise", "scope"]):
        best, count = 0.0, 0
        for r in g.itertuples():
            if count >= 3 and r.e1rm > best * (1 + FLAT_BAND / 2):
                rows.append(
                    {
                        "Date": r.dt,
                        "Exercise": exercise,
                        "Compared": scope,
                        "Best set": r.top,
                        "e1RM": round(r.e1rm, 1),
                        "Previous best": round(best, 1),
                        "Muscle": r.muscle,
                    }
                )
            best, count = max(best, r.e1rm), count + 1
    table = pd.DataFrame(
        rows,
        columns=["Date", "Exercise", "Compared", "Best set", "e1RM", "Previous best", "Muscle"],
    )
    if start is not None:
        table = table[table["Date"] >= pd.Timestamp(start)]
    if end is not None:
        table = table[table["Date"] <= pd.Timestamp(end)]
    return table.sort_values("Date", ascending=False).reset_index(drop=True)


def period_progress(history, start, end):
    """Best e1RM inside the period versus the last time before it, per exercise and scope."""
    return _progress_table(history, pd.Timestamp(start), pd.Timestamp(end))


def weekly_hard_sets(t, start=None, end=None):
    """Hard sets per Monday week (rows) and muscle (columns)."""
    hard = hard_sets(t)
    if start is not None:
        hard = hard[hard["dt"] >= pd.Timestamp(start)]
    if end is not None:
        hard = hard[hard["dt"] <= pd.Timestamp(end)]
    hard = hard[hard["muscle"] != "Other"].copy()
    if hard.empty:
        return pd.DataFrame()
    hard["week"] = hard["dt"].dt.to_period("W-SUN").dt.start_time
    return hard.groupby(["week", "muscle"]).size().unstack(fill_value=0).sort_index()


def muscle_freshness(t, as_of):
    """Per muscle: days since it was last trained, hard sets in the last 7 and 28 days."""
    as_of = pd.Timestamp(as_of)
    hard = hard_sets(t)
    rows = []
    for muscle, g in t[t["muscle"] != "Other"].groupby("muscle"):
        last = g["dt"].max()
        recent = hard[hard["muscle"] == muscle]
        rows.append(
            {
                "Muscle": muscle,
                "Last trained": last,
                "Days since": (as_of - last).days,
                "Hard sets, last 7 days": int(
                    ((recent["dt"] > as_of - pd.Timedelta(days=7)) & (recent["dt"] <= as_of)).sum()
                ),
                "Weekly avg, last 28 days": round(
                    ((recent["dt"] > as_of - pd.Timedelta(days=28)) & (recent["dt"] <= as_of)).sum()
                    / 4,
                    1,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("Days since", ascending=False).reset_index(drop=True)


def series_options(history):
    """(label, exercise, scope) for every exercise and comparison scope, most recent first."""
    last = history.groupby(["exercise", "scope"])["dt"].max().sort_values(ascending=False)
    return [(f"{e} · {sc}" if sc != "All venues" else e, e, sc) for (e, sc) in last.index]


def series_sessions(t, exercise, scope):
    """Per-session rows (date, venue, sets, best e1RM) of one exercise within its scope."""
    rows = with_scope(t[(t["exercise"] == exercise) & t["reps"].gt(0)])
    rows = rows[rows["scope"] == scope]
    out = []
    for dt, g in rows.groupby("dt"):
        g = g.sort_values("set_number") if "set_number" in g.columns else g
        sets = " / ".join(f"{(w if w == w else 0):g}×{r:g}" for w, r in zip(g["weight"], g["reps"]))
        load = g["weight"].fillna(0)
        best = float((load * (1 + g["reps"] / 30)).max())
        out.append(
            {
                "dt": dt,
                "Venue": g["venue"].mode().iloc[0],
                "Sets": sets,
                "Best e1RM": round(best, 1),
            }
        )
    return pd.DataFrame(out)


def weekly_overview(t, health):
    """One row per Monday week: training volume and strength next to the health data."""
    start = t["dt"].min().to_period("W-SUN").start_time
    weeks = pd.date_range(start, t["dt"].max().to_period("W-SUN").start_time, freq="W-MON")
    sessions = session_table(t)
    sessions["week"] = sessions["dt"].dt.to_period("W-SUN").dt.start_time
    hard = hard_sets(t).assign(week=lambda f: f["dt"].dt.to_period("W-SUN").dt.start_time)
    h = health.copy()
    h["week"] = h["date"].dt.to_period("W-SUN").dt.start_time
    move = pd.to_numeric(h.get("move_kcal"), errors="coerce")
    intake = pd.to_numeric(h.get("calories_consumed"), errors="coerce")
    valid = move.gt(0) & intake.gt(0)
    h["deficit"] = np.where(valid, 2000 + move - intake, np.nan)
    h["sleep_h"] = pd.to_numeric(h["sleep_min"], errors="coerce") / 60
    health_weekly = h.groupby("week").agg(
        Weight=("weight", "mean"),
        Waist=("waist_cm", "mean"),
        Intake=("calories_consumed", "mean"),
        Move=("move_kcal", "mean"),
        Deficit=("deficit", "mean"),
        Steps=("steps", "mean"),
        Sleep=("sleep_h", "mean"),
    )
    table = pd.DataFrame(index=weeks)
    table["Sessions"] = sessions.groupby("week").size()
    table["Hard sets"] = hard.groupby("week").size()
    table = table.fillna({"Sessions": 0, "Hard sets": 0}).astype(int)
    table = table.join(health_weekly)
    weights = health.dropna(subset=["weight"]).drop_duplicates("date").set_index("date")["weight"]
    progress = weekly_progress(best_sets(t, weights))
    table["Lifts up %"] = progress["Up share %"].reindex(table.index)
    table["Median change %"] = progress["Median change %"].reindex(table.index)
    table.index.name = "Week"
    return table.round(1)
