"""
Shared constants and helpers for the clean bulk program.

Single source of truth — imported by report_generator, deep_dive, data_entry,
and weekly_report. Do NOT redefine these elsewhere.
"""

import pandas as pd

# ── Bulk phase ────────────────────────────────────────────────────────────────
BULK_START = pd.Timestamp("2025-11-05")

# ── Gain rate thresholds (kg/week) ────────────────────────────────────────────
GAIN_MIN       = 0.12   # clean bulk lower bound (~0.16 %/wk at 74 kg)
GAIN_MAX       = 0.18   # clean bulk upper bound (~0.24 %/wk at 74 kg)
GAIN_WARN_HIGH = 0.40   # excessive fat accretion threshold

# ── Surplus thresholds (kcal/day) ─────────────────────────────────────────────
SURPLUS_LO = 200   # floor for muscle synthesis support
SURPLUS_HI = 300   # ceiling before disproportionate fat gain

# ── Recovery ──────────────────────────────────────────────────────────────────
SLEEP_MIN_H = 7.0     # hours — minimum adequate sleep
NEAT_HIGH   = 12_000  # steps/day — above this, NEAT may offset surplus

# ── Training session ──────────────────────────────────────────────────────────
SHORT_SESSION_WARN_MIN = 45  # minutes — flag unusually short training sessions


# ── PPL program (started ~2026-03-03, 12 weeks) ───────────────────────────────
PROGRAM_START = pd.Timestamp("2026-03-03")

# Maps date.weekday() → PPL session label
PPL_DAY_MAP = {
    0: "Push A",
    1: "Pull A",
    2: "Legs",
    3: "Push B",
    4: "Pull B",
    5: "Rest",
    6: "Rest",
}

# Maps week-in-program (1–12) → (meso_num, week_in_meso, rir_range, is_deload)
_MESO_SCHEDULE: dict[int, tuple[int, int, str, bool]] = {
    1:  (1, 1, "3-4", False),
    2:  (1, 2, "2-3", False),
    3:  (1, 3, "1-2", False),
    4:  (1, 4, "deload", True),
    5:  (2, 1, "3-4", False),
    6:  (2, 2, "2-3", False),
    7:  (2, 3, "1-2", False),
    8:  (2, 4, "deload", True),
    9:  (3, 1, "2-3", False),
    10: (3, 2, "1-2", False),
    11: (3, 3, "0-1", False),
    12: (3, 4, "deload", True),
}


def get_mesocycle_context(reference_date: pd.Timestamp | None = None) -> dict | None:
    """
    Return mesocycle metadata for a given reference date.

    Parameters
    ----------
    reference_date : pd.Timestamp or None
        The date to compute context for (defaults to today).

    Returns
    -------
    dict with keys: meso_num, week_in_meso, rir_range, is_deload, label
    Returns None if the reference_date is before PROGRAM_START or out of range.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.today()

    if reference_date < PROGRAM_START:
        return None

    days_since_start = (reference_date - PROGRAM_START).days
    week_num = int(days_since_start / 7) + 1  # 1-based

    entry = _MESO_SCHEDULE.get(week_num)
    if entry is None:
        return None

    meso_num, week_in_meso, rir_range, is_deload = entry
    if is_deload:
        label = f"Meso {meso_num} · Deload"
    else:
        label = f"Meso {meso_num}, Wk {week_in_meso} · RIR {rir_range}"

    return {
        "meso_num":     meso_num,
        "week_in_meso": week_in_meso,
        "rir_range":    rir_range,
        "is_deload":    is_deload,
        "label":        label,
    }


def classify_status(rate) -> tuple[str, str]:
    """Return (label, hex_color) for a 4-week weight gain rate (kg/week)."""
    if rate is None or (isinstance(rate, float) and pd.isna(rate)):
        return "INSUFFICIENT DATA", "#6b7280"
    if rate > GAIN_WARN_HIGH:
        return "TOO FAST", "#ef4444"
    if rate > GAIN_MAX:
        return "SLIGHTLY FAST", "#f97316"
    if rate >= GAIN_MIN:
        return "ON TRACK", "#10b981"
    if rate >= 0:
        return "TOO SLOW", "#f59e0b"
    return "STALLING", "#8b5cf6"
