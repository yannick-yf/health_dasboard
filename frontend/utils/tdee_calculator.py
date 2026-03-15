"""
Science-based TDEE Calculator — Apple Watch correction layer.

Problem
-------
Apple Watch calories_burned has known biases:
  - Resistance training: overestimates by 30–50% (Stanford 2017, Shcherbina et al.)
  - Bike commute: often not tracked if no explicit workout started
  - Compounding BMR double-counts inside its algorithm

Approach
--------
TDEE = BMR + EAT_gym + EAT_bike + NEAT + TEF

  BMR   — Mifflin-St Jeor (most validated formula, ±~10% RMSE)
  EAT   — Exercise Activity Thermogenesis, MET-based (Ainsworth 2011 Compendium)
            Uses *net* MET (gross MET − 1) to avoid double-counting BMR
  NEAT  — Non-Exercise Activity Thermogenesis, derived from steps
            Gym steps subtracted from total to prevent double-counting with EAT
  TEF   — Thermic Effect of Food (~10% of calories consumed)

References
----------
Mifflin MD et al. (1990). Am J Clin Nutr 51(2):241–247.
Ainsworth BE et al. (2011). Med Sci Sports Exerc 43(8 Suppl 1):S1–S56.
Hall KD et al. (2011). Int J Obes 35(11):1378–1383.  [step energy cost]
Shcherbina A et al. (2017). J Pers Med 7(2):3.       [wearable accuracy]
Levine JA (2004). Science 307(5709):584.              [NEAT importance]
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from datetime import date

import pandas as pd

# ─── Personal profile (loaded from personal_info.json) ───────────────────────
_PROFILE_PATH = Path(__file__).parent.parent.parent / "data" / "personal_info.json"


@lru_cache(maxsize=1)
def _load_profile() -> dict:
    with open(_PROFILE_PATH) as f:
        return json.load(f)

# ─── MET constants (Ainsworth 2011 Compendium of Physical Activities) ─────────
# Gross MET values. Net = MET - 1 to exclude resting metabolic cost already
# captured in BMR. This is the standard practice per Ainsworth et al.

MET_RESISTANCE_MODERATE = 5.0   # Code 02040 — free weights/machines, moderate
MET_RESISTANCE_VIGOROUS = 6.0   # Code 02050 — vigorous (heavy compound lifts)
MET_CYCLING_COMMUTE     = 6.8   # Code 01020 — cycling 16–19 km/h (~15 km/h commute)
MET_WALKING_4KPH        = 3.5   # Code 17151 — walking on level, ~4 km/h

# ─── Activity profile constants ───────────────────────────────────────────────
GYM_MIN_CAP          = 65   # User stated: actual training is 60–70 min max
BIKE_ROUND_TRIP_MIN  = 30   # 15 min each way, office commute
AVG_STRIDE_M         = 0.78 # Average stride length for 185 cm male (m)
NET_WALK_COST_KCAL   = 0.57 # Net walking energy cost kcal/kg/km (Hall et al. 2011)
                              # "net" = gross cost minus resting metabolic rate
GYM_STEPS_PER_MIN    = 35   # Estimated steps/min recorded by watch during resistance
                              # training (moving between sets, squats, etc.)
TEF_FRACTION         = 0.10  # Thermic Effect of Food = 10% of intake (standard)


# ─── Core functions ───────────────────────────────────────────────────────────

def bmr_mifflin(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
    """
    Mifflin-St Jeor BMR (kcal/day).
    Most accurate resting metabolic rate formula for general population.
    Error: ±10% vs indirect calorimetry (gold standard).
    """
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if sex.lower() == "male" else base - 161


def _net_met_kcal(gross_met: float, weight_kg: float, duration_min: float) -> float:
    """
    Convert MET + duration to net kcal, excluding resting metabolic cost.
    Net MET = gross MET - 1  (Ainsworth et al. 2011, Appendix A)
    """
    return (gross_met - 1.0) * weight_kg * (duration_min / 60.0)


def neat_from_steps(
    total_steps: int,
    weight_kg: float,
    gym_min: int,
    stride_m: float = AVG_STRIDE_M,
) -> float:
    """
    Estimate NEAT (kcal) from step count, excluding steps taken during
    structured gym training to prevent double-counting with EAT_gym.

    Formula (Hall et al. 2011):
        NEAT_kcal = NEAT_steps × stride_m × net_walk_cost × weight_kg
                  = NEAT_steps × 0.00057 × weight_kg  (per metre)
    """
    gym_steps = int(GYM_STEPS_PER_MIN * gym_min)
    neat_steps = max(0, total_steps - gym_steps)
    distance_km = neat_steps * stride_m / 1000.0
    return NET_WALK_COST_KCAL * weight_kg * distance_km


def is_bike_day(d: date) -> bool:
    """
    Heuristic: weekdays (Mon–Fri) are potential bike commute days.
    Override per-day if you add a `bike_commute` boolean column to the CSV.
    """
    return d.weekday() < 5  # 0=Mon, 4=Fri


def estimate_tdee(
    weight_kg: float,
    steps: int,
    workout_duration_min: int,
    day: date | None = None,
    calories_consumed: float | None = None,
    biked_today: bool | None = None,
    height_cm: float | None = None,
    age: int | None = None,
    sex: str | None = None,
    resistance_met: float = MET_RESISTANCE_MODERATE,
) -> dict:
    """
    Estimate science-based TDEE for one day.

    Parameters
    ----------
    weight_kg           : body weight in kg (use that day's value)
    steps               : total step count from Apple Watch
    workout_duration_min: total logged workout minutes (includes gym + other)
    day                 : date object (used to infer bike day if biked_today=None)
    calories_consumed   : kcal eaten that day (for TEF; uses BMR estimate if None)
    biked_today         : True/False override; if None → inferred from weekday
    height_cm           : overrides personal_info.json
    age                 : overrides personal_info.json
    sex                 : overrides personal_info.json
    resistance_met      : use MET_RESISTANCE_VIGOROUS for heavy lifting days

    Returns
    -------
    dict with keys:
        bmr, eat_gym, eat_bike, neat, tef, tdee_science,
        gym_min_used, bike_min_used, neat_steps, apple_watch_notes
    """
    profile = _load_profile()
    h = height_cm or profile["height_cm"]
    a = age or profile["age"]
    s = sex or profile["sex"]

    # ── BMR ──────────────────────────────────────────────────────────────────
    bmr = bmr_mifflin(weight_kg, h, a, s)

    # ── EAT: Gym (resistance training) ───────────────────────────────────────
    # Cap at GYM_MIN_CAP: user stated 60–70 min max of actual training.
    # Anything beyond is walking/biking, not structured lifting.
    gym_min = min(workout_duration_min, GYM_MIN_CAP)
    eat_gym = _net_met_kcal(resistance_met, weight_kg, gym_min)

    # ── EAT: Bike commute ────────────────────────────────────────────────────
    # Infer from weekday if not explicitly provided.
    if biked_today is None:
        biked_today = is_bike_day(day) if day is not None else False
    bike_min = BIKE_ROUND_TRIP_MIN if biked_today else 0
    eat_bike = _net_met_kcal(MET_CYCLING_COMMUTE, weight_kg, bike_min)

    # ── NEAT: Steps (non-gym locomotion) ─────────────────────────────────────
    neat = neat_from_steps(total_steps=steps, weight_kg=weight_kg, gym_min=gym_min)

    # ── TEF: Thermic Effect of Food ───────────────────────────────────────────
    # Use actual intake if available; fall back to 10% of BMR (~2450 kcal intake est.)
    tef = (calories_consumed * TEF_FRACTION) if (calories_consumed and calories_consumed > 0) else (bmr * TEF_FRACTION)

    tdee_science = bmr + eat_gym + eat_bike + neat + tef

    return {
        "bmr":           round(bmr),
        "eat_gym":       round(eat_gym),
        "eat_bike":      round(eat_bike),
        "neat":          round(neat),
        "tef":           round(tef),
        "tdee_science":  round(tdee_science),
        # diagnostics
        "gym_min_used":  gym_min,
        "bike_min_used": bike_min,
        "neat_steps":    max(0, steps - int(GYM_STEPS_PER_MIN * gym_min)),
    }


# ─── Dataset-level application ────────────────────────────────────────────────

def apply_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply science-based TDEE to a full dataset.

    Expects columns: date, weight, steps, workout_duration_min_tot,
                     calories_burned (Apple Watch), calories_consumed

    Adds columns:
        tdee_science     — new MET-based estimate
        tdee_delta       — science minus Apple Watch (negative = watch overestimates)
        tdee_delta_pct   — percentage difference
        tdee_bmr, tdee_eat_gym, tdee_eat_bike, tdee_neat, tdee_tef  — components
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    results = []
    for _, row in df.iterrows():
        w = row.get("weight")
        st = row.get("steps")
        wd = row.get("workout_duration_min_tot")
        cc = row.get("calories_consumed")

        # Skip rows with critical missing data
        if pd.isna(w) or pd.isna(st) or pd.isna(wd):
            results.append({k: None for k in [
                "tdee_science", "tdee_delta", "tdee_delta_pct",
                "tdee_bmr", "tdee_eat_gym", "tdee_eat_bike", "tdee_neat", "tdee_tef"
            ]})
            continue

        est = estimate_tdee(
            weight_kg=float(w),
            steps=int(st),
            workout_duration_min=int(wd),
            day=row["date"].date(),
            calories_consumed=float(cc) if pd.notna(cc) else None,
        )

        apple_watch_tdee = row.get("calories_burned")
        delta = (est["tdee_science"] - apple_watch_tdee) if pd.notna(apple_watch_tdee) else None
        delta_pct = (delta / apple_watch_tdee * 100) if (delta is not None and apple_watch_tdee) else None

        results.append({
            "tdee_science":    est["tdee_science"],
            "tdee_delta":      round(delta) if delta is not None else None,
            "tdee_delta_pct":  round(delta_pct, 1) if delta_pct is not None else None,
            "tdee_bmr":        est["bmr"],
            "tdee_eat_gym":    est["eat_gym"],
            "tdee_eat_bike":   est["eat_bike"],
            "tdee_neat":       est["neat"],
            "tdee_tef":        est["tef"],
        })

    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)


# ─── CLI convenience ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from datetime import datetime

    # Quick single-day estimate for today
    today = datetime.today().date()
    result = estimate_tdee(
        weight_kg=74.5,
        steps=9000,
        workout_duration_min=90,   # total logged (gym 65 + other)
        day=today,
        calories_consumed=3500,
    )
    print("\n── Science-based TDEE estimate ──────────────────────────")
    print(f"  BMR (Mifflin-St Jeor):        {result['bmr']} kcal")
    print(f"  EAT gym ({result['gym_min_used']} min resistance):   {result['eat_gym']} kcal")
    print(f"  EAT bike ({result['bike_min_used']} min cycling):    {result['eat_bike']} kcal")
    print(f"  NEAT ({result['neat_steps']:,} non-gym steps):    {result['neat']} kcal")
    print(f"  TEF (10% intake):             {result['tef']} kcal")
    print(f"  ─────────────────────────────────────────")
    print(f"  TDEE (science):               {result['tdee_science']} kcal")
