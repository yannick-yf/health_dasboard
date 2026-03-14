# Deep Dive — Bulk Journey Tracker: How It Works

> **File**: `frontend/sections/deep_dive.py`
> **Purpose**: Evidence-based tracking of a clean bulk (or cut) phase, going beyond raw numbers to answer *"am I gaining the right kind of weight, at the right rate, with the right conditions?"*

---

## Overview

The Deep Dive page is a phase-aware analysis tool. It only uses data **from the bulk start date** (`BULK_START = 2025-11-05`) onward — everything before that is pre-bulk baseline and is excluded from phase metrics. All thresholds are grounded in sports science literature (Helms et al., Ripped Body, Lamon 2021, Levine 2004).

---

## Constants & Thresholds

| Constant | Value | Rationale |
|---|---|---|
| `BULK_START` | 2025-11-05 | Program start date |
| `TARGET_WEIGHT` | 76.0 kg | Goal weight for this phase |
| `START_BF_PCT` | 13.0% | Body fat at bulk start (from `personal_info.json`) |
| `P_RATIO_LEAN` | 0.80 | 80% of gain is lean at ~13% BF + controlled rate |
| `P_RATIO_FAST` | 0.65 | Degrades to 65% lean if gaining > 0.40 kg/wk |
| `GAIN_MIN` | 0.12 kg/wk | Lower bound for clean bulk (~0.16%/wk at 74 kg) |
| `GAIN_MAX` | 0.18 kg/wk | Upper bound (~0.24%/wk at 74 kg) |
| `GAIN_WARN_HIGH` | 0.40 kg/wk | Excessive fat accretion threshold |
| `SURPLUS_LO` | 200 kcal/day | Minimum surplus for anabolic signaling |
| `SURPLUS_HI` | 300 kcal/day | Ceiling before disproportionate fat deposition |
| `SLEEP_MIN_H` | 7.0 h | Minimum adequate recovery sleep |
| `NEAT_HIGH` | 12,000 steps | Step count above which NEAT may offset planned surplus |
| `BF_CEILING` | 17.0% | End-bulk signal (insulin sensitivity degrades above ~18%) |

---

## Section 1 — Status Banner

A gradient HTML banner at the top of the page showing the current bulk phase status at a glance.

### How `gain_rate_4w` is calculated

```
gain_rate_4w = (weight_7d_ma[today] − weight_7d_ma[28 days ago]) / 4
```

- **`weight_7d_ma`**: 7-day rolling mean of scale weight (min 3 valid observations). This smooths out water retention, glycogen swings, and measurement noise.
- Dividing by 4 gives a weekly rate.

**Why rolling average, not raw weight?**
Daily scale weight fluctuates 0.5–2 kg from water, food, bowel contents, and glycogen. Using a 7-day MA before computing the trend removes ~85% of this noise, leaving only real tissue changes.

### Status classification

| Label | Condition | Color |
|---|---|---|
| ON TRACK | 0.12–0.18 kg/wk | Green |
| SLIGHTLY FAST | 0.18–0.40 kg/wk | Orange |
| TOO FAST | > 0.40 kg/wk | Red |
| TOO SLOW | 0–0.12 kg/wk | Amber |
| STALLING | < 0 kg/wk | Purple |
| INSUFFICIENT DATA | < 28 days of data | Grey |

The banner also shows:
- **Weeks in**: `(latest date − BULK_START) / 7`
- **ETA to target**: `remaining kg / current 4w rate`

---

## Section 2 — KPI Cards

Six summary metrics covering the whole bulk phase.

| Card | Metric | How computed |
|---|---|---|
| **7d Avg Weight** | Mean of last 7 weight readings | `weight.tail(7).mean()`. Delta shows vs prior 7 days. |
| **4w Gain Rate** | Weekly gain velocity | `gain_rate_4w` (see above). Target: +0.12–0.18 kg/wk. |
| **Total Gained** | Cumulative since bulk start | `(current 7d MA) − (first valid 7d MA in bulk)`. |
| **7d Mean Surplus** | Average energy balance | `(calories_consumed − calories_burned).tail(7).mean()`. Target: +200–300 kcal/day. |
| **Sleep Score** | % nights ≥ 7h | `(sleep_h ≥ 7.0).mean() × 100` over entire bulk phase. |
| **Training Days** | % days with structured workout | `(workout_duration > 30 min).mean() × 100`. |

---

## Section 3 — Warning Signals

An expandable panel that fires automated warnings based on threshold checks.

### Checks performed

| Signal | Trigger | Recommendation shown |
|---|---|---|
| **Gaining too fast** | `rate > 0.40 kg/wk` | Cut 150–200 kcal/day; P-ratio has degraded |
| **Weight trending down** | `rate < 0` | Add 150–200 kcal; audit logging |
| **Gaining slowly** | `0 < rate < 0.12 kg/wk` | Add 100–150 kcal/day, focus on pre/post-workout carbs |
| **Surplus too high** | `7d surplus > 500 kcal` | Beyond MPS capacity, surplus stores as fat |
| **Surplus too low** | `7d surplus < 200 kcal` | Sub-optimal anabolic environment |
| **Poor sleep** | `< 50%` nights ≥ 7h in last 14 | MPS reduced ~18% on sleep-deprived nights (Lamon et al. 2021) |
| **NEAT compensation** | `≥ 50%` of recent days > 12k steps AND `rate < GAIN_MIN` | High movement can offset 200–400 kcal of planned surplus |
| **Recovery risk days** | `≥ 2` days of sleep < 6h + workout > 45 min in last 14 days | Cortisol elevation impairs MPS for 18–24h |
| **BF ceiling** | `estimated BF > 16%` | Plan maintenance or cut phase |

### P-ratio and BF estimation

The P-ratio model (Phillips & Van Loon 2011) estimates what fraction of gained mass is lean tissue:

```
P_ratio = 0.80  if rate ≤ 0.40 kg/wk
P_ratio = 0.65  if rate > 0.40 kg/wk

fat_gained = cumulative_gain × (1 − P_ratio)
lean_gained = cumulative_gain × P_ratio

estimated_body_fat = (start_fat_mass + fat_gained) / current_weight × 100
```

This is a model-based estimate, not a measurement — treat it as directional.

---

## Chart 1 — Weight Trajectory

**Purpose**: Shows whether the 7-day rolling average is tracking inside the evidence-based bulk corridor.

| Element | Description |
|---|---|
| Grey dots | Raw daily weight (scale noise visible) |
| Blue line | 7-day rolling average (signal) |
| Green corridor | Projected range: `start + elapsed_weeks × [0.12, 0.18]` |
| Yellow dashed | 4-week projection at current rate |
| Green dotted | Target weight (76 kg) |

**Why the corridor matters**: The corridor is drawn from bulk start weight using `GAIN_MIN` and `GAIN_MAX`. If the rolling average is consistently below the lower corridor bound, calories are insufficient. If above the upper bound, fat accretion is likely disproportionate.

---

## Chart 2 — Lean vs Fat Gain Estimate (P-ratio model)

**Purpose**: Visualize the quality of the weight gained, not just the quantity.

- **Stacked area**: Bottom = lean gained (purple), top = fat gained (red), stacked from zero using `stackgroup`.
- **Secondary Y**: Estimated BF% over time (amber dashed line).
- **Red dotted line**: 17% BF ceiling.

The stacked values are derived from the P-ratio model applied to the cumulative 7-day MA change since bulk start. Because this is a model (not a DEXA scan), it systematically underestimates fat gain during fast-gaining periods and is most accurate when the gain rate is controlled.

---

## Chart 3 — Surplus: Training vs Rest Days

**Purpose**: Checks whether caloric strategy is correctly periodized — higher surplus on training days, lower on rest days.

- **Grouped bars**: Weekly mean surplus on training days (purple) vs rest days (grey).
- **Green band**: Target surplus zone (+200–300 kcal).
- **Zero line**: Energy balance reference.

**Why this matters**: On training days, muscle protein synthesis is elevated for 24–48h, so caloric surplus is more anabolically productive. On rest days, excess surplus beyond maintenance is more likely to be stored as fat (Hall et al. 2012).

A well-periodized bulk shows higher training-day bars within the green zone, and rest-day bars near zero or slightly positive.

---

## Chart 4 — Sleep Adequacy & Recovery Risk

**Purpose**: Tracks sleep quality per week and flags days when sleep deprivation coincided with training (a cortisol risk).

| Element | Description |
|---|---|
| Bars | % of nights with ≥7h sleep per week (green ≥70%, amber ≥50%, red <50%) |
| Red line | Recovery risk days per week (sleep < 6h on the same day as workout > 45 min) |
| Dotted line | 70% floor (scientific consensus minimum for consistent recovery) |

**Evidence**: Lamon et al. (2021) showed a ~18% reduction in muscle protein synthesis rates after one night of sleep deprivation in resistance-trained individuals.

**Recovery risk definition**: Sleep < 360 minutes AND workout duration > 45 minutes on the same calendar day. This indicates the body was asked to perform acute muscle repair (post-workout MPS window) without adequate anabolic hormone support (primarily GH, released during deep sleep).

---

## Chart 5 — NEAT × Surplus Interaction (Scatter)

**Purpose**: Answers the question: *do high-activity days erase your caloric surplus?*

- **X axis**: Daily step count (proxy for NEAT/total daily movement).
- **Y axis**: Daily caloric surplus (consumed − burned).
- **Point color**:
  - 🟢 Green = 7-day MA rising (gaining)
  - 🟡 Yellow = stable (±0.10 kg over 7 days)
  - 🔴 Red = 7-day MA falling (losing)
- **Vertical dashed line**: 12,000 steps (NEAT compensation threshold)
- **Green band**: Target surplus zone (+200–300 kcal)

**How to read it**: If most points to the right of 12k steps fall below the green band (surplus < 200 kcal), NEAT compensation is actively limiting gains. This is the "I'm eating enough but not gaining" pattern.

**NEAT compensation**: Levine (2004, Science) established that NEAT is the most variable component of TDEE, ranging 300–2000+ kcal/day. High-step days increase total energy expenditure beyond what the watch reports, shrinking the real surplus. The 12,000-step threshold represents approximately the point where untracked ambulatory energy expenditure begins to meaningfully offset a ~250 kcal planned surplus.

---

## Section: TDEE — Apple Watch vs Science

**Purpose**: Compares two TDEE estimates to reveal Apple Watch bias on training-heavy days.

### The Apple Watch problem

Apple Watch overestimates resistance training calorie burn by 30–50% (Shcherbina et al. 2017, Stanford). It also double-counts resting BMR inside its algorithm. The science TDEE model corrects for this.

### Science TDEE model (Ainsworth 2011 + Mifflin 1990)

```
TDEE = BMR + EAT_gym + EAT_bike + NEAT + TEF
```

| Component | Method |
|---|---|
| **BMR** | Mifflin-St Jeor formula (most validated, ±10% vs calorimetry) |
| **EAT_gym** | Net MET × weight × duration. Net MET = gross MET − 1 (removes BMR double-count). Capped at 65 min actual training. MET = 5.0 (moderate) or 6.0 (vigorous). |
| **EAT_bike** | Net MET 5.8 × weight × 30 min round trip (weekdays only). |
| **NEAT** | `(total_steps − gym_steps) × stride_length × 0.57 kcal/kg/km`. Gym steps estimated at 35 steps/min during resistance sessions. |
| **TEF** | 10% of calories consumed (standard thermogenesis estimate). |

### Reading the TDEE comparison chart

- **Stacked bars**: Science TDEE broken into components (BMR, EAT Gym, EAT Bike, NEAT, TEF)
- **Dashed yellow line**: Apple Watch TDEE for the same days

If the bars are systematically higher than the Watch line, the Watch is underestimating real energy expenditure — meaning the real surplus is smaller than it appears. If the Watch line is above the bars, the Watch is overestimating — meaning the real surplus is larger.

---

## `render_daily_trends` (at the bottom)

This section is inherited from `deep_dive_helpers.py` and renders day-by-day metric plots (steps, sleep, weight, calories). It operates on the full dataset (not just bulk phase) and is unchanged from the original dashboard.

---

## Data Columns Required

| Column | Used for |
|---|---|
| `date` | Time axis for all charts, phase filtering |
| `weight` | Rolling average, gain rate, P-ratio model |
| `steps` | NEAT calculation, NEAT compensation scatter |
| `sleep_min` | Recovery analysis, sleep score |
| `workout_duration_min_tot` | Training day flag, EAT calculation |
| `calories_burned` | Apple Watch TDEE, surplus calculation |
| `calories_consumed` | Surplus calculation, TEF estimate |

---

## Key Data Limitations

1. **Weight measurements**: Scale weight is influenced by hydration, food timing, and glycogen. The 7-day rolling average requires at least 3 valid readings; fewer data points reduce accuracy.
2. **P-ratio model**: A static estimate based on population averages. Individual genetics, training age, and steroid-free natural limits mean actual lean/fat split varies. Use it directionally, not as an exact measurement.
3. **Apple Watch bias**: The watch systematically overestimates resistance training calories by 30–50%. The Science TDEE model corrects this, but the science model also has ±10% error from the BMR formula alone.
4. **Calorie logging accuracy**: `calories_consumed` accuracy depends on logging discipline. Even systematic 5–10% under-logging can explain an apparent stall.
5. **NEAT tracking**: Steps proxy for NEAT but miss bike commutes (if not GPS-tracked), standing time, and fidgeting — all of which contribute meaningfully to TDEE.

---

## References

- Mifflin MD et al. (1990). *Am J Clin Nutr* 51(2):241–247. [BMR formula]
- Ainsworth BE et al. (2011). *Med Sci Sports Exerc* 43(8 Suppl 1):S1–S56. [MET compendium]
- Shcherbina A et al. (2017). *J Pers Med* 7(2):3. [Apple Watch accuracy]
- Lamon S et al. (2021). *Mol Cell Endocrinol* 533:111344. [Sleep and MPS]
- Levine JA (2004). *Science* 307(5709):584. [NEAT variability]
- Hall KD et al. (2011). *Int J Obes* 35(11):1378–1383. [Step energy cost]
- Helms ER et al. (2014). *J Int Soc Sports Nutr* 11:20. [Natural bodybuilding targets]
