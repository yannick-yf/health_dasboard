# Nutrition — Watch-Adaptive Deficit Protocol (WADP)

**Status: ACTIVE.** This is Yannick's sole nutrition method — it replaced all older phase/tier-based
approaches in July 2026 and has been validated multiple times since. For current deficit value and
the pending Sunday-review question, see `00_START_HERE.md` — this file is the stable mechanics
reference, not the live number.

## The formula

```
Daily TDEE estimate = BMR₀ + Apple Watch red ring value (Move / "active energy")
Daily intake target  = Daily TDEE estimate − Deficit
```

## Constants (locked — do not auto-recalibrate)

| Parameter | Value | Notes |
|---|---|---|
| `BMR₀` | **2,000 kcal** | Intentionally ~270 above Mifflin-St Jeor estimate, calibrated to offset Apple's known underestimate of active-energy burn. Both errors are roughly constant and cancel out long-term. **Never audit or "correct" this number** — it was empirically validated (see below), not theoretically derived. |
| `Deficit` | currently 200, see `00_START_HERE.md` | Adjusted in ±100 steps, data-driven only (see triggers below). |

## Why trust this over Apple's own numbers (KEY INSIGHT, proven over a full ~14-week cut)

Apple's own TDEE/energy-balance estimate **consistently understated Yannick's real deficit by
~200 kcal/day**. Concretely: over one full cut cycle, the logged average deficit (intake vs Apple's
own TDEE estimate) worked out to roughly maintenance (~0), yet he lost ~2.6kg — implying a *real*
average deficit of ~200 kcal/day that Apple's number wasn't showing.

**Practical consequence: read the WADP `Deficit` setting as a relative dial, not an absolute
truth.** `Deficit = 200` in the formula probably corresponds to a *real* deficit closer to ~350-400.
This doesn't mean the formula is wrong — it means:
- Always cross-check the deficit number against the scale/waist trend, never trust it in isolation.
- When reasoning about "how big a deficit is this," add roughly 200 to the nominal WADP number to
  get a rough real-world estimate — but the trend in weight/waist is the actual ground truth.

## The two-axis reading (recomp-cut framing)

**Overriding priority: preserve muscle. If fat loss and muscle preservation ever conflict, muscle
wins.** Read weight and waist TOGETHER, they cross-check what's actually being lost:

- Waist ↓ + weight flat/slightly down + strength held or rising = **ideal recomp-cut** (fat off,
  muscle kept). This has been the dominant pattern for weeks — don't second-guess it just because
  weight isn't moving fast; that's the point.
- Weight dropping fast + strength dropping = losing muscle → back off the deficit.
- Waist flat for a genuine stretch (not a food/water blip) while weight also flat = fat isn't
  moving → may warrant a small step up in deficit, but never at the cost of strength.
- **Waist NOT dropping despite fast weight loss** is a red flag for muscle/water loss rather than
  fat loss — treat as a warning sign, not a win.

## Step rules (data-driven only — never on a calendar)

- **Step deficit UP** (e.g. 100→200, 200→300) only if waist has genuinely **stalled** (flat trend
  over ~10-14 days) with no muscle-loss signals.
- **Step deficit DOWN** if: main-lift strength drops >3% for 2 consecutive sessions, OR persistent
  low training energy, OR sleep quality clearly worse than his (newborn-adjusted) baseline, OR
  strength stalls for 2+ weeks straight.
- **Hard cap: never exceed −400 kcal deficit.**
- Judge by the **weekly average**, not any single day — a big surplus or deficit day is normal
  noise (restaurant meals, rest days with low Move, etc.) and shouldn't be over-read.

## Safety caps — weight-rate protection

Research basis: muscle-preserving weight loss ≈ 0.5-0.7% bodyweight/week; >1.0%/week risks muscle.

| Tier | Weekly weight-drop (7dMA) | Action |
|---|---|---|
| 🟢 Green | 0–0.5 kg/wk | Continue, no change |
| 🟡 Yellow | 0.5–0.75 kg/wk | Insert a refeed day at `TDEE + 200`, watch closely |
| 🟠 Orange | >0.75 kg/wk, or yellow 2 weeks running | Refeed + permanently reduce Deficit by 100 |
| 🔴 Red | >1.0 kg/wk, or a main lift drops >3% for 2 sessions running | 3-4 day diet break at true
maintenance (`TDEE + 0`), then resume at `Deficit − 100` from the level that triggered it |

At his current ~72kg bodyweight, the red threshold is roughly a 0.9kg/week drop.

**Composite warning signals** (any 2 together = escalate one tier): strength drop >2% in a single
session, subjectively worse sleep, persistent hunger beyond the first 3-4 days at a new deficit,
cold hands/feet, low mood/motivation, noticeably lower gym energy, waist not dropping despite
weight dropping fast.

**Exceptions — don't trigger caps off these:** a single heavy-activity day (basketball, long hike)
dropping weight via glycogen/water; illness/GI upset; travel or heat-driven sweat losses. Wait for
the trend to re-settle before reacting.

## Refeed days

If `Deficit ≥ 200` for 10+ consecutive days, insert one day at `TDEE + 200` (carb-focused, clean
food, not a free-for-all) to blunt metabolic adaptation. Resume the prior deficit the next day.

## Bulk transition (when the cut ends)

**Triggers to consider flipping to a surplus:**
- 4 abs clearly visible in normal (unpumped, morning) light for 2+ consecutive days, OR
- Waist 14-day MA ≤ ~80cm (already achieved as of Sep 2026 — see `00_START_HERE.md` for the actual
  live decision status; hitting this trigger doesn't auto-flip anything, it's a decision point).

**When it flips, the plan (as designed with Yannick, careful not to repeat a past mistake of
overshooting into fat gain):**
- Don't jump straight from a 200-ish deficit to a surplus. Ramp down in steps: **200 → 100 → 0 →
  small surplus (+100)**, each step confirmed by data, not the calendar.
- Phase B1 (initial): `Surplus = +100`, expect ~0.05-0.10 kg/week gain. Watch waist 14dMA — if it
  rises within 2 weeks, cut the surplus to +50.
- Phase B2 (established): `Surplus = +150-200`, expect ~0.10-0.15 kg/week, run 12-16 weeks.
- **Critical rule for judging the bulk once it's running: added weight only "counts" as muscle if
  the lifts are actually accelerating.** Weight up + lifts flat = mostly fat; weight up + lifts
  progressing faster than they did in the deficit = muscle. Track all three axes (waist, weight,
  strength) exactly as during the cut.
- Endpoint triggers for the bulk: waist 14dMA gains ~2cm cumulative, or total weight gain +3-4kg —
  then either a mini-cut or a return to maintenance.

## Daily entry mechanics

Data lives in `data/health_data.csv` (10 columns, see `05_app_and_data_pipeline.md` for schema).
Entered via the Streamlit form (`.venv/bin/streamlit run frontend/app.py`, port 8502): weight,
waist, sleep minutes, Move (red ring) kcal, calories consumed. Weight/waist/sleep sometimes lag by
a day when Yannick is traveling without a scale — waist alone is still trackable and should be used
as the primary signal until weight resumes.

## Related context

- Protein floor ≥160g/day is tracked externally, not in the CSV — see `04_profile_goals_and_style.md`.
  Don't flag protein as under-tracked or as a limiting factor; it's adequate by design.
- FODMAP-sensitive foods affect *waist water readings* independent of true fat/deficit — see
  `03_health_medical.md` before attributing a waist spike/drop to the deficit alone. Key finding:
  **sodium (e.g. soy sauce) drives waist water in a dose-dependent way; onion/garlic FODMAP drives
  gas, a separate axis** — don't conflate the two when reading a post-restaurant-meal waist bump.
- Newborn sleep constraint (see `04_profile_goals_and_style.md`) means expect slower rates and
  don't steepen a deficit to "compensate" for a bad-sleep stretch.
