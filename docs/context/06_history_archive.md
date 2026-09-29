# History Archive — superseded material, reference only

These describe approaches that are **no longer active**. They're kept only so a past decision or
number can be traced if it ever comes up ("why did we used to do X"). **Do not apply any rule from
this file going forward** — everything here has been replaced by what's in files `00`-`05`.

- **Static-tier calorie targeting** (pre-Jul 14 2026) — fixed daily targets by day-type (e.g.
  "3,300 kcal lift day / 3,000 single-session / 2,700 rest day"). Produced a ~48-day plateau.
  Replaced entirely by WADP (`01_nutrition_wadp.md`) because it couldn't adapt to daily activity
  variance.
- **Phase-based bulk/cut planning** (Phase 1/2/3a/3b, "Option C Hybrid," a May 2026 mini-cut plan,
  an earlier "recomp option" writeup) — a sequence of rigid, hand-planned phases for the bulk→cut
  cycle that ran through roughly March-July 2026. Superseded by WADP's continuous dial approach,
  which Yannick found less plateau-prone and lower mental overhead. The overall long-term framing
  from this era — "athletic body with abs visible" as the top-level goal — is still valid and is
  now expressed in `04_profile_goals_and_style.md`.
- **April 2026 four-week bulk review** — found a bulk plateau (+0.1kg net over 4 weeks), which
  triggered a TDEE recalibration and the eventual bulk→cut decision that started the current cut
  cycle. Historical trigger event, not an active protocol.
- **May 2026 family-context note** — wife+baby were away ~2 months (early March→May 5 2026),
  creating a temporary relative-rest window during that bulk. No longer relevant; the family has
  been fully home since, and the permanent newborn context is in `04_profile_goals_and_style.md`.
- **Weekly-review automation project** (built Mar 2026, `debug-front-end` branch) — a two-button,
  deterministic (no-LLM) weekly report generator. Status of that branch/merge is unknown as of this
  migration — worth checking `git branch -a` / `git log` if a "weekly review button" is ever
  mentioned and doesn't seem to exist in the current `frontend/`.
- **Data imputation pipeline** (`scripts/impute_consumed.py`, built Apr 2026) — a two-stage
  regression + maintenance-equation calibration used once to fill a gap in `calories_consumed` for
  a specific date range (Mar 27-Apr 15 2026). Reusable if a similar gap ever recurs, but not part of
  the daily workflow.
- **A previous PPL (Push/Pull/Legs) 5-day program** (ran ~Mar-Apr 2026) and a **Bro Split** (ran
  ~Apr 21-Jul 19 2026) — both closed out and replaced by the current YF-UL5 program. Only relevant
  if an old training-log row from that period needs interpreting (different exercise list/session
  names than the current program).
- **A one-off "should we schedule recurring reviews" discussion** (Apr 2026) — declined by Yannick
  on cost/preference grounds; captured as a standing style preference in
  `04_profile_goals_and_style.md`, not tracked separately here.
