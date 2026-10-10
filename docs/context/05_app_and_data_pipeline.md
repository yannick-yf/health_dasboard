# App & Data Pipeline Reference

## Repos

- **`health_dasboard`** (this repo, `/Users/yannickflores/PersoProject/health_dasboard`) — all
  personal data (CSVs), analysis, Streamlit dashboard, merge scripts. Private.
- **`yf-tracker`** (sibling repo, `~/PersoProject/yf-tracker/`) — a SEPARATE **public** GitHub repo,
  app files ONLY (`index.html`, `sw.js`, `manifest.json`, `icon-192.png`, `icon-512.png`). **Never**
  put health data or anything from `health_dasboard` into this repo. Hosted live on GitHub Pages.
  Edit the app here, not in any older/dev copy — this is the canonical source.
  - To ship an app update: edit files here, then `git add . && git commit --no-gpg-sign -m "..." &&
    git push` (Yannick's global `gpg.format` config is broken, so `--no-gpg-sign` is required; he
    runs the actual git commands himself, don't do it for him without asking).
  - The app is a single-file offline-capable PWA: IndexedDB storage, service worker
    (stale-while-revalidate caching — bump the cache name in `sw.js` on any logic change so it
    reliably picks up), a location-aware "last time here" / "last time anywhere" reference system
    for exercises (see `02_training_program.md` for why the location feature exists).
    Current locations: Home / Basic Fit / ToTheLimitGym / Hotel / Other. ToTheLimitGym was added
    as a distinct baseline after Basic Fit Saint-Louis closed; never merge its machine history
    into Basic Fit.
    App v1.6 was deployed to GitHub Pages and confirmed current on Yannick's phone Sep 29. Source
    v1.7/cache v8 was prepared Oct 7 but Yannick has paused deployment to reassess the venue
    workflow; preserve the local patch, do not treat it as ready to ship. The Train home screen
    now lets him choose the gym before starting, and changing it in a workout draft refreshes the
    previous sets and placeholders while preserving entered values. The lookup still matches
    exercise name and broad location. The home screen still lists full venue-specific session
    variants, while its Today shortcut uses fixed session names; that broader structure remains
    under review.
  - Program-data changes (adding/editing exercises/sets, not app logic) are pushed as a
    `yf-tracker-program-vX.json` file dropped in the iCloud sync folder (see below) and imported
    via the app's 📤 import panel — this is a separate mechanism from a code push. A program-only
    file carries empty health/workout arrays; the app merges imported records and deletes nothing,
    then applies its program config.

## Daily data flow (training log)

1. Yannick logs a session on his phone in the `yf-tracker` PWA (offline-capable).
2. He exports it: **Export JSON → Save to Files → iCloud Drive → `yf-tracker` folder** at
   `~/Library/Mobile Documents/com~apple~CloudDocs/yf-tracker/` (Mac path). The export always
   contains the app's FULL session history, not just what's new — that's normal and expected.
3. Run `.venv/bin/python scripts/merge_tracker_export.py` (no arguments) from the repo root. It:
   - Finds the newest `yf-tracker-export-*.json` in the iCloud folder.
   - Appends new rows to `data/training_log.csv` (schema: `date,session,exercise_order,exercise,
     target,set_number,weight,reps,rir,note`), one row per SET.
   - Dedupes by `date + session` — safe to re-run repeatedly, it's idempotent.
   - **Current limitation:** that same dedupe skips a workout even if its sets or exercise names
     were later corrected on the phone. A renamed session can appear as a second date/session pair.
     Review such changes explicitly; the merge does not reconcile them automatically.
   - Prepends `[Location]` to the note column when a location was tagged on the phone.
   - Never auto-commits — review the diff and let Yannick commit.
4. **If a fresh export doesn't show up in iCloud** (common — phone→iCloud upload lag, often Low
   Power Mode pausing background uploads, or weak signal while traveling): try
   `brctl download ~/Library/Mobile\ Documents/com~apple~CloudDocs/yf-tracker/` to nudge sync. If
   it still doesn't land after a reasonable wait, **ask Yannick to paste the exported JSON directly
   into the chat** — you can write it straight to the iCloud path yourself (a minimal JSON
   containing just the new session's date/exercises is sufficient; you don't need the full
   historical payload since the merge script dedupes anyway) and merge from there. This has worked
   well in practice.

## Health data (weight/waist/sleep/intake/Move)

`data/health_data.csv` — **10 columns, append-only, never change the header**:
```
user_id, date, steps, sleep_min, workout_duration_min_tot, weight, calories_burned,
calories_consumed, waist_cm, move_kcal
```
- `weight` in kg, `waist_cm` in cm, `sleep_min` total minutes asleep (not time in bed).
- `move_kcal` = Apple Watch Move/red-ring active-energy — this drives WADP (`2000 + move −
  deficit`), entered daily via the Streamlit form.
- `workout_duration_min_tot` is legacy/historical (Apple Watch Exercise/green-ring minutes) — kept
  in the CSV for history but no longer actively entered; don't confuse it with `move_kcal`.
- `calories_burned` historically held Apple's own TDEE estimate for cross-checking — WADP's own
  `2000 + move` figure is the one that actually drives decisions (see `01_nutrition_wadp.md` for
  why Apple's own number reads low).
- Entry happens via `.venv/bin/streamlit run frontend/app.py --server.port 8502` — check
  `lsof -ti:8502` before launching to avoid a duplicate process; background it (`nohup ... &`) so
  the conversation isn't blocked.
- Before an existing health CSV is replaced by Streamlit or a tracker merge, and before an existing
  training CSV is appended by a tracker merge, the prior file is copied to `data/backups/`.
  Backups are named by source file and timestamp, with a unique suffix; the newest 10 for each
  source file are retained. `data/backups/` is ignored by Git. Dry runs and no-op merges make no
  backup. These are local recovery copies, not a substitute for Yannick's own Git review.
- Weight/waist can occasionally be imputed for gap periods (older data) — imputation script is
  `scripts/impute_consumed.py` if that ever needs revisiting, not something to run routinely.

## Sleep / nocturia log

`data/sleep_log.csv` — a SEPARATE file from `health_data.csv`, its own protected header, never add
columns without a real reason:
```
date,pee_count,wake_time,urge,last_drink,alcohol,caffeine_pm,note
```
- `urge` values used so far: `none`, `bladder`, `mild-held`, `awake-anyway`, `dry-thirst`,
  `mind-racing`, `early-wake` — the point is distinguishing a true bladder-driven wake from an
  externally-triggered wake (baby, nightmare, overthinking) where he happens to pee while up. See
  `03_health_medical.md` for the full nocturia context and why this distinction matters.
- **Standing instruction: ask about the previous night's sleep every daily data-update if he
  doesn't volunteer it, and append one line.** Don't wait for him to remember.

## Apple Health XML export parsing (only needed for occasional bulk backfills, not daily use)

If Yannick ever provides a fresh full Apple Health `export.xml` (e.g. to backfill `move_kcal` or
refresh `data/recovery_history.csv` with RHR/HRV/VO2Max/sleep-stage data), use
`scripts/parse_apple_health_export.py` rather than hand-rolling a parser — it already handles the
gotchas below.

**The critical gotcha** (cost real debugging time to discover): Apple's `sourceName` attribute for
the watch uses a **non-breaking/narrow space**, not a normal space, between "Apple" and "Watch" —
so an exact-string filter on `"Apple Watch de Yannick"` silently matches **zero** records. **Fix:
filter on the substring `"Watch de Yannick"`** instead (the space after "Watch" is a normal one).
This also correctly excludes the ~35 `"iPhone de Yannick"` records that also contain "de Yannick".

Records are one per line, line-based parsing is fast (~40-60s for a ~2GB export) once the source
filter is right — don't sum across all sources for active energy (StrongLifts/Freeletics/other
devices double-count against the actual watch ring); **watch-only = the real Move ring value.**

## Streamlit dashboard

`frontend/app.py` is the entry point; `frontend/sections/` and `frontend/utils/` hold the dashboard
sections and helpers. Being evaluated for eventual replacement (Marimo was floated once) but no
active migration in progress — treat Streamlit as the live tool.

Data Entry and Weekly Report use the current WADP settings in `frontend/utils/wadp.py` (2,000
base; deficit 100 since Oct 9 2026). Weekly Report uses only days with both Move and intake for observed
deficit, compares week-end 7-day weight and 7/14-day waist averages, and counts distinct
date/session pairs from `data/training_log.csv`. Carried-forward travel measurements have no flag
in the CSV and remain in moving averages; verify them before making a diet decision. The Deep
Dive bulk tracker is historical and displays a warning because its old targets are retired.

**Weekly Report training section (Oct 10 2026):** `frontend/utils/training_metrics.py` computes
the cycle, hard-set and progress view of the selected week from `data/training_log.csv`; it is
shown on the page and in the HTML download. Cycle numbering follows the notes (a cycle is one pass
through the five sessions; a repeat or a complete cycle starts the next; cycle 9 = Sep 14, 10 =
Sep 19, 11 = Sep 25, 12 = Oct 5–10; the Sep 29–30 TTL tests are extras inside cycle 11 —
`KNOWN_CYCLE_STARTS` / `EXTRA_SESSION_DATES` in that file). Hard sets = RIR ≤ 3 or no RIR.
Progress is the best estimated 1RM versus the previous time the exercise was done; free weights
compare across gyms, machines/cables only within one gym, abs are excluded, bench counts from the
Sep 10 unassisted baseline, and TTL lateral raises are treated as the Flame Sport machine. The
Training Log page shares the same muscle classification (Glute Drive now counts as Glutes).
The same analytics feed two pages Yannick actually uses (Oct 10, `frontend/sections/training_views.py`):
the **Analytics Dashboard** has a compact Training block under its charts that follows the selected
time period (sessions, hard sets per week, PRs, share of lifts that went up, cycles completed,
hard sets per muscle per week, lifts up/flat/down per week, PR list), and the **Deep Dive** has a
"Training deep dive" with four tabs (Cycles, Muscles with freshness and a 12-week heatmap,
Exercises with a venue-aware drill-down and PR markers, Weekly overview joining hard sets and
progress with weight, waist, intake, Move, deficit, steps and sleep), placed after the waist
section. No composite strength index on purpose: a compounded index overstated progress
(~120 vs ~+7% real average), so progress is shown as lifts up/flat/down against each lift's own
previous time.

## Dev environment

- **Use `.venv/bin/...` for everything Python** — the repo's own virtualenv, not Poetry's (Poetry's
  environment exists per `pyproject.toml` but is empty/unused in practice).
- Formatter: black, line-length 100 (per `pyproject.toml` if invoked).
- Don't add new dependencies without checking `pyproject.toml` first.
