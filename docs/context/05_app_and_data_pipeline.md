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
    App v1.6/cache v6 was deployed to GitHub Pages and confirmed current on Yannick's phone Sep 29.
  - Program-data changes (adding/editing exercises/sets, not app logic) are pushed as a
    `yf-tracker-program-vX.json` file dropped in the iCloud sync folder (see below) and imported
    via the app's 📤 import panel — this is a separate mechanism from a code push.

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
(unnamed index), date, steps, sleep_min, workout_duration_min_tot, weight, calories_burned,
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

## Dev environment

- **Use `.venv/bin/...` for everything Python** — the repo's own virtualenv, not Poetry's (Poetry's
  environment exists per `pyproject.toml` but is empty/unused in practice).
- Formatter: black, line-length 100 (per `pyproject.toml` if invoked).
- Don't add new dependencies without checking `pyproject.toml` first.
