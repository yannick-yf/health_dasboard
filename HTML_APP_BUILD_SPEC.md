# Build Spec — Local Health & Training Tracker (single-file HTML app)

**Branch**: `streamlit-to-real-app`
**Target file to create**: `mobile_tracker/index.html` (one self-contained file)
**Author of spec**: prior session (Opus). **Builder**: Fable 5.
**Owner**: Yannick — English only. Direct, honest engineering. No over-accommodation.

---

## 1. Goal & context

Build a **single self-contained HTML file** that runs **fully offline** on both an **iPhone (Add to Home Screen, PWA-style)** and a **Mac browser**. It replaces the daily habit of launching Streamlit just to enter data, and adds gym workout tracking. Heavy weekly analysis stays in Streamlit + Claude Code on the Mac — this app is for **daily entry, gym program tracking, and light at-a-glance trends**.

The user tried and failed to model his program (esp. supersets) in the StrongLifts app. This app must handle supersets cleanly.

### Sync model — LOCKED DECISION (do not deviate)
Data is NOT baked into the HTML file. The app stores data locally (IndexedDB) and syncs by exporting a small **JSON** file to iCloud Drive; the Mac merges it into the CSVs (§6, §11). We explicitly rejected "data embedded in the HTML / both devices write the same file" because iOS browsers cannot write back to a file, which would make the phone read-only and risk last-writer-wins data loss. App (stable HTML) and data (tiny JSON) stay decoupled.

### Why single-file / offline (hard requirements from research)
- iOS Safari does **NOT** support the File System Access API — a web page cannot write directly to a shared file. So sync is via **explicit export** (see §6).
- iOS may **evict** localStorage/IndexedDB after ~7 days of no use. Mitigate: use IndexedDB, call `navigator.storage.persist()`, show an "unsaved since last export" badge, and support **import** of a previously exported file.
- Must work with **no network**: inline ALL CSS and JS. **No CDN, no external fonts, no analytics, no login, no cloud.** No build step, no npm. Just one `.html` file that opens by double-click / tap.

---

## 2. Tech constraints (non-negotiable)

1. One file: `mobile_tracker/index.html`. Everything inline.
2. Vanilla HTML/CSS/JS only. No frameworks, no external requests of any kind.
3. Charts: draw with the Canvas API OR inline a tiny (<15KB) charting routine written by hand. No chart CDN.
4. Storage: **IndexedDB** as primary store (localStorage acceptable fallback). Call `navigator.storage.persist()` on load.
5. PWA basics: `<meta name="apple-mobile-web-app-capable" content="yes">`, `<meta name="viewport" ...>`, `apple-mobile-web-app-status-bar-style`, a title, and an inline `theme-color`. Add-to-Home-Screen must give an app-like standalone launch. (An inline base64 icon is nice-to-have, not required.)
6. **Mobile-first**, dark theme, **large tap targets** (min 44px), high contrast — usable at the gym with sweaty hands / one thumb.
7. Fast: opens instantly, no spinner.

---

## 3. Data model

### 3.1 Health metrics — must map 1:1 to the existing CSV
Existing CSV header (`data/health_data.csv`), **do not change column names or order**:
```
user_id,date,steps,sleep_min,workout_duration_min_tot,weight,calories_burned,calories_consumed,waist_cm
```
- `user_id`: leave empty (CSV rows start with a bare comma).
- `date`: **DD/MM/YYYY** (e.g. `18/07/2026`). This format is mandatory — the Mac CSV uses it.
- All numeric fields are floats or blank. Blank ≠ 0 (missing data stays blank).
- Typical daily entry from the user fills: `date`, `steps`, `sleep_min`, `workout_duration_min_tot`, `weight`, `calories_burned` (Apple Watch "Total"), `calories_consumed`, `waist_cm`. Any can be blank.

Internal storage record (per day):
```json
{ "date": "18/07/2026", "steps": 14484, "sleep_min": 530, "workout_duration_min_tot": 124,
  "weight": 73.9, "calories_burned": 3195, "calories_consumed": 3145, "waist_cm": 82.0 }
```

### 3.2 Workout logs — new file `data/training_log.csv` (Claude will create/merge on Mac)
Internal storage record (per session):
```json
{ "date": "19/07/2026", "session": "Upper A",
  "exercises": [
    { "name": "Tirage vertical prise large", "target": "4x8-10", "rest": "2:30", "rir": "1-2",
      "sets": [ {"weight": 52, "reps": 12}, {"weight": 52, "reps": 10}, {"weight": 46, "reps": 10}, {"weight": 46, "reps": 9} ],
      "note": "BF matrix" }
  ] }
```

---

## 4. Screens / features

Three tabs (bottom nav, thumb-reachable): **Entry**, **Train**, **Trends**.

### 4.1 Entry (daily health metrics) — SIMPLE, mirror the Streamlit form
Keep this deliberately simple; match the existing Streamlit `data_entry` page behavior:
- Date picker defaulting to **today** (DD/MM/YYYY). Selecting an existing date loads it in **edit mode** (show a small "Date exists — editing" vs "New date" hint, like Streamlit does).
- Number inputs for all CSV fields (§3.1), grouped like Streamlit: **Activity & Sleep** (steps, sleep, workout minutes) and **Body & Nutrition** (weight, calories_burned = Apple Watch "Total", calories_consumed, waist_cm). Blank allowed. Mobile keyboards via `inputmode="decimal"`.
- **Weight defaults to the last known weight** from stored data (Streamlit does this).
- Convenience: sleep enterable as `Hh MM` and stored as minutes (e.g. `8h50` → 530). Show both.
- **Save** writes/updates the day record in IndexedDB and marks "unsaved since last export".
- Below the form: a scrollable list of the last ~10 days (editable rows) so mistakes are easy to fix.
- (Optional, nice-to-have) a small "today's WADP target" readout is NOT required — keep entry lean.

### 4.2 Train (program tracking — YF-UL5 v1) — BEST-IN-CLASS, fully editable
This is the flagship screen. Model the UX on best-in-class lifting apps (Hevy / Strong): fast, thumb-friendly set logging.
- Hardcode the 5 session templates from §5 as the **starting** program, but the program must be **fully editable** by the user:
  - **Add / remove / rename / reorder exercises** within any session.
  - **Edit targets** (sets, rep range, rest, RIR) per exercise.
  - **Create custom exercises** and **swap** an exercise for another.
  - **Create/rename/delete sessions**, and mark supersets (pair two exercises).
  - All edits persist in IndexedDB (the program config is user data, not frozen code).
- Logging flow: **pick today's session** → exercise list in order with target sets×reps, rest, RIR → for each exercise, add set rows (**weight × reps**, optional RIR), tick sets done. **Prefill weight/reps from the last time that exercise was logged** (progression aid) — this is the core best-in-class behavior.
- **Supersets**: paired exercises render inside one visually grouped card (left vertical accent bar + "SUPERSET" label), alternating set entry. This is the StrongLifts gap — make pairing obvious and easy to set/unset.
- Optional per-exercise **rest timer** (countdown from the template's rest value; `navigator.vibrate` + optional visual flash; no audio file).
- Free-text **note** per exercise (gym/machine variant, e.g. "BF Martigues poulie normale").
- **Save session** stores the workout record (§3.2) and marks "unsaved".
- Do NOT compute `workout_duration_min_tot` — that comes from Apple Watch via the Entry tab.

### 4.3 Trends (mirror the Streamlit Analytics Dashboard)
Reproduce the look/behavior of the Streamlit `analytics_dashboard` page, adapted to mobile (stack vertically):
- **Time-period selector**: Last 30 Days / Last 3 Months / All Data (default Last 30 Days).
- **Aggregation selector**: Daily / Weekly / Monthly.
- **6 KPI cards** with the SAME green/amber/red threshold coloring as Streamlit:
  - Current Weight (kg, neutral) · Weekly Gain (kg/week; green 0.20–0.50, amber 0.10–0.20 & 0.50–0.70, else red) · Daily Surplus (kcal = consumed − burned; green 200–600, amber 100–200 & 600–900, else red) · Avg Sleep (hours; green ≥7.5, amber 7.0–7.5) · Avg Steps (green ≥8000, amber 5000–8000) · Avg Workout (min/day; green ≥60, amber 30–60).
- **4 charts** (Canvas), matching Streamlit's 2×2 grid, stacked on mobile:
  1. Weight over time (with moving-average line).
  2. Energy balance (consumed vs burned / surplus).
  3. Steps + workout minutes.
  4. Sleep.
- Add a **waist** line with its 14-day MA (waist is the user's primary success metric; Streamlit's deep-dive tracks it — surface it here too).
- Keep heavy analysis (correlations, regressions, TDEE deep-dive) OUT — that stays in Streamlit + Claude.

### 4.4 Training progress view — PHASE 2 (defer)
A dedicated view of per-exercise progression over time (e.g., top-set weight trend, volume per session) is **deferred to a later build**. Note it in the UI as "coming soon" or omit. Do not block v1 on it.

---

## 5. YF-UL5 v1 program (hardcode these templates)

Schedule (for reference/labels): Sun Upper A · Mon Lower A · Tue Upper B · Wed REST · Thu Upper C · Fri Lower B · Sat REST.
Format below: Exercise — target sets×reps — rest — RIR. "SS" = superset pair (group visually).

### Upper A — Chest/Back Heavy
1. Tirage vertical prise large — 4×8-10 — 2:30 — 1-2
2. Tirage horizontal prise serré — 4×10 — 2:30 — 1-2
3. Développé couché barre — 4×5-8 — 3:00 — 1-2
4. Tirage vertical unilatéral — 3×10 — 2:00 — 2
5. Dips (lestés ou BW) — 3×8-10 — 2:00 — 2
6. SS: Shrug barre + Élévations latérales — 3×12-15 each — 1:30 — 1
7. Face pulls — 3×12-15 — 1:00 — 1

### Lower A — Quad/Ham + Abs
1. Back squat barre — 4×5-8 — 3:00 — 2
2. Bulgarian split squat — 4×10/leg — 2:00 — 2
3. Romanian deadlift barre — 4×8 — 3:00 — 2 (note: start light, ramping)
4. AB ROUTINE (see §5b)

### Upper B — Shoulders/Arms
1. Standing OHP barbell — 4×5-8 — 3:00 — 2
2. Élévations latérales — 4×10-15 — 1:30 — 1
3. Rear delt cable fly — 3×12-15 — 1:30 — 1
4. SS: Curl pupitre + Overhead triceps — 4×10 each — 1:30 — 1-2
5. SS: Curl allongé poulie basse + Triceps barre poulie haute — 4×10 each — 1:30 — 1-2
6. SS: Reverse curl EZ + Hammer curl — 3×10 each — 1:30 — 1
7. Wrist curls barre — 3×15 — 1:00 — 1

### Upper C — Chest/Back Volume
1. Développé incliné machine — 3×10 — 2:30 — 1-2
2. Écarté haltère pec — 3×10-12 — 1:30 — 1
3. Pull-ups (BW ou lestés) — 4×max (target 5-8) — 2:30 — 0-1
4. Tirage horizontal prise large — 3×10 — 2:30 — 1-2
5. Pull-over cable — 3×12 — 1:30 — 1
6. SS: Shrug barre + Élévations latérales — 3×12-15 each — 1:30 — 1

### Lower B — Ham/Quad Accessories + Abs
1. Hack squat — 4×10 — 2:30 — 1-2
2. Leg curl allongé — 3×10 — 1:30 — 1
3. Leg curl assis — 2×10 — 1:30 — 1
4. Leg extension — 3×10-12 — 1:30 — 0-1
5. Cable pull-through (optional) — 3×12 — 1:30 — 1
6. AB ROUTINE (see §5b)

### §5b AB ROUTINE (used by Lower A and Lower B; allow rotation)
1. Weighted jackknife (own variation) — 3×12 — 60s — (default 34 kg)
2. Russian twist — 3×20 — 45s — (default 20 kg)
3. Cable crunch — 3×15 — 45s — progressive
4. Leg raise slow — 1×25 (or 2×15) — 45s — BW
5. Plank hold — 2×45-60s — 45s — BW

**Program should be defined in a single JS config object** near the top of the file so it's easy to edit later (versioning: label it `YF-UL5 v1`).

---

## 6. Export / import / sync

### Export (the sync mechanism)
- Prominent **Export** button (and show "unsaved since last export" badge/counter).
- Produces **one JSON file** containing both domains:
```json
{ "app": "yf-tracker", "version": 1, "exported_at": "<ISO string>",
  "health": [ {…§3.1 records…} ],
  "workouts": [ {…§3.2 records…} ] }
```
- Trigger download via a Blob + `<a download>`; ALSO offer `navigator.share()` (Web Share API) when available so iOS can "Save to Files → iCloud Drive" or AirDrop to Mac.
- Filename: `yf-tracker-export-YYYY-MM-DD.json`.
- Also offer a **CSV export of the health table** (exact header from §3.1, DD/MM/YYYY) as a convenience — optional but nice.

### Import (eviction / device recovery)
- **Import** button: load a previously exported JSON back into IndexedDB (merge by date; imported values fill/overwrite). This protects against the 7-day eviction and lets the user move devices.

### Merge on the Mac (done by Claude, NOT by this app — documented here for clarity)
- `health[]` → merged into `data/health_data.csv`, dedupe by `date`, keep existing non-blank values unless the export has a value.
- `workouts[]` → appended into `data/training_log.csv` (Claude creates schema on first merge).

---

## 7. UX details

- Dark theme, large fonts, big buttons. One-thumb operation.
- Persistent "unsaved since last export: N changes" indicator in the header.
- Confirm before overwriting an existing day's entry.
- Never lose data silently: all writes go to IndexedDB immediately.
- No destructive action without confirm (e.g., delete a day / clear all).
- Show the app version and `YF-UL5 v1` label somewhere small.

---

## 8. Acceptance checklist (Fable 5: verify before done)

1. Opens offline (disable network) on Mac browser AND iPhone Safari; no failed network requests in console.
2. "Add to Home Screen" launches standalone (no Safari chrome).
3. Enter a day → Save → reload page → data still present (IndexedDB persists).
4. `navigator.storage.persist()` is requested on load.
5. Export produces valid JSON matching §6 schema; re-import restores identical data.
6. Health CSV export uses the exact header and DD/MM/YYYY dates.
7. Train tab: all 5 sessions present with correct exercises/sets/rest/RIR from §5; supersets visually grouped; last-logged weights prefill.
8. Trends tab: weight (MA7) and waist (MA14) charts render from entered data; latest-value cards correct.
9. Single file, no external requests, no build step.
10. Works one-handed on a phone-width viewport (test ~390px wide).

---

## 9. Out of scope (do NOT build)
- No accounts/auth, no cloud database, no server.
- No automatic writing to `health_data.csv` from the phone (impossible on iOS; merged on Mac — see §11).
- No heavy analytics/correlations in the app (stays in Streamlit + Claude).
- No external libraries or fonts.
- Training progress-over-time view = phase 2 (§4.4).

## 10. How it installs & runs on iPhone (must be documented in-app + here)

The app is just a file, but here is the exact user flow — include a short "How to install" note inside the app (a small Help/ⓘ panel):

**First install (once):**
1. Get `index.html` onto the phone. Easiest: put it in **iCloud Drive** (Mac side) → it appears in the iPhone **Files** app. (Or AirDrop it.)
2. On iPhone, open the file — it opens in Safari. (From Files: tap the file → Share → open in Safari, or host it once from the Mac over LAN to open it in Safari.)
3. In Safari: **Share button → "Add to Home Screen"**. Now it's an app icon that launches **standalone** (no Safari bars), works **offline**.

> Note on the cleanest install: because iOS opening a local `.html` straight from Files into Safari can be fiddly, the most reliable one-time install is to open it once via Safari (LAN URL from the Mac, e.g. `http://<mac-ip>:8000/index.html` served by `python3 -m http.server`), then Add to Home Screen. After that it runs fully offline from the icon. Document both paths; recommend the LAN-open-once path.

**Daily use:** tap the home-screen icon → log entry / workout offline → done. Tap **Export** whenever (daily or before the weekly review) → **Save to Files → iCloud Drive** (or AirDrop to Mac).

**Persistence caveat (tell the user in the Help panel):** iOS may clear the app's local data after ~7 days of NOT opening it. Because the app is used daily this is unlikely, but the **Export is also the backup** — exporting regularly guarantees no loss, and **Import** (§6) restores everything if it ever happens or on a new phone.

## 11. Mac-side merge component (part of THIS repo, not the HTML file)

The user asked the repo to "look at the file regularly and check for updates." Build a small Python script (respect the no-scheduled-agents preference — this is run on demand, not a background daemon):

**`scripts/merge_tracker_export.py`**
- Config: an `EXPORT_DIR` pointing at the iCloud Drive folder where exports land (e.g. `~/Library/Mobile Documents/com~apple~CloudDocs/yf-tracker/`). Make it a top-of-file constant / CLI arg.
- Behavior: find the **newest** `yf-tracker-export-*.json` in `EXPORT_DIR`, then:
  - Merge `health[]` into `data/health_data.csv`: dedupe by `date` (DD/MM/YYYY); for an existing date, fill blanks and update provided values; **never drop existing columns or reorder the header** (§3.1). Append new dates in chronological order.
  - Merge `workouts[]` into `data/training_log.csv` (create with a sensible header on first run; append new sessions; dedupe by date+session).
  - Print a summary: N health rows added/updated, N workouts added. **Do not auto-commit** (user rule).
- Idempotent: re-running on the same export changes nothing.
- Optional convenience: a button in the Streamlit `data_entry` page — "📥 Import from phone export" — that calls this merge and shows the summary. (Optional; the script alone is enough.)

**Workflow tie-in:** this runs at the weekly review (or any time the user is on the Mac), so the phone→iCloud→CSV loop closes exactly when Yannick "comes back here." No background scheduling.

## 12. Acceptance additions (beyond §8)
11. Program is fully editable (add/remove/rename/reorder/swap exercises, edit targets, create custom exercises, set/unset supersets) and edits persist across reload.
12. Entry tab visually parallels the Streamlit form (grouped fields, weight prefill, edit-vs-new hint).
13. Trends tab reproduces the 6 threshold-colored KPI cards + time-period + aggregation + the 4 charts (plus waist MA14).
14. `scripts/merge_tracker_export.py` merges a sample export into the CSVs correctly, dedupes by date, preserves the header, and is idempotent.
15. In-app Help panel documents the iPhone install + export + persistence caveat.
