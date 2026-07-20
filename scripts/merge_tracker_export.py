#!/usr/bin/env python3
"""
Merge a yf-tracker phone export (JSON) into the repo CSVs.

- health[]   -> data/health_data.csv   (dedupe by date DD/MM/YYYY; fill blanks and
                update provided values; header/columns NEVER changed; new dates
                inserted in chronological order)
- workouts[] -> data/training_log.csv  (created on first run; one row per set;
                dedupe by (date, session): an already-present pair is skipped)

Usage:
    python3 scripts/merge_tracker_export.py                # newest export in EXPORT_DIR
    python3 scripts/merge_tracker_export.py --file X.json  # explicit export file
    python3 scripts/merge_tracker_export.py --export-dir DIR
    python3 scripts/merge_tracker_export.py --dry-run

Idempotent: re-running on the same export changes nothing.
Never commits anything (per user rule: no auto-commit of data files).
Standard library only.
"""

import argparse
import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

# Where phone exports land (iCloud Drive). Override with --export-dir or --file.
EXPORT_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/yf-tracker"

REPO_ROOT = Path(__file__).resolve().parent.parent
HEALTH_CSV = REPO_ROOT / "data" / "health_data.csv"
TRAINING_CSV = REPO_ROOT / "data" / "training_log.csv"

HEALTH_COLS = [
    "user_id", "date", "steps", "sleep_min", "workout_duration_min_tot",
    "weight", "calories_burned", "calories_consumed", "waist_cm",
]
# columns the export may update (everything except user_id/date)
UPDATABLE = HEALTH_COLS[2:]

TRAINING_COLS = [
    "date", "session", "exercise_order", "exercise", "target",
    "set_number", "weight", "reps", "rir", "note",
]


def parse_ddmm(s):
    return datetime.strptime(s, "%d/%m/%Y")


def fmt_cell(v):
    """Format a numeric export value the way the CSV stores numbers ('11318.0', '73.9')."""
    if v is None or v == "":
        return ""
    return str(float(v))


def find_export(export_dir, explicit):
    if explicit:
        if not explicit.exists():
            sys.exit(f"Export file not found: {explicit}")
        return explicit
    if not export_dir.is_dir():
        sys.exit(f"Export dir not found: {export_dir}\n"
                 f"Pass --export-dir or --file, or create the folder and drop an export in it.")
    candidates = sorted(export_dir.glob("yf-tracker-export-*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(f"No yf-tracker-export-*.json in {export_dir}")
    return candidates[-1]


def load_export(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("app") != "yf-tracker":
        sys.exit(f"{path.name}: not a yf-tracker export (app={data.get('app')!r})")
    data.setdefault("health", [])
    data.setdefault("workouts", [])
    return data


def merge_health(export_health: list, dry_run: bool):
    """Returns (added, updated). Preserves header verbatim and untouched cells byte-for-byte."""
    raw = HEALTH_CSV.read_text(encoding="utf-8")
    had_trailing_nl = raw.endswith("\n")
    lines = raw.splitlines()
    header, body = lines[0], lines[1:]

    if [h.strip() for h in header.split(",")] != HEALTH_COLS:
        sys.exit(f"Unexpected header in {HEALTH_CSV.name} — aborting to protect the file:\n{header}")

    rows = [next(csv.reader(io.StringIO(line))) if line else [] for line in body]
    rows = [r for r in rows if r and len(r) >= 2 and r[1].strip()]
    # pad short rows to full width (defensive)
    for r in rows:
        while len(r) < len(HEALTH_COLS):
            r.append("")

    by_date = {r[1]: r for r in rows}
    added = updated = 0

    for rec in export_health:
        d = (rec or {}).get("date")
        if not d:
            continue
        try:
            parse_ddmm(d)
        except ValueError:
            print(f"  ! skipping bad date in export: {d!r}")
            continue

        if d in by_date:
            row = by_date[d]
            changed = False
            for idx, col in enumerate(HEALTH_COLS):
                if col not in UPDATABLE:
                    continue
                v = rec.get(col)
                if v is None:
                    continue  # export has no value -> keep existing cell untouched
                new_cell = fmt_cell(v)
                if row[idx] != new_cell:
                    row[idx] = new_cell
                    changed = True
            if changed:
                updated += 1
        else:
            new_row = [""] * len(HEALTH_COLS)
            new_row[1] = d
            for idx, col in enumerate(HEALTH_COLS):
                if col in UPDATABLE and rec.get(col) is not None:
                    new_row[idx] = fmt_cell(rec[col])
            # insert keeping chronological order (after last row with date <= new date)
            nd = parse_ddmm(d)
            pos = len(rows)
            for i in range(len(rows) - 1, -1, -1):
                try:
                    if parse_ddmm(rows[i][1]) <= nd:
                        pos = i + 1
                        break
                except ValueError:
                    continue
            else:
                pos = 0
            rows.insert(pos, new_row)
            by_date[d] = new_row
            added += 1

    if (added or updated) and not dry_run:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        for r in rows:
            w.writerow(r)
        out = header + "\n" + buf.getvalue()
        if not had_trailing_nl:
            out = out.rstrip("\n")
        HEALTH_CSV.write_text(out, encoding="utf-8")
    return added, updated


def merge_workouts(export_workouts: list, dry_run: bool):
    """Returns (added_sessions, skipped_dup). One CSV row per set; dedupe on (date, session)."""
    existing_pairs = set()
    file_exists = TRAINING_CSV.exists()
    if file_exists:
        with open(TRAINING_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_pairs.add((row.get("date", ""), row.get("session", "")))

    new_rows, added, skipped = [], 0, 0
    for w in export_workouts:
        d, s = (w or {}).get("date"), (w or {}).get("session")
        if not d or not s:
            continue
        if (d, s) in existing_pairs:
            skipped += 1
            continue
        existing_pairs.add((d, s))
        for ei, exo in enumerate(w.get("exercises", []), start=1):
            # a performed set must have reps; drop template rows whose weight was
            # pre-filled but never actually done (reps blank) so they don't pollute the log
            real_sets = [st for st in exo.get("sets", []) if st.get("reps") not in (None, "")]
            for si, st in enumerate(real_sets, start=1):
                new_rows.append({
                    "date": d, "session": s,
                    "exercise_order": ei, "exercise": exo.get("name", ""),
                    "target": exo.get("target", ""),
                    "set_number": si,
                    "weight": "" if st.get("weight") is None else st.get("weight"),
                    "reps": st.get("reps"),
                    "rir": "" if st.get("rir") is None else st.get("rir"),
                    "note": exo.get("note", ""),
                })
        added += 1

    if new_rows and not dry_run:
        with open(TRAINING_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TRAINING_COLS, lineterminator="\n")
            if not file_exists:
                w.writeheader()
            w.writerows(new_rows)
    return added, skipped


def main():
    ap = argparse.ArgumentParser(description="Merge yf-tracker phone export into repo CSVs")
    ap.add_argument("--file", type=Path, help="explicit export JSON (skips dir scan)")
    ap.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    export_path = find_export(args.export_dir, args.file)
    data = load_export(export_path)

    print(f"Export : {export_path}")
    print(f"        exported_at={data.get('exported_at', '?')}  "
          f"health={len(data['health'])}  workouts={len(data['workouts'])}")
    if args.dry_run:
        print("        (dry run — nothing will be written)")

    added, updated = merge_health(data["health"], args.dry_run)
    w_added, w_skipped = merge_workouts(data["workouts"], args.dry_run)

    print(f"Health : {added} row(s) added, {updated} row(s) updated -> {HEALTH_CSV.relative_to(REPO_ROOT)}")
    print(f"Train  : {w_added} session(s) added, {w_skipped} duplicate(s) skipped -> {TRAINING_CSV.relative_to(REPO_ROOT)}")
    if not (added or updated or w_added):
        print("Nothing to do — CSVs already up to date (idempotent).")
    print("Reminder: review changes and commit manually (no auto-commit).")


if __name__ == "__main__":
    main()
