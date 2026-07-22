#!/usr/bin/env python3
"""
Parse Yannick's Apple Health `export.xml` into daily series.

WHY THIS SCRIPT EXISTS / THE GOTCHA THAT COST HOURS
---------------------------------------------------
The Apple Watch `sourceName` uses a NON-BREAKING space between "Apple" and
"Watch" (it renders like a normal space but is a different byte). So filtering
on the literal "Apple Watch de Yannick" (regular space) matches ZERO records
and every parser silently returns empty.
  --> ALWAYS filter on the substring "Watch de Yannick" (spaces after "Watch"
      are normal). This is the one thing that makes it work.

Other notes:
- The file is huge (~2 GB) but energy/HR/sleep Records are one line each
  (0 carriage returns). Line-based streaming is correct and fast (~60s).
- Do NOT sum all sources for active energy — StrongLifts/Freeletics/Huawei/old
  iPhone also log it and double-count the Move ring. Watch-only == the ring.
- Timestamps are local (offset embedded); the date prefix IS the local date.
- Sleep is a category type with value="HKCategoryValueSleepAnalysis<Stage>";
  duration = endDate - startDate; attribute each segment to a "night" via a
  +6h shift on the start time (so an evening→morning sleep lands on wake day).

Validation anchors (must match Yannick's known Move ring):
  2026-07-19 = 1173, 2026-07-20 = 1191, 2026-07-21 = 1273.

USAGE
  python3 scripts/parse_apple_health_export.py --recovery      # write data/recovery_history.csv
  python3 scripts/parse_apple_health_export.py --move-backfill # fill move_kcal in data/health_data.csv
  python3 scripts/parse_apple_health_export.py --validate      # print Move for the anchor days
Options: --xml PATH (default apple_health_export/export.xml), --start YYYY-MM-DD
         (default 2025-08-04, the health CSV start). Never auto-commits.
"""
import argparse
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_XML = REPO / "apple_health_export" / "export.xml"
HEALTH_CSV = REPO / "data" / "health_data.csv"
RECOVERY_CSV = REPO / "data" / "recovery_history.csv"
SOURCE_FILTER = "Watch de Yannick"      # <-- the fix; do NOT use "Apple Watch de Yannick"
DEFAULT_START = "2025-08-04"

_sd = re.compile(r'startDate="(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
_ed = re.compile(r'endDate="(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
_val = re.compile(r' value="([-0-9.]+)"')
_stage = re.compile(r'value="HKCategoryValueSleepAnalysis([A-Za-z]+)"')
_QTY = {
    "RestingHeartRate": "rhr",
    "HeartRateVariabilitySDNN": "hrv",
    "VO2Max": "vo2max",
    "ActiveEnergyBurned": "move",
    "BasalEnergyBurned": "bmr",
}


def parse(xml_path):
    """Single streaming pass. Returns (qsum, qcnt, sleep) dicts keyed by (metric,date)/(stage,night)."""
    qsum = defaultdict(float)
    qcnt = defaultdict(int)
    sleep = defaultdict(float)
    with open(xml_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if SOURCE_FILTER not in line:
                continue
            if "SleepAnalysis" in line:
                s, e, g = _sd.search(line), _ed.search(line), _stage.search(line)
                if s and e and g:
                    a = datetime.strptime(s.group(1), "%Y-%m-%d %H:%M:%S")
                    b = datetime.strptime(e.group(1), "%Y-%m-%d %H:%M:%S")
                    mins = (b - a).total_seconds() / 60
                    if 0 < mins <= 16 * 60:
                        night = (a + timedelta(hours=6)).date().isoformat()
                        sleep[(g.group(1), night)] += mins
                continue
            for t, key in _QTY.items():
                if "Identifier" + t in line:
                    d, v = _sd.search(line), _val.search(line)
                    if d and v:
                        date = d.group(1)[:10]
                        qsum[(key, date)] += float(v.group(1))
                        qcnt[(key, date)] += 1
                    break
    return qsum, qcnt, sleep


def daily_move(qsum):
    return {dt: round(val) for (k, dt), val in qsum.items() if k == "move"}


def write_recovery(qsum, qcnt, sleep, start, out=RECOVERY_CSV):
    dates = sorted({dt for (_, dt) in qsum} | {dt for (_, dt) in sleep})
    dates = [d for d in dates if d >= start]
    mean = lambda k, d: round(qsum[(k, d)] / qcnt[(k, d)], 1) if qcnt[(k, d)] else ""
    with open(out, "w") as f:
        f.write("date,rhr_bpm,hrv_sdnn_ms,vo2max,apple_bmr,"
                "sleep_asleep_min,sleep_awake_min,sleep_deep_min,sleep_rem_min\n")
        for d in dates:
            asleep = sum(sleep[(s, d)] for s in
                         ("AsleepCore", "AsleepDeep", "AsleepREM", "AsleepUnspecified"))
            f.write(f"{d},"
                    f"{round(qsum[('rhr', d)] / qcnt[('rhr', d)]) if qcnt[('rhr', d)] else ''},"
                    f"{mean('hrv', d)},{mean('vo2max', d)},"
                    f"{round(qsum[('bmr', d)]) if ('bmr', d) in qsum else ''},"
                    f"{round(asleep) if asleep else ''},"
                    f"{round(sleep[('Awake', d)]) if sleep[('Awake', d)] else ''},"
                    f"{round(sleep[('AsleepDeep', d)]) if sleep[('AsleepDeep', d)] else ''},"
                    f"{round(sleep[('AsleepREM', d)]) if sleep[('AsleepREM', d)] else ''}\n")
    return len(dates), (dates[0], dates[-1]) if dates else (None, None)


def move_backfill(move, cutoff):
    """Append/fill move_kcal in health_data.csv for dates <= cutoff. Preserves everything else."""
    raw = HEALTH_CSV.read_text(encoding="utf-8")
    trail = raw.endswith("\n")
    lines = raw.splitlines()
    header = lines[0]
    has_col = "move_kcal" in header.split(",")
    out = [header if has_col else header + ",move_kcal"]
    filled = 0
    for ln in lines[1:]:
        if not ln.strip():
            continue
        cols = ln.split(",")
        iso = "-".join(reversed(cols[1].strip().split("/")))
        cur = cols[-1] if has_col else ""
        if iso <= cutoff and iso in move and (not has_col or cur == ""):
            out.append((",".join(cols[:-1]) if has_col else ln) + "," + str(float(move[iso])))
            filled += 1
        else:
            out.append(ln if has_col else ln + ",")
    HEALTH_CSV.write_text("\n".join(out) + ("\n" if trail else ""), encoding="utf-8")
    return filled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--recovery", action="store_true", help="write data/recovery_history.csv")
    ap.add_argument("--move-backfill", action="store_true", help="fill move_kcal in health_data.csv")
    ap.add_argument("--validate", action="store_true", help="print Move for anchor days")
    args = ap.parse_args()
    if not args.xml.exists():
        raise SystemExit(f"export not found: {args.xml}")

    print(f"parsing {args.xml} (filter source ~ '{SOURCE_FILTER}') ...")
    qsum, qcnt, sleep = parse(args.xml)
    move = daily_move(qsum)
    print(f"  move days={len(move)}  ({min(move)} → {max(move)})")

    if args.validate or not (args.recovery or args.move_backfill):
        for k in ("2026-07-19", "2026-07-20", "2026-07-21"):
            print(f"  validate {k}: move={move.get(k, 'NA')} (expect 1173/1191/1273)")
    if args.recovery:
        n, rng = write_recovery(qsum, qcnt, sleep, args.start)
        print(f"  wrote {RECOVERY_CSV.name}: {n} days {rng[0]}→{rng[1]}")
    if args.move_backfill:
        # cutoff = latest COMPLETE day = max move date minus today-partial guard handled by caller
        cutoff = max(move)
        n = move_backfill(move, cutoff)
        print(f"  backfilled move_kcal into {HEALTH_CSV.name}: {n} rows (cutoff {cutoff})")
    print("done (no commit — review changes yourself).")


if __name__ == "__main__":
    main()
