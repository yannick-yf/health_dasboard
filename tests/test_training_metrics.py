"""Checks for cycles, venue-aware progression and weekly volume."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))

from utils import training_metrics as tm  # noqa: E402

REAL_LOG = ROOT / "data" / "training_log.csv"


def make_log(rows):
    """rows: (date dd/mm/yyyy, session, exercise, weight, reps, rir, venue)."""
    frame = pd.DataFrame(
        rows, columns=["date", "session", "exercise", "weight", "reps", "rir", "venue"]
    )
    frame["target"] = ""
    frame["set_number"] = 1
    frame["exercise_order"] = 1
    frame["note"] = frame["venue"].map(lambda v: f"[{v}]" if v else "")
    path = Path(tempfile.mkdtemp()) / "training_log.csv"
    frame.drop(columns="venue").to_csv(path, index=False)
    return tm.load_training_log(path)


def sessions_of(*pairs):
    frame = pd.DataFrame([{"dt": pd.Timestamp(d), "family": f} for d, f in pairs])
    frame["session"] = frame["family"]
    return frame


class CycleTests(unittest.TestCase):
    def test_cycle_closes_after_five_and_a_repeat_starts_a_new_one(self):
        s = sessions_of(
            ("2027-01-04", "Upper A"),
            ("2027-01-05", "Lower A"),
            ("2027-01-07", "Upper B"),
            ("2027-01-08", "Upper C"),
            ("2027-01-09", "Lower B"),
            ("2027-01-11", "Upper A"),
            ("2027-01-12", "Upper A"),
        )
        self.assertEqual(tm.assign_cycles(s)["cycle"].tolist(), [1, 1, 1, 1, 1, 2, 3])

    def test_known_trial_start_and_equipment_tests_match_the_notes(self):
        s = sessions_of(
            ("2026-09-17", "Lower B"),
            ("2026-09-19", "Upper A"),
            ("2026-09-25", "Upper B"),
            ("2026-09-26", "Lower B"),
            ("2026-09-28", "Upper A"),
            ("2026-09-29", "Lower B"),
            ("2026-09-30", "Upper B"),
            ("2026-10-01", "Lower A"),
            ("2026-10-02", "Upper C"),
            ("2026-10-05", "Upper A"),
        )
        cycled = tm.assign_cycles(s)
        self.assertEqual(cycled["cycle"].tolist()[1:], [2, 3, 3, 3, 3, 3, 3, 3, 4])
        self.assertEqual(cycled["extra"].tolist()[5:7], [True, True])

    def test_real_history_numbers_cycles_like_the_notes(self):
        if not REAL_LOG.exists():
            self.skipTest("no training log")
        cycled = tm.assign_cycles(tm.session_table(tm.load_training_log(REAL_LOG)))
        first = lambda day: int(cycled.loc[cycled["dt"] == day, "cycle"].iloc[0])
        self.assertEqual(first("2026-09-14"), 9)  # 4-day trial
        self.assertEqual(first("2026-09-19"), 10)  # opened with Upper A
        self.assertEqual(first("2026-09-25"), 11)  # started with Upper B
        self.assertEqual(first("2026-10-10"), 12)  # Oct 5-10, closed by Lower A
        summary = tm.cycles_in_range(
            tm.session_table(tm.load_training_log(REAL_LOG)), "2026-10-05", "2026-10-10"
        )[-1]
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["number"], 12)


class ProgressionTests(unittest.TestCase):
    def test_free_weights_compare_across_venues_but_machines_do_not(self):
        t = make_log(
            [
                ("01/10/2026", "Upper A", "Développé incliné haltères", 25, 8, 1, "Basic Fit"),
                ("08/10/2026", "Upper A", "Développé incliné haltères", 25, 8, 1, "ToTheLimitGym"),
                ("01/10/2026", "Upper A", "Tirage vertical prise large", 66, 9, 1, "Basic Fit"),
                ("08/10/2026", "Upper A", "Tirage vertical prise large", 70, 9, 1, "ToTheLimitGym"),
            ]
        )
        progress = tm.weekly_training_summary(t, "2026-10-05")["progress"].set_index("Exercise")
        self.assertEqual(progress.loc["Développé incliné haltères", "Status"], "flat")
        self.assertEqual(progress.loc["Tirage vertical prise large", "Status"], "new")
        self.assertEqual(progress.loc["Tirage vertical prise large", "Compared"], "ToTheLimitGym")

    def test_ttl_lateral_raise_is_a_machine_not_dumbbells(self):
        t = make_log(
            [
                ("01/10/2026", "Upper A", "Élévations latérales", 8, 12, 1, "Basic Fit"),
                ("08/10/2026", "Upper A", "Élévations latérales", 2.5, 11, 1, "ToTheLimitGym"),
            ]
        )
        row = tm.weekly_training_summary(t, "2026-10-05")["progress"].iloc[0]
        self.assertEqual((row["Compared"], row["Status"]), ("ToTheLimitGym", "new"))

    def test_status_labels(self):
        rows = [
            ("%02d/09/2026" % d, "Upper A", "Back squat barre", w, 5, 1, "Basic Fit")
            for d, w in [(1, 80), (8, 82.5), (15, 85)]
        ]
        rows += [("22/09/2026", "Upper A", "Back squat barre", 90, 5, 1, "Basic Fit")]
        self.assertEqual(
            tm.weekly_training_summary(make_log(rows), "2026-09-21")["progress"]["Status"].iloc[0],
            "PR",
        )
        flat = rows[:3] + [("22/09/2026", "Upper A", "Back squat barre", 85, 5, 2, "Basic Fit")]
        self.assertEqual(
            tm.weekly_training_summary(make_log(flat), "2026-09-21")["progress"]["Status"].iloc[0],
            "flat",
        )
        down = rows[:3] + [("22/09/2026", "Upper A", "Back squat barre", 75, 5, 1, "Basic Fit")]
        self.assertEqual(
            tm.weekly_training_summary(make_log(down), "2026-09-21")["progress"]["Status"].iloc[0],
            "down",
        )
        up = rows[:2] + [("22/09/2026", "Upper A", "Back squat barre", 84, 5, 1, "Basic Fit")]
        self.assertEqual(
            tm.weekly_training_summary(make_log(up), "2026-09-21")["progress"]["Status"].iloc[0],
            "up",
        )

    def test_bodyweight_is_added_to_dips_and_bench_before_unassisted_is_ignored(self):
        t = make_log(
            [
                ("05/09/2026", "Upper A", "Développé couché barre", 80, 5, 1, "Basic Fit"),
                ("12/09/2026", "Upper A", "Dips (lestés ou BW)", 10, 8, 1, "Basic Fit"),
            ]
        )
        weights = pd.Series([72.0], index=pd.to_datetime(["2026-09-01"]))
        best = tm.best_sets(t, weights)
        self.assertNotIn("Développé couché barre", best["exercise"].tolist())
        self.assertAlmostEqual(
            best.loc[best["exercise"].str.startswith("Dips"), "e1rm"].iloc[0], 82 * (1 + 8 / 30)
        )


class DeepDiveTests(unittest.TestCase):
    def setUp(self):
        rows = [
            ("%02d/09/2026" % d, "Upper A", "Back squat barre", w, 5, 1, "Basic Fit")
            for d, w in [(1, 80), (8, 82.5), (15, 85), (22, 90)]
        ]
        rows += [
            ("08/09/2026", "Lower A", "Weighted jackknife", 40, 12, 1, "Basic Fit"),
            ("08/09/2026", "Upper B", "Tirage vertical prise large", 66, 9, 1, "Basic Fit"),
            ("22/09/2026", "Upper B", "Tirage vertical prise large", 70, 9, 1, "ToTheLimitGym"),
        ]
        self.log = make_log(rows)
        self.history = tm.best_sets(self.log)

    def test_weekly_progress_counts_only_comparable_lifts(self):
        progress = tm.weekly_progress(self.history)
        self.assertEqual(int(progress["Up"].sum()), 3)  # 3 squat sessions beat the previous one
        self.assertEqual(
            int(progress[["Up", "Flat", "Down"]].sum().sum()), 3
        )  # gym change is not a comparison

    def test_pr_events_need_three_earlier_sessions(self):
        prs = tm.pr_events(self.history)
        self.assertEqual(prs["Date"].dt.strftime("%d/%m").tolist(), ["22/09"])
        self.assertEqual(tm.pr_events(self.history, start="2026-10-01").shape[0], 0)

    def test_muscle_freshness_counts_days_and_recent_hard_sets(self):
        fresh = tm.muscle_freshness(self.log, "2026-09-24").set_index("Muscle")
        self.assertEqual(fresh.loc["Quads", "Days since"], 2)
        self.assertEqual(fresh.loc["Quads", "Hard sets, last 7 days"], 1)
        self.assertEqual(fresh.loc["Abs", "Days since"], 16)

    def test_cycle_history_and_series_helpers(self):
        history = tm.cycle_history(tm.session_table(self.log))
        self.assertEqual(history["Cycle"].tolist(), [1, 2, 3, 4])
        labels = [option[0] for option in tm.series_options(self.history)]
        self.assertIn("Back squat barre", labels)
        self.assertIn("Tirage vertical prise large · ToTheLimitGym", labels)
        sessions = tm.series_sessions(self.log, "Back squat barre", "All venues")
        self.assertEqual(sessions["Sets"].iloc[-1], "90×5")

    def test_weekly_overview_joins_health_by_monday_week(self):
        health = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-09-22", "2026-09-23"]),
                "weight": [72.0, 72.2],
                "waist_cm": [80.0, 79.8],
                "calories_consumed": [2900, 3000],
                "move_kcal": [1000, 1000],
                "steps": [9000, 11000],
                "sleep_min": [420, 480],
            }
        )
        overview = tm.weekly_overview(self.log, health)
        week = overview.loc[pd.Timestamp("2026-09-21")]
        self.assertEqual(week["Sessions"], 2)
        self.assertAlmostEqual(week["Deficit"], 50.0)
        self.assertAlmostEqual(week["Weight"], 72.1)
        self.assertEqual(overview.loc[pd.Timestamp("2026-09-14"), "Sessions"], 1)


class VolumeTests(unittest.TestCase):
    def test_hard_sets_count_rir_up_to_three_and_missing_but_not_abs_in_progress(self):
        t = make_log(
            [
                ("08/10/2026", "Upper A", "Tirage vertical prise large", 60, 10, 3, "Basic Fit"),
                ("08/10/2026", "Upper A", "Tirage vertical prise large", 60, 9, 4, "Basic Fit"),
                ("08/10/2026", "Upper A", "Tirage vertical prise large", 60, 8, None, "Basic Fit"),
                ("08/10/2026", "Lower A", "Weighted jackknife", 40, 12, 1, "Basic Fit"),
            ]
        )
        summary = tm.weekly_training_summary(t, "2026-10-05")
        self.assertEqual(summary["volume"].loc["Back", "This week"], 2)
        self.assertEqual(summary["volume"].loc["Abs", "This week"], 1)
        self.assertNotIn("Weighted jackknife", summary["progress"]["Exercise"].tolist())

    def test_missing_or_incomplete_logs_give_none(self):
        self.assertIsNone(tm.load_training_log(Path(tempfile.mkdtemp()) / "nothing.csv"))
        path = Path(tempfile.mkdtemp()) / "partial.csv"
        path.write_text("date,session\n21/09/2026,Upper A\n")
        self.assertIsNone(tm.load_training_log(path))

    def test_real_week_summary_runs(self):
        if not REAL_LOG.exists():
            self.skipTest("no training log")
        summary = tm.weekly_training_summary(tm.load_training_log(REAL_LOG), "2026-10-05")
        self.assertEqual(len(summary["sessions"]), 5)
        self.assertGreater(summary["hard_sets"], 50)


if __name__ == "__main__":
    unittest.main()
