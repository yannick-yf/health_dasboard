"""Focused checks for weekly WADP arithmetic and pre-write snapshots."""

import tempfile
import unittest
import subprocess
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frontend"))

from backup_utils import create_backup
from data_utils import save_data
from utils.report_generator import generate_weekly_report_data
from utils.html_export import render_report_html
from scripts import merge_tracker_export as merger


class WeeklyReportTests(unittest.TestCase):
    def test_week_uses_move_and_intake_and_distinct_training_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "training.csv"
            log.write_text("date,session\n21/09/2026,Upper A\n21/09/2026,Upper A\n23/09/2026,Lower B\n")
            days = pd.to_datetime(["2026-09-14", "2026-09-15", "2026-09-21", "2026-09-22"])
            data = pd.DataFrame({
                "date": days, "move_kcal": [1000, 1000, 1000, None],
                "calories_consumed": [2800, 3000, 2700, 2600],
                "calories_burned": [9999] * 4, "weight": [73.0, 73.2, 72.8, 72.9],
                "waist_cm": [80.3, 80.1, 80.0, 79.9], "sleep_min": [420] * 4,
                "steps": [8000] * 4,
            })
            report = generate_weekly_report_data(data, pd.Timestamp("2026-09-21"), log)
            self.assertEqual(report["avg_deficit"], 300)
            self.assertEqual(report["prior_avg_deficit"], 100)
            self.assertEqual(report["deficit_days"], 1)
            self.assertEqual(report["sessions"], 2)
            self.assertAlmostEqual(report["weight_ma7"], 72.85)
            self.assertAlmostEqual(report["waist_ma14"], 80.075)
            html = render_report_html(report)
            self.assertIn("Observed WADP deficit", html)
            self.assertNotIn("Avg Surplus", html)


class BackupTests(unittest.TestCase):
    def test_streamlit_import_path_finds_backup_helper(self):
        result = subprocess.run(
            [sys.executable, "-c", "from data_utils import load_data"],
            cwd=ROOT / "frontend", capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_save_keeps_prior_contents_and_retention_is_per_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            health = root / "health_data.csv"
            training = root / "training_log.csv"
            health.write_text("old health\n")
            training.write_text("old training\n")
            for index in range(12):
                create_backup(health, keep=3)
                if index == 0:
                    create_backup(training, keep=3)
            self.assertEqual(len(list((root / "backups").glob("health_data_backup_*.csv"))), 3)
            self.assertEqual(len(list((root / "backups").glob("training_log_backup_*.csv"))), 1)
            frame = pd.DataFrame({"date": pd.to_datetime(["2026-10-01"]), "weight": [72.7]})
            save_data(frame, health)
            self.assertIn("01/10/2026", health.read_text())
            self.assertTrue(any(p.read_text() == "old health\n" for p in
                                (root / "backups").glob("health_data_backup_*.csv")))

    def test_merge_backups_only_when_a_csv_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            health = root / "health_data.csv"
            training = root / "training_log.csv"
            health.write_text(",".join(merger.HEALTH_COLS) + "\n,01/10/2026,,,,,,,,\n")
            training.write_text(",".join(merger.TRAINING_COLS) + "\n")
            old_health, old_training = merger.HEALTH_CSV, merger.TRAINING_CSV
            merger.HEALTH_CSV, merger.TRAINING_CSV = health, training
            try:
                record = [{"date": "01/10/2026", "move_kcal": 800}]
                merger.merge_health(record, dry_run=True)
                self.assertFalse((root / "backups").exists())
                merger.merge_health(record, dry_run=False)
                self.assertEqual(len(list((root / "backups").glob("health_data_backup_*.csv"))), 1)
                merger.merge_health(record, dry_run=False)
                self.assertEqual(len(list((root / "backups").glob("health_data_backup_*.csv"))), 1)
                workout = [{"date": "01/10/2026", "session": "Upper A",
                            "exercises": [{"name": "Bench", "sets": [{"weight": 60, "reps": 8}]}]}]
                merger.merge_workouts(workout, dry_run=False)
                self.assertEqual(len(list((root / "backups").glob("training_log_backup_*.csv"))), 1)
                merger.merge_workouts(workout, dry_run=False)
                self.assertEqual(len(list((root / "backups").glob("training_log_backup_*.csv"))), 1)
            finally:
                merger.HEALTH_CSV, merger.TRAINING_CSV = old_health, old_training


if __name__ == "__main__":
    unittest.main()
