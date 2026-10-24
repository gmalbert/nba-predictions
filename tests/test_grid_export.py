"""Feed health and pick-side regressions using real parquet artifacts."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import export_best_bets as exporter


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 24, 16, tzinfo=timezone.utc).astimezone(tz)


class GridExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "data_files/best_bets_today.json"
        self.output.parent.mkdir(parents=True)
        self.patches = [patch.object(exporter, "OUT_PATH", self.output), patch.object(exporter, "datetime", Clock)]
        for item in self.patches:
            item.start()
        self.cwd = Path.cwd()
        import os
        os.chdir(self.root)

    def tearDown(self):
        import os
        os.chdir(self.cwd)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def payload(self):
        return json.loads(self.output.read_text(encoding="utf-8"))

    def save(self, frame, day="2026-10-24"):
        path = self.root / f"data_files/historical/predictions_{day}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def test_missing_predictions_replace_old_export_with_pending_health(self):
        self.output.write_text('{"meta":{"generated_at":"2026-06-09T11:00:00Z"},"bets":[]}', encoding="utf-8")
        self.assertEqual(exporter.main(), 1)
        self.assertEqual(self.payload()["meta"]["status"], "pipeline_pending")
        self.assertTrue(self.payload()["meta"]["generated_at"].startswith("2026-10-24"))

    def test_successful_empty_artifact_is_no_picks(self):
        self.save(pd.DataFrame())
        self.assertEqual(exporter.main(), 0)
        self.assertEqual(self.payload()["meta"]["status"], "no_picks")

    def test_undated_yesterday_rows_are_never_relabelled_as_today(self):
        self.save(pd.DataFrame([{"home_team": "Home", "away_team": "Away", "home_win_prob": .7}]), day="2026-10-23")
        self.assertEqual(exporter.main(), 1)
        self.assertEqual(self.payload()["bets"], [])

    def test_away_value_pick_uses_away_probability_and_positive_edge(self):
        self.save(pd.DataFrame([{"home_team": "Home", "away_team": "Away", "home_win_prob": .35, "edge": -.04, "confidence": "Low", "predicted_spread": -8}]))
        self.assertEqual(exporter.main(), 0)
        bet = self.payload()["bets"][0]
        self.assertEqual((bet["pick"], bet["confidence"], bet["edge"], bet["tier"]), ("Away", .65, .04, "Strong"))
        self.assertIsNone(bet["line"])

    def test_nonfinite_fields_are_json_null_and_unpriced_picks_are_standard(self):
        self.save(pd.DataFrame([{"home_team": "Home", "away_team": "Away", "home_win_prob": .7, "edge": float("nan"), "odds": float("inf")}]))
        self.assertEqual(exporter.main(), 0)
        bet = self.payload()["bets"][0]
        self.assertIsNone(bet["edge"])
        self.assertIsNone(bet["odds"])
        self.assertEqual(bet["tier"], "Standard")

    def test_corrupt_parquet_is_failure(self):
        path = self.root / "data_files/historical/predictions_2026-10-24.parquet"
        path.parent.mkdir(parents=True)
        path.write_text("broken")
        self.assertEqual(exporter.main(), 1)
        self.assertEqual(self.payload()["meta"]["status"], "pipeline_failed")


if __name__ == "__main__":
    unittest.main()
