"""Regression checks for failed provider refreshes and season rollover."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import preload_cache
from utils import data_fetcher


class FeedPipelineTests(unittest.TestCase):
    def test_season_rolls_over_in_october(self):
        for at, expected in [(datetime(2026, 9, 30), "2025-26"), (datetime(2026, 10, 1), "2026-27"), (datetime(2027, 1, 1), "2026-27")]:
            self.assertEqual(data_fetcher.current_season(at), expected)

    def test_failed_force_refresh_retains_valid_disk_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stats.parquet"
            pd.DataFrame({"PTS": [110]}).to_parquet(path)
            before = path.read_bytes()
            def fail():
                raise TimeoutError("provider timeout")
            result = data_fetcher._read_or_fetch(path, fail, force=True)
            self.assertEqual(result.PTS.tolist(), [110])
            self.assertEqual(path.read_bytes(), before)

    def test_empty_force_refresh_retains_disk_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stats.parquet"
            pd.DataFrame({"PTS": [110]}).to_parquet(path)
            self.assertEqual(data_fetcher._read_or_fetch(path, pd.DataFrame, force=True).PTS.tolist(), [110])

    def test_failed_fetch_without_cache_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            def fail():
                raise TimeoutError("provider timeout")
            with self.assertRaises(TimeoutError):
                data_fetcher._read_or_fetch(Path(folder) / "missing.parquet", fail, force=True)

    def test_strict_scoreboard_failure_is_not_an_empty_schedule(self):
        with patch.object(data_fetcher, "_sleep"), patch.object(data_fetcher.scoreboardv3, "ScoreboardV3", side_effect=TimeoutError("provider timeout")):
            data_fetcher.get_today_scoreboard.clear()
            with self.assertRaises(TimeoutError):
                data_fetcher.get_today_scoreboard("10/24/2026", strict=True)

    def test_prediction_preload_propagates_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(data_fetcher, "run_and_cache_predictions", side_effect=TimeoutError("provider timeout")):
            with self.assertRaises(TimeoutError):
                preload_cache.refresh_predictions(Path(folder))


if __name__ == "__main__":
    unittest.main()
