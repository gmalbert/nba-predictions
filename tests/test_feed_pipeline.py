"""Regression checks for failed provider refreshes and season rollover."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from scripts import preload_cache, scrape_external
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

    def test_espn_scoreboard_fallback_maps_teams_and_labels_game_id(self):
        payload = {"events": [{"id": "401909881", "date": "2026-10-28T23:00Z",
            "status": {"type": {"state": "pre", "detail": "7:00 pm ET"}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"abbreviation": "BOS"}},
                {"homeAway": "away", "team": {"abbreviation": "WSH"}},
            ]}]}]}
        response = Mock()
        response.json.return_value = payload
        with patch.object(data_fetcher, "_today_et", return_value=datetime(2026, 10, 28)), \
             patch.object(data_fetcher, "_sleep"), \
             patch.object(data_fetcher.scoreboardv3, "ScoreboardV3", side_effect=TimeoutError()), \
             patch.object(data_fetcher, "_nba_live_scoreboard", side_effect=TimeoutError()), \
             patch.object(data_fetcher.requests, "get", return_value=response):
            data_fetcher.get_today_scoreboard.clear()
            games, _ = data_fetcher.get_today_scoreboard("10/28/2026", strict=True)
        self.assertEqual(games.iloc[0]["GAME_ID"], "espn:401909881")
        self.assertEqual(games.iloc[0]["HOME_TEAM_ID"], 1610612738)
        self.assertEqual(games.iloc[0]["VISITOR_TEAM_ABBREVIATION"], "WAS")

    def test_espn_scoreboard_rejects_stale_data(self):
        response = Mock()
        response.json.return_value = {"events": [{"date": "2026-10-27T23:00Z"}]}
        with patch.object(data_fetcher.requests, "get", return_value=response):
            with self.assertRaises(ValueError):
                data_fetcher._espn_scoreboard("10/28/2026")

    def test_today_scoreboard_propagates_fallback_failure(self):
        with patch.object(data_fetcher, "_today_et", return_value=datetime(2026, 10, 28)), \
             patch.object(data_fetcher, "_sleep"), \
             patch.object(data_fetcher.scoreboardv3, "ScoreboardV3", side_effect=TimeoutError()), \
             patch.object(data_fetcher, "_nba_live_scoreboard", side_effect=TimeoutError()), \
             patch.object(data_fetcher, "_espn_scoreboard", side_effect=ValueError("stale")):
            data_fetcher.get_today_scoreboard.clear()
            with self.assertRaisesRegex(ValueError, "stale"):
                data_fetcher.get_today_scoreboard("10/28/2026", strict=True)

    def test_live_scoreboard_fallback_preserves_nba_ids(self):
        response = Mock()
        response.json.return_value = {"scoreboard": {"gameDate": "2026-10-28", "games": [{
            "gameId": "0022600001", "gameStatus": 1, "gameStatusText": "7:00 pm ET",
            "homeTeam": {"teamId": 1610612738, "teamTricode": "BOS", "teamCity": "Boston", "teamName": "Celtics"},
            "awayTeam": {"teamId": 1610612747, "teamTricode": "LAL", "teamCity": "Los Angeles", "teamName": "Lakers"},
        }]}}
        with patch.object(data_fetcher, "_today_et", return_value=datetime(2026, 10, 28)), \
             patch.object(data_fetcher, "_sleep"), \
             patch.object(data_fetcher.scoreboardv3, "ScoreboardV3", side_effect=TimeoutError()), \
             patch.object(data_fetcher.requests, "get", return_value=response) as request, \
             patch.object(data_fetcher, "_espn_scoreboard") as espn:
            data_fetcher.get_today_scoreboard.clear()
            games, _ = data_fetcher.get_today_scoreboard("10/28/2026", strict=True)
        self.assertEqual(games.iloc[0]["GAME_ID"], "0022600001")
        self.assertEqual(games.iloc[0]["HOME_TEAM_ID"], 1610612738)
        self.assertEqual(games.iloc[0]["VISITOR_TEAM_NAME"], "Los Angeles Lakers")
        self.assertEqual(request.call_args.kwargs["headers"]["Referer"], "https://www.nba.com/")
        espn.assert_not_called()

    def test_live_scoreboard_rejects_stale_or_malformed_data(self):
        response = Mock()
        for payload in [
            {"gameDate": "2026-10-27", "games": []},
            {"gameDate": "2026-10-28", "games": {}},
            {"gameDate": "2026-10-28", "games": [{"gameId": "bad", "homeTeam": {}, "awayTeam": {}}]},
        ]:
            response.json.return_value = {"scoreboard": payload}
            with self.subTest(payload=payload), patch.object(data_fetcher.requests, "get", return_value=response):
                with self.assertRaises((ValueError, KeyError)):
                    data_fetcher._nba_live_scoreboard("10/28/2026")

    def test_live_scoreboard_accepts_genuine_empty_day(self):
        response = Mock()
        response.json.return_value = {"scoreboard": {"gameDate": "2026-10-28", "games": []}}
        with patch.object(data_fetcher.requests, "get", return_value=response):
            games, _ = data_fetcher._nba_live_scoreboard("10/28/2026")
        self.assertTrue(games.empty)

    def test_referee_lookup_uses_nba_ids_and_skips_espn_ids(self):
        games = pd.DataFrame([
            {"GAME_ID": "0022600001", "HOME_TEAM_ABBREVIATION": "BOS", "VISITOR_TEAM_ABBREVIATION": "LAL",
             "HOME_TEAM_NAME": "Boston Celtics", "VISITOR_TEAM_NAME": "Los Angeles Lakers"},
            {"GAME_ID": "espn:401909881", "HOME_TEAM_ABBREVIATION": "BOS", "VISITOR_TEAM_ABBREVIATION": "WAS",
             "HOME_TEAM_NAME": "Boston Celtics", "VISITOR_TEAM_NAME": "Washington Wizards"},
        ])
        with patch.object(data_fetcher, "get_today_scoreboard", return_value=(games, pd.DataFrame())):
            result = scrape_external._referee_nba_games("10/28/2026")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["nba_game_id"], "0022600001")
        self.assertEqual(result[0]["home_display"], "boston celtics")

    def test_cached_prediction_ids_upgrade_without_changing_probabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.parquet"
            original = pd.DataFrame({"game_id": ["espn:1"], "home_team_id": [1610612738],
                                     "away_team_id": [1610612747], "home_win_prob": [0.72]})
            original.to_parquet(path)
            live = pd.DataFrame({"GAME_ID": ["0022600001"], "HOME_TEAM_ID": [1610612738],
                                 "VISITOR_TEAM_ID": [1610612747]})
            with patch.object(data_fetcher, "_nba_live_scoreboard", return_value=(live, pd.DataFrame())):
                self.assertTrue(preload_cache._upgrade_cached_prediction_ids(path))
            result = pd.read_parquet(path)
            self.assertEqual(result.game_id.tolist(), ["0022600001"])
            self.assertEqual(result.home_win_prob.tolist(), [0.72])

    def test_cached_prediction_id_upgrade_is_all_or_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.parquet"
            pd.DataFrame({"game_id": ["espn:1", "espn:2"], "home_team_id": [1, 3],
                          "away_team_id": [2, 4]}).to_parquet(path)
            before = path.read_bytes()
            live = pd.DataFrame({"GAME_ID": ["0022600001"], "HOME_TEAM_ID": [1], "VISITOR_TEAM_ID": [2]})
            with patch.object(data_fetcher, "_nba_live_scoreboard", return_value=(live, pd.DataFrame())):
                self.assertFalse(preload_cache._upgrade_cached_prediction_ids(path))
            self.assertEqual(path.read_bytes(), before)

    def test_prediction_preload_propagates_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(data_fetcher, "run_and_cache_predictions", side_effect=TimeoutError("provider timeout")):
            with self.assertRaises(TimeoutError):
                preload_cache.refresh_predictions(Path(folder))


if __name__ == "__main__":
    unittest.main()
