"""Validate ESPN box-score conversion and atomic fallback refreshes."""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from utils import espn_game_log


class ESPNGameLogTests(unittest.TestCase):
    def test_box_score_maps_to_model_game_log(self):
        event = {"id": "1", "date": "2026-10-28T23:00Z", "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"abbreviation": "BOS"}, "score": "107"},
            {"homeAway": "away", "team": {"abbreviation": "WSH"}, "score": "100"},
        ]}]}
        def box(side, abbr, fg, threes, ft):
            stats = {"fieldGoalsMade-fieldGoalsAttempted": fg,
                "threePointFieldGoalsMade-threePointFieldGoalsAttempted": threes,
                "freeThrowsMade-freeThrowsAttempted": ft, "offensiveRebounds": "10",
                "defensiveRebounds": "30", "totalRebounds": "40", "assists": "20",
                "steals": "5", "blocks": "3", "totalTurnovers": "12", "fouls": "18"}
            return {"homeAway": side, "team": {"abbreviation": abbr},
                    "statistics": [{"name": k, "displayValue": v} for k, v in stats.items()]}
        response = {"boxscore": {"teams": [box("home", "BOS", "40-80", "10-30", "17-20"),
                                                box("away", "WSH", "40-90", "10-35", "10-15")]}}
        teams = {"BOS": {"id": 1610612738}, "WAS": {"id": 1610612764}}
        with patch.object(espn_game_log, "_json", return_value=response):
            home, away = espn_game_log._game_rows(event, teams)
        self.assertEqual((home["GAME_ID"], home["WL"], home["PLUS_MINUS"]), ("espn:1", "W", 7))
        self.assertEqual((away["TEAM_ID"], away["MATCHUP"], away["WL"]), (1610612764, "WAS @ BOS", "L"))
        self.assertEqual(home["FG3_PCT"], 1 / 3)

    def test_failed_game_refresh_never_overwrites_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "espn.parquet"
            pd.DataFrame({"GAME_ID": ["espn:1", "espn:1"], "TEAM_ID": [1, 2]}).to_parquet(path)
            before = path.read_bytes()
            with patch.object(espn_game_log, "_completed_games", return_value=[{"id": "1"}, {"id": "2"}]), \
                 patch.object(espn_game_log, "_game_rows", side_effect=ValueError("bad box")):
                with self.assertRaisesRegex(ValueError, "bad box"):
                    espn_game_log.get_espn_game_log("2026-27", path, today=date(2026, 10, 1))
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
