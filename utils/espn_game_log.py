"""Build a current-season team game log from completed ESPN box scores."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from nba_api.stats.static import teams as nba_teams
from utils.hoopr_fetcher import normalize_abbr_to_nba

_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_ET = ZoneInfo("America/New_York")


def _json(endpoint: str, params: dict) -> dict:
    response = requests.get(f"{_BASE}/{endpoint}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _abbr(value: str) -> str:
    value = normalize_abbr_to_nba(value)
    return {"WSH": "WAS", "UTAH": "UTA"}.get(value, value)


def _completed_games(day: date) -> list[dict]:
    payload = _json("scoreboard", {"dates": day.strftime("%Y%m%d"), "limit": 100})
    events = payload["events"]
    if not isinstance(events, list):
        raise ValueError("Invalid ESPN scoreboard events")
    result = []
    for event in events:
        event_date = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(_ET).date()
        if event_date != day:
            raise ValueError("ESPN scoreboard returned an event for a different date")
        if event["season"]["type"] == 2 and event["status"]["type"]["completed"]:
            result.append(event)
    return result


def _game_rows(event: dict, teams: dict) -> list[dict]:
    game_id = str(event["id"])
    competitors = {c["homeAway"]: c for c in event["competitions"][0]["competitors"]}
    if set(competitors) != {"home", "away"}:
        raise ValueError(f"Invalid teams for ESPN game {game_id}")
    box = _json("summary", {"event": game_id})["boxscore"]["teams"]
    boxes = {item["homeAway"]: item for item in box}
    if set(boxes) != {"home", "away"}:
        raise ValueError(f"Incomplete box score for ESPN game {game_id}")
    game_date = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(_ET).date().isoformat()
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        competitor, opponent = competitors[side], competitors[other]
        abbr, opponent_abbr = _abbr(competitor["team"]["abbreviation"]), _abbr(opponent["team"]["abbreviation"])
        if _abbr(boxes[side]["team"]["abbreviation"]) != abbr:
            raise ValueError(f"Team mismatch in ESPN game {game_id}")
        stats = {s["name"]: s["displayValue"] for s in boxes[side]["statistics"]}
        fgm, fga = map(int, stats["fieldGoalsMade-fieldGoalsAttempted"].split("-"))
        fg3m, fg3a = map(int, stats["threePointFieldGoalsMade-threePointFieldGoalsAttempted"].split("-"))
        ftm, fta = map(int, stats["freeThrowsMade-freeThrowsAttempted"].split("-"))
        points, opp_points = int(competitor["score"]), int(opponent["score"])
        if points == opp_points or points != 2 * fgm + fg3m + ftm:
            raise ValueError(f"Invalid score in ESPN game {game_id}")
        rows.append({
            "GAME_ID": f"espn:{game_id}", "GAME_DATE": game_date,
            "TEAM_ID": teams[abbr]["id"], "TEAM_ABBREVIATION": abbr,
            "MATCHUP": f"{abbr} {'vs.' if side == 'home' else '@'} {opponent_abbr}",
            "WL": "W" if points > opp_points else "L", "PTS": points,
            "FGM": fgm, "FGA": fga, "FG_PCT": fgm / fga if fga else 0.0,
            "FG3M": fg3m, "FG3A": fg3a, "FG3_PCT": fg3m / fg3a if fg3a else 0.0,
            "FTM": ftm, "FTA": fta, "FT_PCT": ftm / fta if fta else 0.0,
            "OREB": int(stats["offensiveRebounds"]), "DREB": int(stats["defensiveRebounds"]),
            "REB": int(stats["totalRebounds"]), "AST": int(stats["assists"]),
            "STL": int(stats["steals"]), "BLK": int(stats["blocks"]),
            "TOV": int(stats["totalTurnovers"]), "PF": int(stats["fouls"]),
            "PLUS_MINUS": points - opp_points,
        })
    return rows


def get_espn_game_log(season: str, cache_path: Path, *, today: date | None = None) -> pd.DataFrame:
    """Refresh a separately labelled ESPN cache, failing on incomplete data."""
    today = today or datetime.now(_ET).date()
    start_year = int(season.split("-")[0])
    if season != f"{start_year}-{str(start_year + 1)[-2:]}" or today < date(start_year, 10, 1):
        raise ValueError("Invalid current season")
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cached.empty:
        counts = cached.groupby("GAME_ID").size()
        if not counts.eq(2).all():
            raise ValueError("Incomplete ESPN game-log cache")
    known = set(cached["GAME_ID"].astype(str)) if not cached.empty else set()
    dates = [date(start_year, 10, 1) + timedelta(days=n) for n in range((today - date(start_year, 10, 1)).days + 1)]
    teams = {team["abbreviation"]: team for team in nba_teams.get_teams()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        daily = list(pool.map(_completed_games, dates))
        events = [event for games in daily for event in games]
        event_ids = [f"espn:{event['id']}" for event in events]
        if len(set(event_ids)) != len(event_ids) or not known.issubset(set(event_ids)):
            raise ValueError("ESPN schedule does not match cached completed games")
        missing = [event for event in events if f"espn:{event['id']}" not in known]
        new_rows = list(pool.map(lambda event: _game_rows(event, teams), missing))
    fresh = pd.DataFrame([row for rows in new_rows for row in rows])
    result = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    if result.empty:
        raise ValueError("ESPN has no completed regular-season games")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".parquet.tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(cache_path)
    return result
