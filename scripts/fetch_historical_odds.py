"""
scripts/fetch_historical_odds.py

Fetches NBA odds data and stores it to data_files/historical/odds_{season}.parquet.

Data sources (tried in order):
  1. sbrscrape — free, 7 sportsbooks, scraped from SBR Odds
  2. The Odds API — paid but reliable; requires ODDS_API_KEY env var

Usage
-----
  # Fetch today's odds (default)
  python scripts/fetch_historical_odds.py

  # Fetch a specific date
  python scripts/fetch_historical_odds.py --date 2026-03-24

  # Backfill a full season (slow — one request per day)
  python scripts/fetch_historical_odds.py --season 2025-26

  # Backfill a date range
  python scripts/fetch_historical_odds.py --start 2026-01-01 --end 2026-03-24

Output schema (one row per game per date)
-----------------------------------------
  date, home_team, away_team,
  ml_home_{book}, ml_away_{book},
  spread_{book},
  total_{book},
  over_odds_{book}, under_odds_{book},
  source   — 'sbrscrape' | 'odds_api'

Where {book} is one of: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
HIST_DIR = ROOT / "data_files" / "historical"
HIST_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

BOOKS = ["fanduel", "draftkings", "betmgm", "pointsbet", "caesars", "wynn", "bet_rivers_ny"]

# NBA season date ranges (start, end) — approximate regular-season boundaries
SEASON_DATES: dict[str, tuple[str, str]] = {
    "2025-26": ("2025-10-22", "2026-04-13"),
    "2024-25": ("2024-10-22", "2025-04-13"),
    "2023-24": ("2023-10-24", "2024-04-14"),
    "2022-23": ("2022-10-18", "2023-04-09"),
    "2021-22": ("2021-10-19", "2022-04-10"),
    "2020-21": ("2020-12-22", "2021-05-16"),
    "2019-20": ("2019-10-22", "2020-08-14"),
    "2018-19": ("2018-10-16", "2019-04-10"),
    "2017-18": ("2017-10-17", "2018-04-11"),
}

SCRAPE_DELAY = 1.5  # seconds between sbrscrape requests


# ── sbrscrape source ───────────────────────────────────────────────────────────

def _fetch_sbrscrape(date: datetime) -> list[dict]:
    """
    Fetch odds for a single date via sbrscrape.
    Returns a list of game-row dicts (flat, one per game).
    """
    try:
        from sbrscrape import Scoreboard  # optional dependency
        sb = Scoreboard(sport="NBA", date=date)
        games = getattr(sb, "games", None) or []
    except Exception as exc:
        log.debug("sbrscrape failed for %s: %s", date.date(), exc)
        return []

    rows = []
    date_str = date.strftime("%Y-%m-%d")
    for g in games:
        row: dict = {
            "date":      date_str,
            "home_team": g.get("home_team", ""),
            "away_team": g.get("away_team", ""),
            "source":    "sbrscrape",
        }
        for book in BOOKS:
            row[f"ml_home_{book}"]    = (g.get("home_ml") or {}).get(book)
            row[f"ml_away_{book}"]    = (g.get("away_ml") or {}).get(book)
            row[f"spread_{book}"]     = (g.get("home_spread") or {}).get(book)
            row[f"total_{book}"]      = (g.get("total") or {}).get(book)
            row[f"over_odds_{book}"]  = (g.get("over_odds") or {}).get(book)
            row[f"under_odds_{book}"] = (g.get("under_odds") or {}).get(book)
        rows.append(row)
    return rows


# ── The Odds API source ────────────────────────────────────────────────────────

_ODDS_API_TEAM_MAP: dict[str, str] = {
    # The Odds API uses full city+nickname; map to common short names for merging
    # This is used for fallback only; sbrscrape is preferred.
    "Atlanta Hawks": "Atlanta Hawks",
    "Boston Celtics": "Boston Celtics",
    "Brooklyn Nets": "Brooklyn Nets",
    "Charlotte Hornets": "Charlotte Hornets",
    "Chicago Bulls": "Chicago Bulls",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Dallas Mavericks": "Dallas Mavericks",
    "Denver Nuggets": "Denver Nuggets",
    "Detroit Pistons": "Detroit Pistons",
    "Golden State Warriors": "Golden State Warriors",
    "Houston Rockets": "Houston Rockets",
    "Indiana Pacers": "Indiana Pacers",
    "LA Clippers": "LA Clippers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Miami Heat": "Miami Heat",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "New York Knicks": "New York Knicks",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "Orlando Magic": "Orlando Magic",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Phoenix Suns": "Phoenix Suns",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Sacramento Kings": "Sacramento Kings",
    "San Antonio Spurs": "San Antonio Spurs",
    "Toronto Raptors": "Toronto Raptors",
    "Utah Jazz": "Utah Jazz",
    "Washington Wizards": "Washington Wizards",
}


def _fetch_odds_api(api_key: str) -> list[dict]:
    """
    Fetch TODAY's odds from The Odds API (all supported US books).
    Returns the same flat row format as _fetch_sbrscrape.
    Note: The Odds API free tier doesn't support historical dates.
    """
    if not api_key:
        return []

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey":       api_key,
        "regions":      "us",
        "markets":      "h2h,spreads,totals",
        "oddsFormat":   "american",
        "bookmakers":   ",".join([
            "draftkings", "fanduel", "betmgm", "pointsbet",
            "caesars", "wynnbet", "betrivers",
        ]),
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("The Odds API request failed: %s", exc)
        return []

    # Map Odds API book keys to our standard keys
    book_key_map = {
        "draftkings": "draftkings",
        "fanduel":    "fanduel",
        "betmgm":     "betmgm",
        "pointsbet":  "pointsbet",
        "caesars":    "caesars",
        "wynnbet":    "wynn",
        "betrivers":  "bet_rivers_ny",
    }

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    rows = []
    for game in resp.json():
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        row: dict = {
            "date":      date_str,
            "home_team": home,
            "away_team": away,
            "source":    "odds_api",
        }
        # Initialize all book columns to None
        for book in BOOKS:
            for prefix in ("ml_home", "ml_away", "spread", "total", "over_odds", "under_odds"):
                row[f"{prefix}_{book}"] = None

        for bm in game.get("bookmakers", []):
            bm_key = book_key_map.get(bm["key"])
            if not bm_key:
                continue
            for market in bm.get("markets", []):
                mkey = market["key"]
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}
                if mkey == "h2h":
                    row[f"ml_home_{bm_key}"] = (outcomes.get(home) or {}).get("price")
                    row[f"ml_away_{bm_key}"] = (outcomes.get(away) or {}).get("price")
                elif mkey == "spreads":
                    ho = outcomes.get(home, {})
                    row[f"spread_{bm_key}"] = ho.get("point")
                elif mkey == "totals":
                    over_o  = outcomes.get("Over", {})
                    under_o = outcomes.get("Under", {})
                    row[f"total_{bm_key}"]      = over_o.get("point")
                    row[f"over_odds_{bm_key}"]  = over_o.get("price")
                    row[f"under_odds_{bm_key}"] = under_o.get("price")
        rows.append(row)
    return rows


# ── Parquet storage ────────────────────────────────────────────────────────────

def _season_for_date(date: datetime) -> str:
    """Return the NBA season string for a given date (e.g. '2025-26')."""
    year  = date.year
    month = date.month
    # NBA season starts in October of year Y and ends in April of Y+1
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def _parquet_path(season: str) -> Path:
    ss = season.replace("-", "_")
    return HIST_DIR / f"odds_{ss}.parquet"


def _load_existing(season: str) -> pd.DataFrame:
    path = _parquet_path(season)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame()


def _save(df: pd.DataFrame, season: str) -> None:
    path = _parquet_path(season)
    df.to_parquet(path, index=False)
    log.info("Saved %d rows → %s", len(df), path.name)


def _upsert_rows(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    """Merge new_rows into existing DataFrame, deduplicating by (date, home_team)."""
    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        return new_df
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "home_team"], keep="last")
    return combined.sort_values(["date", "home_team"]).reset_index(drop=True)


# ── Main fetch logic ───────────────────────────────────────────────────────────

def fetch_date(date: datetime, api_key: str = "") -> list[dict]:
    """Fetch odds for a single date.  Tries sbrscrape first, then Odds API."""
    log.info("Fetching odds for %s ...", date.strftime("%Y-%m-%d"))

    rows = _fetch_sbrscrape(date)
    if rows:
        log.info("  sbrscrape: %d games", len(rows))
        return rows

    log.info("  sbrscrape returned 0 games — trying The Odds API fallback")
    rows = _fetch_odds_api(api_key)
    if rows:
        log.info("  Odds API: %d games", len(rows))
    else:
        log.info("  No games found from any source for %s", date.strftime("%Y-%m-%d"))
    return rows


def fetch_today(api_key: str = "") -> None:
    """Fetch today's odds and persist."""
    today = datetime.utcnow()
    rows  = fetch_date(today, api_key)
    if not rows:
        return
    season   = _season_for_date(today)
    existing = _load_existing(season)
    updated  = _upsert_rows(existing, rows)
    _save(updated, season)


def fetch_season(season: str, api_key: str = "") -> None:
    """Backfill odds for an entire season.  Skips dates already stored."""
    if season not in SEASON_DATES:
        log.error("Unknown season: %s. Valid: %s", season, ", ".join(SEASON_DATES))
        return

    start_str, end_str = SEASON_DATES[season]
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end   = min(
        datetime.strptime(end_str, "%Y-%m-%d"),
        datetime.utcnow(),
    )

    existing = _load_existing(season)
    already_fetched: set[str] = set()
    if not existing.empty and "date" in existing.columns:
        already_fetched = set(existing["date"].tolist())

    date = start
    all_rows: list[dict] = [] if existing.empty else existing.to_dict("records")

    while date <= end:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in already_fetched:
            log.debug("  Skipping %s (already stored)", date_str)
            date += timedelta(days=1)
            continue

        rows = fetch_date(date, api_key)
        all_rows.extend(rows)
        time.sleep(SCRAPE_DELAY)
        date += timedelta(days=1)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset=["date", "home_team"], keep="last")
        df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
        _save(df, season)


def fetch_range(start: datetime, end: datetime, api_key: str = "") -> None:
    """Backfill a custom date range."""
    date = start
    while date <= end:
        rows = fetch_date(date, api_key)
        if rows:
            season   = _season_for_date(date)
            existing = _load_existing(season)
            updated  = _upsert_rows(existing, rows)
            _save(updated, season)
        time.sleep(SCRAPE_DELAY)
        date += timedelta(days=1)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NBA odds data to parquet")
    parser.add_argument("--date",   help="Single date YYYY-MM-DD (default: today)")
    parser.add_argument("--season", help="Backfill full season, e.g. 2025-26")
    parser.add_argument("--start",  help="Backfill start YYYY-MM-DD (use with --end)")
    parser.add_argument("--end",    help="Backfill end   YYYY-MM-DD (use with --start)")
    args = parser.parse_args()

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        log.info("ODDS_API_KEY not set — will use sbrscrape only (no Odds API fallback).")

    if args.season:
        fetch_season(args.season, api_key)
    elif args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d")
        e = datetime.strptime(args.end,   "%Y-%m-%d")
        fetch_range(s, e, api_key)
    elif args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d")
        rows = fetch_date(d, api_key)
        if rows:
            season   = _season_for_date(d)
            existing = _load_existing(season)
            updated  = _upsert_rows(existing, rows)
            _save(updated, season)
    else:
        fetch_today(api_key)


if __name__ == "__main__":
    main()
