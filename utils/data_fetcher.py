"""
NBA data fetching utilities using nba_api.

Handles rate limiting, disk-based caching (parquet), and Streamlit cache
decorators so that API calls are minimized in both local dev and Streamlit Cloud.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from datetime import datetime
import streamlit as st

# nba_api imports
from nba_api.stats.endpoints import (
    scoreboardv3,
    leaguegamelog,
    teamgamelog,
    playergamelog,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
    leaguedashteamstats,
    leaguedashplayerstats,
    leaguestandingsv3,
    teamestimatedmetrics,
    commonteamroster,
    teaminfocommon,
)
from nba_api.stats.static import teams as nba_teams, players as nba_players

# ── Directory setup ────────────────────────────────────────────────────────────
DATA_DIR = Path("data_files")
HIST_DIR = DATA_DIR / "historical"
DATA_DIR.mkdir(exist_ok=True)
HIST_DIR.mkdir(exist_ok=True)

RATE_LIMIT_DELAY = 0.7  # seconds between nba_api calls

HISTORICAL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
CURRENT_SEASON = "2025-26"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sleep():
    """Respect NBA.com rate limits."""
    time.sleep(RATE_LIMIT_DELAY)


def _read_or_fetch(path: Path, fetch_fn) -> pd.DataFrame:
    """Return cached parquet if it exists; otherwise call fetch_fn, persist, and return."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass  # Re-fetch if cached file is corrupt
    df = fetch_fn()
    if df is not None and not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df if df is not None else pd.DataFrame()


# ── Scoreboard ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_today_scoreboard(game_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch scoreboard for today or a given date (MM/DD/YYYY).
    Returns (game_header_df, line_score_df).  Cached for 5 minutes.

    Uses ScoreboardV3 which is fully supported for the 2025-26 season.
    game_header columns: GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID,
        HOME_TEAM_NAME, VISITOR_TEAM_NAME, HOME_TEAM_ABBREVIATION,
        VISITOR_TEAM_ABBREVIATION, GAME_STATUS_TEXT, GAME_STATUS_ID
    """
    try:
        if game_date is None:
            game_date = datetime.today().strftime("%m/%d/%Y")
        _sleep()
        sb = scoreboardv3.ScoreboardV3(game_date=game_date, league_id="00")

        games_df = sb.data_sets[1].get_data_frame()  # game header (1 row/game)
        teams_df = sb.data_sets[2].get_data_frame()  # teams     (2 rows/game)

        if games_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Build tricode lookups from the teams dataset
        tricode_to_id   = dict(zip(teams_df["teamTricode"], teams_df["teamId"]))
        tricode_to_name = {
            r["teamTricode"]: f"{r['teamCity']} {r['teamName']}"
            for _, r in teams_df.iterrows()
        }

        # gameCode format: "YYYYMMDD/AWYHME" e.g. "20260323/LALDET"
        # first 3 chars of the team part = away tricode, last 3 = home tricode
        rows = []
        for _, g in games_df.iterrows():
            code  = g.get("gameCode", "")
            parts = code.split("/")
            if len(parts) == 2 and len(parts[1]) == 6:
                away_abbr = parts[1][:3]
                home_abbr = parts[1][3:]
            else:
                continue
            rows.append({
                "GAME_ID":                    g["gameId"],
                "HOME_TEAM_ID":               tricode_to_id.get(home_abbr),
                "VISITOR_TEAM_ID":            tricode_to_id.get(away_abbr),
                "HOME_TEAM_ABBREVIATION":     home_abbr,
                "VISITOR_TEAM_ABBREVIATION":  away_abbr,
                "HOME_TEAM_NAME":             tricode_to_name.get(home_abbr, home_abbr),
                "VISITOR_TEAM_NAME":          tricode_to_name.get(away_abbr, away_abbr),
                "GAME_STATUS_TEXT":           g.get("gameStatusText", ""),
                "GAME_STATUS_ID":             g.get("gameStatus", 1),
            })

        return pd.DataFrame(rows), pd.DataFrame()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ── League Game Logs ───────────────────────────────────────────────────────────

def get_league_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    All team-level game rows for a full season.  Cached to disk (parquet).
    Uses the LeagueGameLog endpoint (player_or_team='T').
    """
    tag = season_type.replace(" ", "_")
    path = HIST_DIR / f"league_gamelog_{season.replace('-', '_')}_{tag}.parquet"

    def fetch():
        _sleep()
        raw = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="T",
        )
        return raw.get_data_frames()[0]

    return _read_or_fetch(path, fetch)


@st.cache_data(ttl=3600)
def get_league_game_log_cached(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    return get_league_game_log(season, season_type)


# ── Team Game Logs ─────────────────────────────────────────────────────────────

def get_team_game_log(team_id: int, season: str) -> pd.DataFrame:
    """Single-team game log. Cached to disk."""
    path = HIST_DIR / f"teamlog_{team_id}_{season.replace('-', '_')}.parquet"

    def fetch():
        _sleep()
        raw = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        return raw.get_data_frames()[0]

    return _read_or_fetch(path, fetch)


@st.cache_data(ttl=3600)
def get_team_game_log_cached(team_id: int, season: str) -> pd.DataFrame:
    return get_team_game_log(team_id, season)


# ── Player Game Logs ───────────────────────────────────────────────────────────

def get_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    """Single-player game log. Cached to disk."""
    path = HIST_DIR / f"playerlog_{player_id}_{season.replace('-', '_')}.parquet"

    def fetch():
        _sleep()
        raw = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = raw.get_data_frames()[0]
        df["PLAYER_ID"] = player_id
        return df

    return _read_or_fetch(path, fetch)


@st.cache_data(ttl=3600)
def get_player_game_log_cached(player_id: int, season: str) -> pd.DataFrame:
    return get_player_game_log(player_id, season)


# ── Box Scores ─────────────────────────────────────────────────────────────────

def get_box_score_traditional(game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player_stats, team_stats). Cached to disk."""
    pp = HIST_DIR / f"bst_player_{game_id}.parquet"
    tp = HIST_DIR / f"bst_team_{game_id}.parquet"
    if pp.exists() and tp.exists():
        try:
            return pd.read_parquet(pp), pd.read_parquet(tp)
        except Exception:
            pass
    try:
        _sleep()
        bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        pdf = bs.player_stats.get_data_frame()
        tdf = bs.team_stats.get_data_frame()
        if not pdf.empty:
            pdf.to_parquet(pp, index=False)
        if not tdf.empty:
            tdf.to_parquet(tp, index=False)
        return pdf, tdf
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def get_box_score_advanced(game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player_advanced, team_advanced). Cached to disk."""
    pp = HIST_DIR / f"bsa_player_{game_id}.parquet"
    tp = HIST_DIR / f"bsa_team_{game_id}.parquet"
    if pp.exists() and tp.exists():
        try:
            return pd.read_parquet(pp), pd.read_parquet(tp)
        except Exception:
            pass
    try:
        _sleep()
        bs = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        pdf = bs.player_stats.get_data_frame()
        tdf = bs.team_stats.get_data_frame()
        if not pdf.empty:
            pdf.to_parquet(pp, index=False)
        if not tdf.empty:
            tdf.to_parquet(tp, index=False)
        return pdf, tdf
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ── League-Wide Stats ──────────────────────────────────────────────────────────

def get_league_team_stats(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game team stats for all 30 teams. Cached to disk."""
    tag = season_type.replace(" ", "_")
    path = HIST_DIR / f"league_teamstats_{season.replace('-', '_')}_{tag}.parquet"

    def fetch():
        _sleep()
        raw = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
        )
        return raw.get_data_frames()[0]

    return _read_or_fetch(path, fetch)


@st.cache_data(ttl=3600)
def get_league_team_stats_cached(season: str) -> pd.DataFrame:
    return get_league_team_stats(season)


def get_league_player_stats(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game player stats league-wide. Cached to disk."""
    tag = season_type.replace(" ", "_")
    path = HIST_DIR / f"league_playerstats_{season.replace('-', '_')}_{tag}.parquet"

    def fetch():
        _sleep()
        raw = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
        )
        return raw.get_data_frames()[0]

    return _read_or_fetch(path, fetch)


@st.cache_data(ttl=3600)
def get_league_player_stats_cached(season: str) -> pd.DataFrame:
    return get_league_player_stats(season)


def get_team_estimated_metrics(season: str) -> pd.DataFrame:
    """
    Advanced estimated team metrics: ORtg, DRtg, pace, net rating.
    Cached to disk.
    """
    path = HIST_DIR / f"team_est_metrics_{season.replace('-', '_')}.parquet"

    def fetch():
        _sleep()
        raw = teamestimatedmetrics.TeamEstimatedMetrics(season=season)
        return raw.get_data_frames()[0]

    return _read_or_fetch(path, fetch)


@st.cache_data(ttl=3600)
def get_team_estimated_metrics_cached(season: str) -> pd.DataFrame:
    return get_team_estimated_metrics(season)


@st.cache_data(ttl=3600)
def get_standings(season: str) -> pd.DataFrame:
    """Current league standings from LeagueStandingsV3. Disk-cached with same-day freshness."""
    path = HIST_DIR / f"standings_{season.replace('-', '_')}.parquet"
    today_str = datetime.today().strftime("%Y-%m-%d")
    if path.exists():
        try:
            cached = pd.read_parquet(path)
            if (
                "FETCH_DATE" in cached.columns
                and str(cached["FETCH_DATE"].iloc[0])[:10] == today_str
            ):
                return cached.drop(columns=["FETCH_DATE"])
        except Exception:
            pass
    try:
        _sleep()
        raw = leaguestandingsv3.LeagueStandingsV3(season=season)
        df = raw.standings.get_data_frame()
        if not df.empty:
            df["FETCH_DATE"] = today_str
            df.to_parquet(path, index=False)
            return df.drop(columns=["FETCH_DATE"])
        return df
    except Exception:
        # API unavailable — return stale disk data if present
        if path.exists():
            try:
                return pd.read_parquet(path).drop(columns=["FETCH_DATE"], errors="ignore")
            except Exception:
                pass
        return pd.DataFrame()


# ── Team & Player Metadata ─────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_all_teams() -> pd.DataFrame:
    """All 30 NBA teams (static, from nba_api.stats.static)."""
    return pd.DataFrame(nba_teams.get_teams())


@st.cache_data(ttl=86400)
def get_all_active_players() -> pd.DataFrame:
    """All currently active NBA players (static)."""
    return pd.DataFrame(nba_players.get_active_players())


@st.cache_data(ttl=86400)
def get_team_roster(team_id: int, season: str) -> pd.DataFrame:
    """Current roster for a team."""
    try:
        _sleep()
        raw = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        return raw.common_team_roster.get_data_frame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_team_info(team_id: int) -> dict:
    """Team info (city, conference, division, arena)."""
    try:
        _sleep()
        raw = teaminfocommon.TeamInfoCommon(team_id=team_id)
        df = raw.team_info_common.get_data_frame()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception:
        return {}


# ── ESPN Injury Report ─────────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def get_injury_report() -> pd.DataFrame:
    """
    Current NBA injury data from ESPN's public JSON API.
    Returns columns: team, player_name, position, status, description.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        records = []
        for team_entry in data.get("injuries", []):
            team_name = team_entry.get("displayName", "")
            for inj in team_entry.get("injuries", []):
                athlete = inj.get("athlete", {})
                records.append({
                    "team": team_name,
                    "player_name": athlete.get("displayName", ""),
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "status": inj.get("status", "Unknown"),
                    "description": inj.get("shortComment", ""),
                })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame(columns=["team", "player_name", "position", "status", "description"])


# ── Odds Data (The Odds API) ───────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def get_nba_odds(api_key: str) -> pd.DataFrame:
    """
    NBA moneyline, spread, and totals from The Odds API using DraftKings.
    Requires a free API key from https://the-odds-api.com
    Returns one row per outcome per market.
    """
    if not api_key:
        return pd.DataFrame()
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "bookmakers": "draftkings",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        records = []
        for game in resp.json():
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            commence = game.get("commence_time", "")
            for bm in game.get("bookmakers", []):
                for market in bm.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        records.append({
                            "game_id": game["id"],
                            "home_team": home,
                            "away_team": away,
                            "commence_time": commence,
                            "market": market["key"],
                            "name": outcome["name"],
                            "price": outcome["price"],
                            "point": outcome.get("point"),
                        })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def american_odds_to_prob(odds: float) -> float:
    """Convert American odds to raw implied probability."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal (European) format."""
    if american_odds >= 100:
        return american_odds / 100.0 + 1.0
    return 100.0 / abs(american_odds) + 1.0


def expected_value(model_prob: float, american_odds: float) -> float:
    """Expected value per $100 wagered. Positive = profitable bet."""
    if american_odds > 0:
        payout = float(american_odds)
    else:
        payout = 100.0 / abs(american_odds) * 100.0
    return round(model_prob * payout - (1.0 - model_prob) * 100.0, 2)


def kelly_criterion(model_prob: float, american_odds: float) -> float:
    """Fraction of bankroll to wager (full Kelly). Returns 0 if bet is −EV."""
    decimal = american_to_decimal(american_odds)
    fraction = (decimal * model_prob - (1.0 - model_prob)) / decimal
    return round(max(fraction, 0.0), 4)


def get_implied_probs(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    From get_nba_odds output extract vig-adjusted moneyline probabilities.
    Returns: game_id, home_team, away_team, home_prob, away_prob.
    """
    if odds_df.empty:
        return pd.DataFrame()
    ml = odds_df[odds_df["market"] == "h2h"].copy()
    if ml.empty:
        return pd.DataFrame()
    ml["raw_prob"] = ml["price"].apply(american_odds_to_prob)
    ml["is_home"] = ml["name"] == ml["home_team"]
    home = ml[ml["is_home"]][["game_id", "home_team", "away_team", "raw_prob"]].rename(
        columns={"raw_prob": "home_raw"}
    )
    away = ml[~ml["is_home"]][["game_id", "raw_prob"]].rename(
        columns={"raw_prob": "away_raw"}
    )
    merged = home.merge(away, on="game_id", how="inner")
    merged["vig"] = merged["home_raw"] + merged["away_raw"]
    merged["home_prob"] = merged["home_raw"] / merged["vig"]
    merged["away_prob"] = merged["away_raw"] / merged["vig"]
    return merged[["game_id", "home_team", "away_team", "home_prob", "away_prob"]]


# ── Multi-Book Odds (sbrscrape) ────────────────────────────────────────────────

SBRSCRAPE_BOOKS = [
    "fanduel", "draftkings", "betmgm", "pointsbet",
    "caesars", "wynn", "bet_rivers_ny",
]

# Display label per book key
BOOK_LABELS = {
    "fanduel":       "FanDuel",
    "draftkings":    "DraftKings",
    "betmgm":        "BetMGM",
    "pointsbet":     "PointsBet",
    "caesars":       "Caesars",
    "wynn":          "Wynn",
    "bet_rivers_ny": "BetRivers",
}


@st.cache_data(ttl=300)
def get_multi_book_odds() -> list[dict]:
    """
    Fetch live NBA odds from all major sportsbooks via sbrscrape.

    Returns a list of game dicts, each with keys:
        home_team, away_team,
        home_ml, away_ml,          — {book: american_odds}
        home_spread, away_spread,  — {book: spread}
        home_spread_odds, away_spread_odds,
        total, over_odds, under_odds  — {book: value}

    Falls back to empty list on any error (e.g. sbrscrape unavailable).
    """
    try:
        from sbrscrape import Scoreboard  # optional dependency
        sb = Scoreboard(sport="NBA")
        games = getattr(sb, "games", None) or []
        return list(games)
    except Exception:
        return []


def get_best_lines(games: list[dict]) -> list[dict]:
    """
    Given output of get_multi_book_odds(), compute best available line per
    market for each game.

    Returns a list of dicts with keys:
        home_team, away_team,
        best_home_ml, best_home_ml_book,
        best_away_ml, best_away_ml_book,
        best_home_spread, best_home_spread_book,
        consensus_total, total_books,
    """
    result = []
    for g in games:
        home_mls   = {k: v for k, v in (g.get("home_ml") or {}).items() if v is not None}
        away_mls   = {k: v for k, v in (g.get("away_ml") or {}).items() if v is not None}
        spreads    = {k: v for k, v in (g.get("home_spread") or {}).items() if v is not None}
        totals     = {k: v for k, v in (g.get("total") or {}).items() if v is not None}

        if not home_mls:
            continue

        # Best home ML = highest (most positive / least negative) for home bettor
        best_home_ml_book = max(home_mls, key=home_mls.get)
        best_away_ml_book = max(away_mls, key=away_mls.get) if away_mls else None
        best_spread_book  = min(spreads, key=lambda k: abs(spreads[k])) if spreads else None

        result.append({
            "home_team":             g.get("home_team", ""),
            "away_team":             g.get("away_team", ""),
            "best_home_ml":          home_mls[best_home_ml_book],
            "best_home_ml_book":     BOOK_LABELS.get(best_home_ml_book, best_home_ml_book),
            "best_away_ml":          away_mls[best_away_ml_book] if best_away_ml_book else None,
            "best_away_ml_book":     BOOK_LABELS.get(best_away_ml_book, best_away_ml_book) if best_away_ml_book else None,
            "best_home_spread":      spreads[best_spread_book] if best_spread_book else None,
            "best_home_spread_book": BOOK_LABELS.get(best_spread_book, best_spread_book) if best_spread_book else None,
            "consensus_total":       float(np.mean(list(totals.values()))) if totals else None,
            "total_books":           len(totals),
            "all_home_ml":           {BOOK_LABELS.get(k, k): v for k, v in home_mls.items()},
            "all_away_ml":           {BOOK_LABELS.get(k, k): v for k, v in away_mls.items()},
            "all_spreads":           {BOOK_LABELS.get(k, k): v for k, v in spreads.items()},
            "all_totals":            {BOOK_LABELS.get(k, k): v for k, v in totals.items()},
        })
    return result


# ── External Scraped Data (nbastuffer / databallr) ─────────────────────────────
# These functions read pre-scraped parquet files.
# Run scripts/scrape_external.py first to populate the cache.

def _ext_path(name: str) -> Path:
    return HIST_DIR / name


@st.cache_data(ttl=86400)
def get_nbastuffer_teamstats(season: str, split: str = "regular") -> pd.DataFrame:
    """
    Advanced team stats from nbastuffer.com for a given season.

    split : 'regular' | 'last5' | 'road' | 'home'

    Run ``python scripts/scrape_external.py --source nbastuffer`` to populate.
    """
    ss = season.replace("-", "_")
    if split != "regular":
        path = _ext_path(f"nbastuffer_teamstats_{ss}_{split}.parquet")
    else:
        path = _ext_path(f"nbastuffer_team_{ss}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_nbastuffer_playerstats(season: str) -> pd.DataFrame:
    """
    Advanced player stats from nbastuffer.com for a given season.

    Run ``python scripts/scrape_external.py --source nbastuffer`` to populate.
    """
    ss = season.replace("-", "_")
    path = _ext_path(f"nbastuffer_player_{ss}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_nbastuffer_refstats(season: str) -> pd.DataFrame:
    """
    Season-aggregate referee stats from nbastuffer.com.

    Columns include: REFEREE, LEVEL, GP, W_PCT, AED, PTS,
    FOULS_PER_GAME, HOME_W_PCT, AWAY_W_PCT, HOME_ADV.

    Run ``python scripts/scrape_external.py --source nbastuffer`` to populate.
    """
    ss = season.replace("-", "_")
    path = _ext_path(f"nbastuffer_referee_{ss}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_nbastuffer_restdays(season: str) -> pd.DataFrame:
    """
    Team W%/AED by rest-day type from nbastuffer.com.

    Each row is a team; columns include GP/W_PCT/AED for
    B2B, 3IN4, 1DAY, 2DAY, 3PLUS scenarios.

    Run ``python scripts/scrape_external.py --source nbastuffer`` to populate.
    """
    ss = season.replace("-", "_")
    path = _ext_path(f"nbastuffer_restdays_{ss}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_databallr_shotquality(season: str | None = None) -> pd.DataFrame:
    """
    Shot quality metrics from databallr.com.

    Run ``python scripts/scrape_external.py --source databallr`` to populate.
    """
    ss = season.replace("-", "_") if season else "current"
    path = _ext_path(f"databallr_shotquality_{ss}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_databallr_teams(season: str | None = None) -> pd.DataFrame:
    """
    Team metrics from databallr.com (current season; no historical support).

    Run ``python scripts/scrape_external.py --source databallr`` to populate.
    """
    # databallr only exposes current season data; always use the 'current' file
    path = _ext_path("databallr_teams_current.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(ttl=900)
def get_today_referee_assignments() -> pd.DataFrame:
    """
    Today's referee crew assignments (scraped from official.nba.com).

    Run ``python scripts/scrape_external.py --source refs`` to populate.
    Returns columns that vary by scrape; typically includes game description
    and the three-person officiating crew (Crew Chief, Umpire 1, Umpire 2).
    """
    path = _ext_path("nba_ref_assignments_today.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def build_ref_lookup(season: str) -> dict[str, dict]:
    """
    Build a {referee_name: stats_dict} lookup from the nbastuffer ref stats
    parquet for the given season.  Useful for the Game Predictions page.
    """
    df = get_nbastuffer_refstats(season)
    if df.empty:
        return {}
    # Try to find a name column
    name_col = next(
        (c for c in ["REFEREE", "Name", "name", df.columns[1]] if c in df.columns),
        df.columns[1] if len(df.columns) > 1 else None,
    )
    if name_col is None:
        return {}
    return {
        row[name_col]: row.to_dict()
        for _, row in df.iterrows()
        if pd.notna(row.get(name_col, None))
    }


def build_restdays_lookup(season: str) -> dict[str, dict]:
    """
    Build a {team_name: rest_day_stats_dict} lookup.
    Used by feature_engine to fetch team's rest-type W% for enriched predictions.
    """
    df = get_nbastuffer_restdays(season)
    if df.empty:
        return {}
    team_col = next(
        (c for c in ["TEAM NAME", "TEAM", "Team", "team"] if c in df.columns),
        df.columns[1] if len(df.columns) > 1 else None,
    )
    if team_col is None:
        return {}
    return {
        row[team_col]: row.to_dict()
        for _, row in df.iterrows()
        if pd.notna(row.get(team_col, None))
    }


# ── Pre-cached Game Predictions ───────────────────────────────────────────────

def _predictions_path(date_str: str) -> Path:
    """Return the parquet path for a given YYYY-MM-DD date's predictions."""
    return HIST_DIR / f"predictions_{date_str}.parquet"


def run_and_cache_predictions(date_str: str | None = None) -> pd.DataFrame:
    """
    Run the prediction pipeline for *date_str* (YYYY-MM-DD, defaults to today)
    and persist the result to disk.  Safe to call from non-Streamlit scripts.
    """
    today_str = date_str or datetime.today().strftime("%Y-%m-%d")
    # Convert to MM/DD/YYYY for predict_today_games
    try:
        mm_dd_yyyy = datetime.strptime(today_str, "%Y-%m-%d").strftime("%m/%d/%Y")
    except ValueError:
        mm_dd_yyyy = today_str  # already in another format, pass through
    path = _predictions_path(today_str)
    # Import here to avoid circular import (prediction_engine imports data_fetcher)
    from utils.prediction_engine import predict_today_games  # noqa: PLC0415
    df = predict_today_games(game_date=mm_dd_yyyy)
    if not df.empty:
        df.to_parquet(path, index=False)
    return df if not df.empty else pd.DataFrame()


@st.cache_data(ttl=3600)
def get_today_predictions(game_date_mmddyyyy: str | None = None) -> pd.DataFrame:
    """
    Return today's (or a given date's) game predictions, reading from disk cache
    when available so the ML pipeline only runs once per day.

    game_date_mmddyyyy : MM/DD/YYYY string (matches predict_today_games convention).
                         Defaults to today.
    """
    if game_date_mmddyyyy:
        try:
            date_str = datetime.strptime(game_date_mmddyyyy, "%m/%d/%Y").strftime("%Y-%m-%d")
        except ValueError:
            date_str = datetime.today().strftime("%Y-%m-%d")
    else:
        date_str = datetime.today().strftime("%Y-%m-%d")

    path = _predictions_path(date_str)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return run_and_cache_predictions(date_str)
