"""
Feature engineering pipeline for NBA game outcome and player prop prediction.

Every function uses .shift(1) on rolling/expanding calculations to prevent
data leakage (we only know what happened *before* a given game at prediction time).
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Low-level utilities ────────────────────────────────────────────────────────

def parse_is_home(matchup: str) -> int:
    """Return 1 for home ('vs.'), 0 for away ('@')."""
    return 1 if "vs." in str(matchup) else 0


def compute_rest_days(df: pd.DataFrame, date_col: str = "GAME_DATE") -> pd.DataFrame:
    """Add REST_DAYS: calendar days since last game (capped at 14, default 3)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df["REST_DAYS"] = (
        df[date_col].diff().dt.days.fillna(3).clip(lower=1, upper=14).astype(float)
    )
    return df


def compute_back_to_back(df: pd.DataFrame) -> pd.DataFrame:
    """Add IS_B2B (1 if REST_DAYS == 1) and IS_3IN4 (3 games in 4 nights)."""
    df = df.copy()
    if "REST_DAYS" not in df.columns:
        df = compute_rest_days(df)
    df["IS_B2B"] = (df["REST_DAYS"] == 1).astype(int)
    # 3-in-4: this game AND one of the prior two were a B2B
    df["IS_3IN4"] = (
        df["IS_B2B"].rolling(3, min_periods=2).sum().shift(0) >= 2
    ).astype(int)
    return df


def compute_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add STREAK: +N for N consecutive wins, -N for N consecutive losses.
    Value is the streak *entering* this game (pre-game feature).
    """
    df = df.copy()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    streaks = []
    current = 0
    for wl in df["WL"]:
        streaks.append(current)  # record pre-game streak
        if wl == "W":
            current = max(1, current + 1)
        else:
            current = min(-1, current - 1)
    df["STREAK"] = streaks
    return df


def add_rolling_features(
    df: pd.DataFrame,
    stat_cols: list[str],
    windows: list[int] = [5, 10],
) -> pd.DataFrame:
    """
    For each stat and each window, add:
      - {stat}_L{w}    : rolling mean over last w games (shifted 1 to avoid leakage)
      - {stat}_STD{w}  : rolling std  over last w games (shifted 1)
    """
    df = df.copy().sort_values("GAME_DATE").reset_index(drop=True)
    for col in stat_cols:
        if col not in df.columns:
            continue
        for w in windows:
            min_p = max(1, w // 2)
            df[f"{col}_L{w}"] = (
                df[col].rolling(w, min_periods=min_p).mean().shift(1)
            )
            df[f"{col}_STD{w}"] = (
                df[col].rolling(w, min_periods=min_p).std().shift(1)
            )
    return df


def add_season_averages(df: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """Add season-to-date expanding means (shifted 1 — pre-game value)."""
    df = df.copy().sort_values("GAME_DATE").reset_index(drop=True)
    for col in stat_cols:
        if col not in df.columns:
            continue
        df[f"{col}_SEASON_AVG"] = (
            df[col].expanding(min_periods=1).mean().shift(1)
        )
    return df


def compute_home_away_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add IS_HOME from MATCHUP column."""
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].apply(parse_is_home)
    return df


def compute_win_pct(df: pd.DataFrame, windows: list[int] = [5, 10]) -> pd.DataFrame:
    """Add WIN (binary), WIN_PCT_SEASON and WIN_PCT_L{w}."""
    df = df.copy().sort_values("GAME_DATE").reset_index(drop=True)
    df["WIN"] = (df["WL"] == "W").astype(int)
    df["WIN_PCT_SEASON"] = df["WIN"].expanding(min_periods=1).mean().shift(1)
    for w in windows:
        df[f"WIN_PCT_L{w}"] = df["WIN"].rolling(w, min_periods=1).mean().shift(1)
    return df


# ── Team feature pipeline ──────────────────────────────────────────────────────

TEAM_STAT_COLS = [
    "PTS", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]


def engineer_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full team feature pipeline on a team's game-log DataFrame.

    Expected source: LeagueGameLog (player_or_team='T') for one team.
    Added columns include rolling averages, situational flags, derived rates.
    """
    df = compute_home_away_splits(df)
    df = compute_rest_days(df)
    df = compute_back_to_back(df)
    df = compute_streak(df)
    df = compute_win_pct(df, windows=[5, 10])
    df = add_rolling_features(df, TEAM_STAT_COLS, windows=[5, 10])
    df = add_season_averages(df, ["PTS", "PLUS_MINUS", "FG3_PCT", "TOV", "REB", "AST"])

    # Derived per-game rates
    if {"FGM", "FG3M", "FGA"}.issubset(df.columns):
        df["EFG_PCT"] = (df.get("FGM", 0) + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)
        df["EFG_PCT"] = df["EFG_PCT"].fillna(0)

    if {"PTS", "FGA", "FTA"}.issubset(df.columns):
        denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
        df["TS_PCT"] = df["PTS"] / denom.replace(0, np.nan)
        df["TS_PCT"] = df["TS_PCT"].fillna(0)

    if {"TOV", "FGA", "FTA"}.issubset(df.columns):
        denom = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
        df["TOV_PCT"] = df["TOV"] / denom.replace(0, np.nan)
        df["TOV_PCT"] = df["TOV_PCT"].fillna(0)

    # Oreb rate proxy: OREB / (OREB + opposing DREB) — approximated with OREB / REB
    if {"OREB", "REB"}.issubset(df.columns):
        df["OREB_RATE"] = df["OREB"] / df["REB"].replace(0, np.nan)
        df["OREB_RATE"] = df["OREB_RATE"].fillna(0)

    return df


# ── Player feature pipeline ────────────────────────────────────────────────────

PLAYER_STAT_COLS = [
    "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FG3_PCT", "FT_PCT", "PLUS_MINUS",
]


def engineer_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full player feature pipeline on a PlayerGameLog DataFrame.

    Adds rolling averages, derived stats, and Pick 6 stat combinations.
    """
    df = compute_home_away_splits(df)
    df = compute_rest_days(df)
    df = compute_back_to_back(df)

    # Parse MIN (format: "MM:SS" or float)
    if "MIN" in df.columns and df["MIN"].dtype == object:
        df["MIN"] = pd.to_numeric(
            df["MIN"].str.extract(r"^(\d+)", expand=False), errors="coerce"
        ).fillna(0)

    df = add_rolling_features(df, PLAYER_STAT_COLS, windows=[5, 10])
    df = add_season_averages(df, ["PTS", "REB", "AST", "FG3M", "MIN"])

    # True shooting %
    if {"PTS", "FGA", "FTA"}.issubset(df.columns):
        denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
        df["TS_PCT"] = df["PTS"] / denom.replace(0, np.nan)
        df["TS_PCT"] = df["TS_PCT"].fillna(0)

    # Pick 6 combo stats
    if {"PTS", "REB", "AST"}.issubset(df.columns):
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        df = add_rolling_features(df, ["PRA"], windows=[5, 10])
        df = add_season_averages(df, ["PRA"])

    if {"STL", "BLK"}.issubset(df.columns):
        df["STL_BLK"] = df["STL"] + df["BLK"]
        df = add_rolling_features(df, ["STL_BLK"], windows=[5, 10])

    if {"PTS", "AST"}.issubset(df.columns):
        df["PA"] = df["PTS"] + df["AST"]
        df = add_rolling_features(df, ["PA"], windows=[5, 10])

    if {"PTS", "REB"}.issubset(df.columns):
        df["PR"] = df["PTS"] + df["REB"]
        df = add_rolling_features(df, ["PR"], windows=[5, 10])

    return df


# ── Game-level feature vector construction ────────────────────────────────────

GAME_FEATURE_NAMES = [
    "home_WIN_PCT_SEASON", "away_WIN_PCT_SEASON",
    "home_WIN_PCT_L10",    "away_WIN_PCT_L10",
    "home_WIN_PCT_L5",     "away_WIN_PCT_L5",
    "home_PTS_L10",        "away_PTS_L10",
    "home_PLUS_MINUS_L10", "away_PLUS_MINUS_L10",
    "home_FG3_PCT_L10",    "away_FG3_PCT_L10",
    "home_TOV_L10",        "away_TOV_L10",
    "home_REB_L10",        "away_REB_L10",
    "home_AST_L10",        "away_AST_L10",
    "home_REST_DAYS",      "away_REST_DAYS",
    "home_IS_B2B",         "away_IS_B2B",
    "home_STREAK",         "away_STREAK",
    "win_pct_diff",
    "pts_diff_L10",
    "plus_minus_diff_L10",
    "rest_diff",
    "streak_diff",
    "fg3_pct_diff_L10",
    "tov_diff_L10",
]


def build_game_feature_vector(
    home_row: pd.Series, away_row: pd.Series
) -> pd.Series:
    """
    Combine pre-game team feature rows into a single game feature vector.
    home_row / away_row come from engineer_team_features() output.
    """
    d: dict = {}
    for col in home_row.index:
        d[f"home_{col}"] = home_row[col]
    for col in away_row.index:
        d[f"away_{col}"] = away_row[col]

    # Key differentials (home − away)
    d["win_pct_diff"]        = d.get("home_WIN_PCT_SEASON", 0) - d.get("away_WIN_PCT_SEASON", 0)
    d["pts_diff_L10"]        = d.get("home_PTS_L10", 0)        - d.get("away_PTS_L10", 0)
    d["plus_minus_diff_L10"] = d.get("home_PLUS_MINUS_L10", 0) - d.get("away_PLUS_MINUS_L10", 0)
    d["rest_diff"]           = d.get("home_REST_DAYS", 2)       - d.get("away_REST_DAYS", 2)
    d["streak_diff"]         = d.get("home_STREAK", 0)          - d.get("away_STREAK", 0)
    d["fg3_pct_diff_L10"]    = d.get("home_FG3_PCT_L10", 0)     - d.get("away_FG3_PCT_L10", 0)
    d["tov_diff_L10"]        = d.get("home_TOV_L10", 0)         - d.get("away_TOV_L10", 0)

    return pd.Series(d)


def build_training_dataset(game_log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a model-ready training DataFrame from a full league game log.

    Each row represents one game.  TARGET = 1 if home team won, 0 otherwise.
    Rolling features are already shifted so no future information leaks.

    Parameters
    ----------
    game_log_df : pd.DataFrame
        Concatenated output of get_league_game_log() across multiple seasons.
        Columns expected: GAME_ID, GAME_DATE, TEAM_ID, MATCHUP, WL, PTS, ...
    """
    game_log_df = game_log_df.copy()
    game_log_df["GAME_DATE"] = pd.to_datetime(game_log_df["GAME_DATE"])
    game_log_df["IS_HOME"] = game_log_df["MATCHUP"].apply(parse_is_home)

    # Engineer features per team across all their games
    team_features: dict[int, pd.DataFrame] = {}
    for team_id, grp in game_log_df.groupby("TEAM_ID"):
        team_features[int(team_id)] = engineer_team_features(
            grp.sort_values("GAME_DATE").reset_index(drop=True)
        )

    rows = []
    home_games = game_log_df[game_log_df["IS_HOME"] == 1]

    for _, home_row in home_games.iterrows():
        game_id   = home_row["GAME_ID"]
        home_tid  = int(home_row["TEAM_ID"])
        away_rows = game_log_df[
            (game_log_df["GAME_ID"] == game_id) &
            (game_log_df["TEAM_ID"] != home_tid)
        ]
        if away_rows.empty:
            continue
        away_tid = int(away_rows.iloc[0]["TEAM_ID"])

        hf = team_features.get(home_tid)
        af = team_features.get(away_tid)
        if hf is None or af is None:
            continue

        hfrow = hf[hf["GAME_ID"] == game_id]
        afrow = af[af["GAME_ID"] == game_id]
        if hfrow.empty or afrow.empty:
            continue

        fv = build_game_feature_vector(hfrow.iloc[0], afrow.iloc[0])
        fv["GAME_ID"]      = game_id
        fv["GAME_DATE"]    = home_row["GAME_DATE"]
        fv["HOME_TEAM_ID"] = home_tid
        fv["AWAY_TEAM_ID"] = away_tid
        fv["TARGET"]       = 1 if home_row["WL"] == "W" else 0
        fv["HOME_PTS"]     = home_row.get("PTS", np.nan)
        fv["AWAY_PTS"]     = away_rows.iloc[0].get("PTS", np.nan)
        fv["MARGIN"]       = fv["HOME_PTS"] - fv["AWAY_PTS"]
        fv["TOTAL_PTS"]    = fv["HOME_PTS"] + fv["AWAY_PTS"]
        rows.append(fv)

    return pd.DataFrame(rows).reset_index(drop=True)


def get_training_dataset(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Disk-backed training dataset for a single season.

    Reads from data_files/historical/training_dataset_{season}_{season_type}.parquet
    if it exists. Otherwise builds it from the game log and saves it.
    This lets preload_cache.py pre-build datasets so the Model Performance
    page never has to compute them at page-load time.
    """
    from pathlib import Path
    from utils.data_fetcher import HIST_DIR, get_league_game_log_cached  # lazy import avoids circular dep

    tag  = season_type.replace(" ", "_")
    path = HIST_DIR / f"training_dataset_{season.replace('-', '_')}_{tag}.parquet"

    if path.exists():
        return pd.read_parquet(path)

    game_log = get_league_game_log_cached(season, season_type)
    if game_log.empty:
        return pd.DataFrame()

    df = build_training_dataset(game_log)
    if not df.empty:
        df.to_parquet(path, index=False)
    return df


# ── Player prop feature vector ─────────────────────────────────────────────────

PROP_STAT_MAP = {
    "PTS":     "PTS",
    "REB":     "REB",
    "AST":     "AST",
    "3PM":     "FG3M",
    "PRA":     "PRA",
    "STL":     "STL",
    "BLK":     "BLK",
    "STL+BLK": "STL_BLK",
    "PA":      "PA",
    "PR":      "PR",
}


def build_prop_feature_vector(
    player_game_log: pd.DataFrame,
    stat_col: str,
    opp_def_rating: float = 110.0,
    pace: float = 100.0,
    is_home: int = 1,
    rest_days: int = 2,
    is_b2b: int = 0,
) -> pd.Series:
    """
    Build a feature vector for a single player prop prediction.

    Parameters
    ----------
    player_game_log : DataFrame from get_player_game_log()
    stat_col        : Internal stat column name (use PROP_STAT_MAP to convert)
    opp_def_rating  : Opponent's defensive rating (points per 100 possessions)
    pace            : Game pace estimate (possessions per 48 min)
    """
    df = engineer_player_features(player_game_log)
    if df.empty or stat_col not in df.columns:
        return pd.Series(dtype=float)

    latest = df.sort_values("GAME_DATE").iloc[-1]
    d = {
        f"{stat_col}_L5":         latest.get(f"{stat_col}_L5", np.nan),
        f"{stat_col}_L10":        latest.get(f"{stat_col}_L10", np.nan),
        f"{stat_col}_SEASON_AVG": latest.get(f"{stat_col}_SEASON_AVG", np.nan),
        f"{stat_col}_STD10":      latest.get(f"{stat_col}_STD10", np.nan),
        "OPP_DEF_RATING":         opp_def_rating,
        "PACE":                   pace,
        "IS_HOME":                float(is_home),
        "REST_DAYS":              float(rest_days),
        "IS_B2B":                 float(is_b2b),
    }
    return pd.Series(d)


# ── External-data enrichment helpers ─────────────────────────────────────────
# These functions add supplemental features from scraped nbastuffer / databallr
# data.  They do NOT modify the core GAME_FEATURE_NAMES used by the base model;
# instead they return extra fields that can be displayed or used to train an
# extended model (see EXTENDED_GAME_FEATURE_NAMES below).

# Rest-day scenario column mapping (matches FALLBACK_COLS["restdays"])
_REST_SCENARIO_MAP = {
    1:  ("3IN4_B2B_GP", "3IN4_B2B_W_PCT", "3IN4_B2B_AED"),
    1:  ("B2B_GP",      "B2B_W_PCT",       "B2B_AED"),     # overridden below
}
_REST_SCENARIO_BY_NAME = {
    "b2b":    ("B2B GP",      "B2B W%",       "B2B AED"),
    "3in4":   ("3IN4 GP",     "3IN4 W%",      "3IN4 AED"),
    "1day":   ("1 DAY GP",    "1 DAY W%",     "1 DAY AED"),
    "2day":   ("2 DAYS GP",   "2 DAYS W%",    "2 DAYS AED"),
    "3plus":  ("3+ DAYS GP",  "3+ DAYS W%",   "3+ DAYS AED"),
}


def _rest_scenario_key(rest_days: int, is_b2b: int) -> str:
    """Map rest_days / is_b2b flags to the matching nbastuffer scenario key."""
    if is_b2b:
        return "b2b"
    if rest_days == 1:
        return "1day"
    elif rest_days == 2:
        return "2day"
    elif rest_days >= 3:
        return "3plus"
    return "1day"  # default


def enrich_with_restdays(
    team_name: str,
    rest_days: int,
    is_b2b: int,
    restdays_lookup: dict,
) -> dict:
    """
    Look up a team's historical win rate under today's specific rest situation
    from the nbastuffer rest days data.

    Parameters
    ----------
    team_name       : Partial or full team name, e.g. 'LA Lakers' or 'Lakers'
    rest_days       : Numeric rest days before today's game
    is_b2b          : 1 if the team is playing on a back-to-back
    restdays_lookup : Output of data_fetcher.build_restdays_lookup(season)

    Returns
    -------
    dict with keys: REST_SCENARIO, REST_W_PCT, REST_AED, REST_GP
    """
    empty = {"REST_SCENARIO": None, "REST_W_PCT": np.nan, "REST_AED": np.nan, "REST_GP": 0}
    if not restdays_lookup:
        return empty

    # Fuzzy match team name
    matched_key = None
    for k in restdays_lookup:
        if str(k).lower() in team_name.lower() or team_name.lower() in str(k).lower():
            matched_key = k
            break
    if matched_key is None:
        return empty

    row: dict = restdays_lookup[matched_key]
    scenario = _rest_scenario_key(rest_days, is_b2b)
    gp_col, wpc_col, aed_col = _REST_SCENARIO_BY_NAME.get(scenario, (None, None, None))
    if gp_col is None:
        return empty

    return {
        "REST_SCENARIO": scenario,
        "REST_W_PCT":    _safe_float(row.get(wpc_col, np.nan)),
        "REST_AED":      _safe_float(row.get(aed_col, np.nan)),
        "REST_GP":       int(_safe_float(row.get(gp_col, 0)) or 0),
    }


def enrich_with_referee(referee_name: str, ref_lookup: dict) -> dict:
    """
    Return a referee's aggregated stats from the nbastuffer referee data.

    Parameters
    ----------
    referee_name : Full or partial name of the lead referee
    ref_lookup   : Output of data_fetcher.build_ref_lookup(season)

    Returns
    -------
    dict with keys: REF_NAME, REF_GP, REF_W_PCT, REF_AED, REF_PTS,
                    REF_FOULS_PER_GAME, REF_HOME_W_PCT, REF_AWAY_W_PCT,
                    REF_HOME_ADV
    """
    empty = {
        "REF_NAME": None, "REF_GP": 0, "REF_W_PCT": np.nan,
        "REF_AED": np.nan, "REF_PTS": np.nan,
        "REF_FOULS_PER_GAME": np.nan, "REF_HOME_W_PCT": np.nan,
        "REF_AWAY_W_PCT": np.nan, "REF_HOME_ADV": np.nan,
    }
    if not ref_lookup or not referee_name:
        return empty

    matched_key = None
    for k in ref_lookup:
        if str(k).lower() in referee_name.lower() or referee_name.lower() in str(k).lower():
            matched_key = k
            break
    if matched_key is None:
        return empty

    row: dict = ref_lookup[matched_key]
    return {
        "REF_NAME":           matched_key,
        "REF_GP":             int(_safe_float(row.get("GAMESOFFICIATED", 0)) or 0),
        "REF_W_PCT":          _safe_float(row.get("HOME TEAMWIN%", np.nan)),
        "REF_AED":            _safe_float(row.get("HOME TEAMPOINTS DIFFERENTIAL", np.nan)),
        "REF_PTS":            _safe_float(row.get("TOTALPOINTS PER GAME", np.nan)),
        "REF_FOULS_PER_GAME": _safe_float(row.get("CALLED FOULSPER GAME", np.nan)),
        "REF_HOME_W_PCT":     _safe_float(row.get("HOME TEAMWIN%", np.nan)),
        "REF_AWAY_W_PCT":     _safe_float(row.get("FOUL%AGAINST ROAD TEAMS", np.nan)),
        "REF_HOME_ADV":       _safe_float(row.get("FOUL DIFFERENTIAL(Ag.Rd Tm) - (Ag. Hm Tm)", np.nan)),
    }


def enrich_with_nbastuffer_team(
    team_name: str,
    nbastuffer_team_df: pd.DataFrame,
) -> dict:
    """
    Pull advanced team metrics (SAR, efficiency differential, etc.) from the
    nbastuffer team stats DataFrame for a named team.

    Parameters
    ----------
    team_name            : Partial or full team name
    nbastuffer_team_df   : Output of data_fetcher.get_nbastuffer_teamstats(season)

    Returns
    -------
    dict with keys: NBS_SAR, NBS_eDIFF, NBS_CONS, NBS_a4F, NBS_eWIN_PCT, NBS_W_PCT
    """
    empty = {
        "NBS_SAR": np.nan, "NBS_eDIFF": np.nan, "NBS_CONS": np.nan,
        "NBS_a4F": np.nan, "NBS_eWIN_PCT": np.nan, "NBS_W_PCT": np.nan,
    }
    if nbastuffer_team_df is None or nbastuffer_team_df.empty:
        return empty

    team_col = next(
        (c for c in ["TEAM", "Team", "team"] if c in nbastuffer_team_df.columns),
        nbastuffer_team_df.columns[1] if len(nbastuffer_team_df.columns) > 1 else None,
    )
    if team_col is None:
        return empty

    mask = nbastuffer_team_df[team_col].astype(str).str.lower().str.contains(
        team_name.lower().split()[-1]  # use last word (e.g.  'Lakers' from 'LA Lakers')
    )
    rows = nbastuffer_team_df[mask]
    if rows.empty:
        return empty

    row = rows.iloc[0]
    return {
        "NBS_SAR":      _safe_float(row.get("SAR", np.nan)),
        "NBS_eDIFF":    _safe_float(row.get("eDIFF", np.nan)),
        "NBS_CONS":     _safe_float(row.get("CONS", np.nan)),
        "NBS_a4F":      _safe_float(row.get("A4F", np.nan)),
        "NBS_eWIN_PCT": _safe_float(row.get("eWIN%", np.nan)),
        "NBS_W_PCT":    _safe_float(row.get("WIN%", np.nan)),
    }


def _safe_float(val) -> float:
    """Convert a value to float, returning NaN on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


# ── Extended feature set (superset of GAME_FEATURE_NAMES) ────────────────────
# These features require the external data to be scraped and properly joined.
# They are NOT used by the base ensemble model; use them to train an extended
# model after running scripts/scrape_external.py.

EXTENDED_GAME_FEATURE_NAMES = GAME_FEATURE_NAMES + [
    # Rest-day performance from nbastuffer (previous season, no leakage)
    "home_REST_W_PCT",     "away_REST_W_PCT",
    # Schedule-adjusted rating from nbastuffer
    "home_NBS_SAR",        "away_NBS_SAR",
    # Efficiency differential from nbastuffer
    "home_NBS_eDIFF",      "away_NBS_eDIFF",
    # Consistency rating (lower = more reliable)
    "home_NBS_CONS",       "away_NBS_CONS",
    # Differentials
    "rest_w_pct_diff",
    "sar_diff",
    "efficiency_diff_ext",
]


# ── Odds-derived features ─────────────────────────────────────────────────────

ODDS_FEATURE_NAMES = [
    "implied_prob_home",     # vig-adjusted home win probability from consensus odds
    "implied_prob_away",
    "spread_consensus",      # average spread across available books (home perspective)
    "total_consensus",       # average total line across books
    "odds_disagreement_ml",  # std dev of home ML across books (higher = more uncertain)
    "odds_disagreement_total", # std dev of total across books
]


def add_odds_features(
    feature_row: pd.Series,
    home_mls: dict[str, float],
    away_mls: dict[str, float],
    spreads:  dict[str, float],
    totals:   dict[str, float],
) -> pd.Series:
    """
    Compute odds-derived features from multi-book odds dicts and append them
    to an existing game feature row.

    Parameters
    ----------
    feature_row : pd.Series
        Output of build_game_feature_vector().
    home_mls    : {book: american_odds} for home team moneyline
    away_mls    : {book: american_odds} for away team moneyline
    spreads     : {book: spread} for home team (negative = home favored)
    totals      : {book: total_line}

    Returns
    -------
    pd.Series with additional odds-derived columns appended.
    """
    row = feature_row.copy()

    # Raw implied probabilities per book
    home_probs = [
        _american_ml_to_prob(v) for v in home_mls.values() if v is not None
    ]
    away_probs = [
        _american_ml_to_prob(v) for v in away_mls.values() if v is not None
    ]

    if home_probs and away_probs:
        avg_home_raw = float(np.mean(home_probs))
        avg_away_raw = float(np.mean(away_probs))
        vig = avg_home_raw + avg_away_raw
        row["implied_prob_home"] = avg_home_raw / vig if vig > 0 else 0.5
        row["implied_prob_away"] = avg_away_raw / vig if vig > 0 else 0.5
    else:
        row["implied_prob_home"] = np.nan
        row["implied_prob_away"] = np.nan

    spread_vals = [v for v in spreads.values() if v is not None]
    row["spread_consensus"] = float(np.mean(spread_vals)) if spread_vals else np.nan

    total_vals = [v for v in totals.values() if v is not None]
    row["total_consensus"] = float(np.mean(total_vals)) if total_vals else np.nan

    # Disagreement between books (high std = sharper action / uncertain line)
    home_ml_vals = [v for v in home_mls.values() if v is not None]
    row["odds_disagreement_ml"]    = float(np.std(home_ml_vals)) if len(home_ml_vals) > 1 else 0.0
    row["odds_disagreement_total"] = float(np.std(total_vals))   if len(total_vals)   > 1 else 0.0

    return row


def _american_ml_to_prob(odds: float) -> float:
    """Raw implied probability from American moneyline (not vig-adjusted)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


ODDS_EXTENDED_FEATURE_NAMES = EXTENDED_GAME_FEATURE_NAMES + ODDS_FEATURE_NAMES


# ── Historical odds feature merger ───────────────────────────────────────────

def merge_odds_features(
    training_df: pd.DataFrame,
    hist_dir: "str | None" = None,
) -> pd.DataFrame:
    """
    Join historical odds data from data_files/historical/odds_{season}.parquet
    to the training DataFrame produced by build_training_dataset().

    Adds columns: implied_prob_home, implied_prob_away, spread_consensus,
    total_consensus, odds_disagreement_ml, odds_disagreement_total.

    Rows with no matching odds data have NaN for these columns, which can be
    filled with 0 / median before model training.

    Parameters
    ----------
    training_df : Output of build_training_dataset() with GAME_DATE column.
    hist_dir    : Override for data_files/historical/ directory path.
    """
    from pathlib import Path as _Path

    if hist_dir is None:
        hist_dir = _Path(__file__).resolve().parent.parent / "data_files" / "historical"
    else:
        hist_dir = _Path(hist_dir)

    # Load all available odds parquets
    odds_frames = []
    for p in sorted(hist_dir.glob("odds_*.parquet")):
        try:
            df = pd.read_parquet(p)
            odds_frames.append(df)
        except Exception:
            pass

    if not odds_frames:
        # No odds data on disk — return df with NaN odds columns
        for col in ODDS_FEATURE_NAMES:
            training_df[col] = np.nan
        return training_df

    odds_all = pd.concat(odds_frames, ignore_index=True)
    odds_all["date"] = pd.to_datetime(odds_all["date"]).dt.date

    # Normalise team names: last word lowercased (e.g. "Lakers")
    def _last_word(s: str) -> str:
        return str(s).split()[-1].lower()

    odds_all["_home_key"] = odds_all["home_team"].apply(_last_word)
    odds_all["_away_key"] = odds_all["away_team"].apply(_last_word)

    training_df = training_df.copy()
    training_df["GAME_DATE"] = pd.to_datetime(training_df["GAME_DATE"])

    # Build lookup: (date, home_key, away_key) → row
    # We need team names for training_df — pull from any team-name column available
    # The training_df does NOT have team names, so we join on GAME_ID if it exists,
    # otherwise fall back to a simpler date-based join approach.
    books = [
        "fanduel", "draftkings", "betmgm", "pointsbet",
        "caesars", "wynn", "bet_rivers_ny",
    ]

    # Compute consensus from each odds row
    def _row_to_features(r: pd.Series) -> dict:
        home_mls = {b: r.get(f"ml_home_{b}") for b in books}
        away_mls = {b: r.get(f"ml_away_{b}") for b in books}
        totals   = {b: r.get(f"total_{b}") for b in books}
        spreads  = {b: r.get(f"spread_{b}") for b in books}

        home_probs = [_american_ml_to_prob(v) for v in home_mls.values() if v is not None]
        away_probs = [_american_ml_to_prob(v) for v in away_mls.values() if v is not None]
        total_vals  = [v for v in totals.values() if v is not None]
        spread_vals = [v for v in spreads.values() if v is not None]
        hml_vals    = [v for v in home_mls.values() if v is not None]

        imp_home = imp_away = np.nan
        if home_probs and away_probs:
            raw_h = float(np.mean(home_probs))
            raw_a = float(np.mean(away_probs))
            vig   = raw_h + raw_a
            if vig > 0:
                imp_home = raw_h / vig
                imp_away = raw_a / vig

        return {
            "date":                    r["date"],
            "_home_key":               r["_home_key"],
            "_away_key":               r["_away_key"],
            "implied_prob_home":        imp_home,
            "implied_prob_away":        imp_away,
            "spread_consensus":         float(np.mean(spread_vals)) if spread_vals else np.nan,
            "total_consensus":          float(np.mean(total_vals))  if total_vals  else np.nan,
            "odds_disagreement_ml":     float(np.std(hml_vals))     if len(hml_vals) > 1 else 0.0,
            "odds_disagreement_total":  float(np.std(total_vals))   if len(total_vals) > 1 else 0.0,
        }

    odds_features = pd.DataFrame([_row_to_features(r) for _, r in odds_all.iterrows()])

    # Initialize odds columns in training_df
    for col in ODDS_FEATURE_NAMES:
        training_df[col] = np.nan

    # We can't join directly without team names in training_df.
    # Use a secondary join: merge odds_features into training_df on GAME_DATE only
    # (many-to-many, then resolve by checking home/away team IDs via teams lookup).
    # For simplicity: join on date only, then keep first match per date×game.
    date_to_features: dict = {}
    for _, row in odds_features.iterrows():
        key = (row["date"], row["_home_key"], row["_away_key"])
        date_to_features[key] = row

    # training_df has HOME_TEAM_ID / AWAY_TEAM_ID — need to get team abbreviations.
    # Try to load teams lookup from nba_api static.
    team_last_word: dict[int, str] = {}
    try:
        from nba_api.stats.static import teams as _nba_teams
        for t in _nba_teams.get_teams():
            team_last_word[t["id"]] = t["full_name"].split()[-1].lower()
    except Exception:
        pass

    for idx, trow in training_df.iterrows():
        h_id = trow.get("HOME_TEAM_ID")
        a_id = trow.get("AWAY_TEAM_ID")
        gdate = pd.to_datetime(trow["GAME_DATE"]).date()
        h_key = team_last_word.get(int(h_id), "") if h_id else ""
        a_key = team_last_word.get(int(a_id), "") if a_id else ""

        match = date_to_features.get((gdate, h_key, a_key))
        if match is not None:
            for col in ODDS_FEATURE_NAMES:
                training_df.at[idx, col] = match.get(col, np.nan)

    return training_df
