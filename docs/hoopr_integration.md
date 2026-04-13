# hoopR / sportsdataverse Integration Guide

> **Source:** [hoopr.sportsdataverse.org](https://hoopr.sportsdataverse.org/)
> **Data repo:** [sportsdataverse/hoopR-data](https://github.com/sportsdataverse/hoopR-data)
>
> hoopR is an R package that wraps ESPN, NBA Stats API, KenPom, and NCAA
> endpoints into clean data frames. The pre-built **parquet files** on GitHub
> can be consumed directly from Python — no R required.

---

## Table of Contents

1. [New Data Sources](#1-new-data-sources)
2. [Play-by-Play Derived Features](#2-play-by-play-derived-features)
3. [Advanced Team Box Features](#3-advanced-team-box-features)
4. [Player-Level Enhancements](#4-player-level-enhancements)
5. [Win Probability & Clutch Modeling](#5-win-probability--clutch-modeling)
6. [Shot-Chart / Spatial Features](#6-shot-chart--spatial-features)
7. [Hustle & Tracking Stats](#7-hustle--tracking-stats)
8. [Lineup & Matchup Intelligence](#8-lineup--matchup-intelligence)
9. [Synergy Play-Type Features](#9-synergy-play-type-features)
10. [ESPN Betting Endpoint](#10-espn-betting-endpoint)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. New Data Sources

### 1a. Pre-built Parquet Files (zero API calls)

The `sportsdataverse/hoopR-data` repo publishes daily-updated parquet files.
These are ESPN-sourced and cover **2002–present**.

| Dataset | URL pattern | Key columns |
|---------|------------|-------------|
| Play-by-Play | `nba/pbp/nba_pbp_{YYYY}.parquet` | `coordinate_x/y`, `game_spread`, `scoring_play`, `type_text`, `clock_display_value`, `score_value` |
| Team Box | `nba/team_box/nba_team_box_{YYYY}.parquet` | `fast_break_points`, `points_in_paint`, `turnover_points`, `largest_lead`, `lead_changes`, `lead_percentage` |
| Player Box | `nba/player_box/nba_player_box_{YYYY}.parquet` | `starter`, `ejected`, `did_not_play`, `minutes`, all box stats |
| Schedule | `nba/schedules/nba_schedule_{YYYY}.parquet` | Game metadata, notes, broadcasts |

```python
# utils/hoopr_fetcher.py  —  Data loader for sportsdataverse parquets

import pandas as pd
from pathlib import Path

_BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-data/main"
_CACHE = Path("data_files/hoopr")
_CACHE.mkdir(parents=True, exist_ok=True)


def load_hoopr_parquet(dataset: str, season: int, force: bool = False) -> pd.DataFrame:
    """
    Load a sportsdataverse parquet from disk cache or GitHub.

    Parameters
    ----------
    dataset : str
        One of 'pbp', 'team_box', 'player_box', 'schedule'.
    season : int
        Ending year of the season (e.g. 2025 for 2024-25).
    force : bool
        Re-download even if cached.

    Returns
    -------
    pd.DataFrame
    """
    fname = f"nba_{dataset}_{season}.parquet"
    local = _CACHE / fname
    if local.exists() and not force:
        return pd.read_parquet(local)
    url = f"{_BASE}/nba/{dataset}/nba_{dataset}_{season}.parquet"
    df = pd.read_parquet(url)
    df.to_parquet(local, index=False)
    return df


def load_seasons(dataset: str, seasons: list[int], force: bool = False) -> pd.DataFrame:
    """Load and concatenate multiple seasons."""
    frames = [load_hoopr_parquet(dataset, s, force=force) for s in seasons]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
```

### 1b. ESPN Real-Time Endpoints (via direct HTTP)

hoopR wraps these ESPN endpoints which we can hit directly in Python:

| Endpoint | What it gives us | Use case |
|----------|-----------------|----------|
| `espn_nba_pbp(game_id)` | Live play-by-play | In-game win probability |
| `espn_nba_betting(day)` | Opening/closing lines from ESPN | Free odds cross-reference |
| `espn_nba_wp(game_id)` | Pre-computed win probability curve | Benchmark our model |
| `espn_nba_scoreboard(dates)` | Live/final scores | Replace or supplement nba_api scoreboard |

```python
# ESPN betting endpoint — free alternative to The Odds API for line data

import requests

def fetch_espn_nba_betting(date_str: str) -> pd.DataFrame:
    """
    Fetch ESPN betting lines for a given date (YYYYMMDD).

    Returns DataFrame with columns: game_id, spread, over_under,
    home_team_odds, away_team_odds, provider, etc.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for event in data.get("events", []):
        game_id = event["id"]
        competitions = event.get("competitions", [{}])
        for comp in competitions:
            odds_list = comp.get("odds", [])
            for odds in odds_list:
                rows.append({
                    "game_id": game_id,
                    "provider": odds.get("provider", {}).get("name"),
                    "spread": odds.get("spread"),
                    "over_under": odds.get("overUnder"),
                    "home_ml": odds.get("homeTeamOdds", {}).get("moneyLine"),
                    "away_ml": odds.get("awayTeamOdds", {}).get("moneyLine"),
                    "details": odds.get("details"),
                })
    return pd.DataFrame(rows)
```

> **Cost: $0.** This is a public ESPN API — no key needed, no quota.

### 1c. NBA Stats API Endpoints (via nba_api)

hoopR documents 127+ NBA Stats API endpoints. Key ones we aren't using yet:

| Endpoint | `nba_api` equivalent | Why it matters |
|----------|---------------------|----------------|
| `nba_hustlestatsboxscore` | `boxscorehustlev2` | Deflections, loose balls, screen assists, contested shots |
| `nba_boxscoreplayertrackv3` | `boxscoreplayertrackv3` | Distance, speed, touches, passes, paint touches |
| `nba_leaguedashlineups` | `leaguedashlineups` | Net ratings for 5-man combos |
| `nba_synergyplaytypes` | `synergyplaytypes` | PnR, ISO, transition, post-up frequencies & PPP |
| `nba_shotchartdetail` | `shotchartdetail` | Shot x/y coords, zone, distance, make/miss |
| `nba_leaguedashplayerclutch` | `leaguedashplayerclutch` | Player stats in clutch situations |
| `nba_leaguedashteamclutch` | `leaguedashteamclutch` | Team stats in clutch situations |
| `nba_winprobabilitypbp` | `winprobabilitypbp` | Official NBA win probability per play |
| `nba_boxscorefourfactorsv3` | `boxscorefourfactorsv3` | eFG%, TOV%, OREB%, FT rate per game |
| `nba_teamplayeronoffdetails` | `teamplayeronoffdetails` | On/off court net rating per player |

---

## 2. Play-by-Play Derived Features

The PBP parquet has **64 columns** per play, including `coordinate_x`, `coordinate_y`,
`scoring_play`, `shooting_play`, `type_text`, `clock_display_value`, and `score_value`.

### Feature Ideas

| Feature | Description | Model impact |
|---------|-------------|--------------|
| `pace_possessions` | Estimated possessions from PBP events (FGA + 0.44*FTA + TOV - OREB) | Controls for game tempo |
| `run_frequency` | # of 8-0+ runs per game | Captures team resilience / volatility |
| `clutch_net_rating` | Net rating in last 5 min when score within 5 | Predicts close-game outcomes |
| `transition_pct` | % of scoring plays within 7 sec of possession start | Fast-break teams overperform vs slow teams |
| `shot_quality_xy` | Expected FG% from (x, y) coords using league-wide make rate | Better than raw FG% |
| `garbage_time_adj` | Exclude plays when lead > 20 in Q4 | Cleaner signal for actual competitiveness |
| `scoring_drought_avg` | Average longest scoreless streak per game | Captures offensive consistency |

```python
# utils/pbp_features.py  —  Play-by-play feature engineering

import pandas as pd
import numpy as np


def engineer_pbp_team_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ESPN PBP data into per-team-per-game features.

    Expected input: output of load_hoopr_parquet('pbp', season).
    """
    # Normalize column names
    pbp = pbp.copy()
    pbp.columns = pbp.columns.str.lower()

    # Filter to plays with a team attribution
    plays = pbp[pbp["team_id"].notna()].copy()
    plays["game_id"] = plays["game_id"].astype(str)
    plays["team_id"] = plays["team_id"].astype(str)

    features = []
    for (gid, tid), gdf in plays.groupby(["game_id", "team_id"]):
        row = {"game_id": gid, "team_id": tid}

        # ── Scoring distribution ──────────────────────────────
        scoring = gdf[gdf["scoring_play"] == True]
        row["total_scoring_plays"] = len(scoring)
        row["avg_score_value"] = scoring["score_value"].mean() if len(scoring) else 0

        # ── Shot quality from coordinates ─────────────────────
        shots = gdf[gdf["shooting_play"] == True]
        if len(shots) and "coordinate_x" in shots.columns:
            # Distance from basket — ESPN coords: basket at (25, 0) in half-court
            shots = shots.copy()
            shots["shot_dist"] = np.sqrt(
                (shots["coordinate_x"] - 25) ** 2 +
                (shots["coordinate_y"]) ** 2
            )
            row["avg_shot_distance"] = shots["shot_dist"].mean()
            row["pct_paint_shots"] = (shots["shot_dist"] < 8).mean()
            row["pct_three_range"] = (shots["shot_dist"] >= 22).mean()
            row["shot_distance_std"] = shots["shot_dist"].std()
        else:
            row["avg_shot_distance"] = np.nan
            row["pct_paint_shots"] = np.nan
            row["pct_three_range"] = np.nan
            row["shot_distance_std"] = np.nan

        # ── Run detection (scoring runs of 6+ unanswered) ────
        # Sort by game clock descending (start of game → end)
        g_plays = pbp[(pbp["game_id"].astype(str) == gid)].sort_values(
            "start_game_seconds_remaining", ascending=False
        )
        if len(g_plays) > 0:
            runs = _detect_runs(g_plays, tid, min_run=6)
            row["big_runs_for"] = runs["runs_for"]
            row["big_runs_against"] = runs["runs_against"]
            row["max_run_for"] = runs["max_run_for"]
        else:
            row["big_runs_for"] = 0
            row["big_runs_against"] = 0
            row["max_run_for"] = 0

        # ── Clutch performance (last 5 min, score within 5) ──
        if "start_game_seconds_remaining" in gdf.columns:
            clutch_mask = (gdf["start_game_seconds_remaining"] <= 300) & (gdf["period"] >= 4)
            clutch = gdf[clutch_mask]
            clutch_scoring = clutch[clutch["scoring_play"] == True]
            row["clutch_scoring_plays"] = len(clutch_scoring)
            row["clutch_total_plays"] = len(clutch)
        else:
            row["clutch_scoring_plays"] = 0
            row["clutch_total_plays"] = 0

        # ── Transition plays (scoring within first 7 seconds) ─
        if "start_game_seconds_remaining" in gdf.columns and "end_game_seconds_remaining" in gdf.columns:
            early = scoring[
                (scoring["start_game_seconds_remaining"] - scoring["end_game_seconds_remaining"]) <= 7
            ] if len(scoring) else pd.DataFrame()
            row["transition_scoring_pct"] = len(early) / max(len(scoring), 1)
        else:
            row["transition_scoring_pct"] = np.nan

        features.append(row)

    return pd.DataFrame(features)


def _detect_runs(game_plays: pd.DataFrame, team_id: str, min_run: int = 6) -> dict:
    """Detect scoring runs for/against a team in a single game."""
    scoring = game_plays[game_plays["scoring_play"] == True].copy()
    runs_for = 0
    runs_against = 0
    max_run_for = 0
    current_run = 0
    current_team = None

    for _, play in scoring.iterrows():
        play_team = str(play.get("team_id", ""))
        pts = play.get("score_value", 0) or 0

        if play_team == current_team:
            current_run += pts
        else:
            # Check completed run
            if current_run >= min_run:
                if current_team == team_id:
                    runs_for += 1
                    max_run_for = max(max_run_for, current_run)
                else:
                    runs_against += 1
            current_run = pts
            current_team = play_team

    # Final streak
    if current_run >= min_run:
        if current_team == team_id:
            runs_for += 1
            max_run_for = max(max_run_for, current_run)
        else:
            runs_against += 1

    return {"runs_for": runs_for, "runs_against": runs_against, "max_run_for": max_run_for}
```

---

## 3. Advanced Team Box Features

The hoopR team box parquet contains columns **not available in `nba_api`'s standard
`LeagueGameLog`**:

| Column | What it is | Why it matters for prediction |
|--------|-----------|-------------------------------|
| `fast_break_points` | Points scored in transition | High fast-break teams blow out poor transition defenders |
| `points_in_paint` | Points scored inside the paint | Paint dominance correlates with winning |
| `turnover_points` | Points scored off opponent turnovers | Efficiency of converting steals |
| `largest_lead` | Max lead held during game | Game control indicator — teams that build big leads tend to win more consistently |
| `lead_changes` | Number of lead changes | Proxy for competitiveness — predict close games |
| `lead_percentage` | % of game with the lead | Dominant predictor of outcome in rolling averages |

```python
# Feature integration for team box extras

def engineer_hoopr_team_box_features(team_box: pd.DataFrame) -> pd.DataFrame:
    """
    Add features from sportsdataverse team box data to the existing pipeline.

    Must be called per-team, sorted by game_date.  Uses .shift(1) to avoid leakage.
    """
    df = team_box.copy()
    df.columns = df.columns.str.lower()
    df = df.sort_values("game_date").reset_index(drop=True)

    extra_cols = [
        "fast_break_points", "points_in_paint", "turnover_points",
        "largest_lead", "lead_changes", "lead_percentage",
    ]

    for col in extra_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Rolling averages (L5, L10)
        for w in [5, 10]:
            df[f"{col}_L{w}"] = (
                df[col].rolling(w, min_periods=max(1, w // 2)).mean().shift(1)
            )
        # Season average
        df[f"{col}_season_avg"] = df[col].expanding(min_periods=1).mean().shift(1)

    # ── Derived features ──
    if {"fast_break_points", "points_in_paint"}.issubset(set(df.columns)):
        df["paint_dominance_L10"] = df.get("points_in_paint_L10", 0) - df.get("fast_break_points_L10", 0)

    if "lead_percentage" in df.columns:
        df["lead_pct_trend"] = (
            df["lead_percentage_L5"].fillna(0.5) - df["lead_percentage_L10"].fillna(0.5)
        )

    return df


# New GAME_FEATURE_NAMES to add to model_utils.FEATURE_COLS_GAME_EXTENDED:
HOOPR_TEAM_BOX_FEATURES = [
    "home_fast_break_points_L10",  "away_fast_break_points_L10",
    "home_points_in_paint_L10",    "away_points_in_paint_L10",
    "home_turnover_points_L10",    "away_turnover_points_L10",
    "home_largest_lead_L10",       "away_largest_lead_L10",
    "home_lead_percentage_L10",    "away_lead_percentage_L10",
    "paint_dominance_diff",        # home - away
    "lead_pct_diff",               # home - away
]
```

---

## 4. Player-Level Enhancements

### 4a. Starter Detection

The hoopR player box has a `starter` boolean column. Use it to weight
player contributions when building team-level features:

```python
def compute_starter_weighted_stats(player_box: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team-game level features weighted by starter/bench status.

    Returns one row per (game_id, team_id) with:
      - starter_avg_pts, bench_avg_pts
      - starter_total_plus_minus, bench_total_plus_minus
      - bench_scoring_pct  (what % of team points came from bench)
    """
    df = player_box.copy()
    df.columns = df.columns.str.lower()
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    df["plus_minus"] = pd.to_numeric(df.get("plus_minus", 0), errors="coerce").fillna(0)

    results = []
    for (gid, tid), gdf in df.groupby(["game_id", "team_id"]):
        starters = gdf[gdf["starter"] == True]
        bench = gdf[gdf["starter"] == False]

        total_pts = gdf["points"].sum()
        bench_pts = bench["points"].sum()
        results.append({
            "game_id": gid,
            "team_id": tid,
            "starter_avg_pts": starters["points"].mean() if len(starters) else 0,
            "bench_avg_pts": bench["points"].mean() if len(bench) else 0,
            "starter_plus_minus": starters["plus_minus"].sum(),
            "bench_plus_minus": bench["plus_minus"].sum(),
            "bench_scoring_pct": bench_pts / max(total_pts, 1),
            "players_used": len(gdf[gdf["minutes"] > 0]),
        })
    return pd.DataFrame(results)
```

### 4b. Did Not Play / Ejection Tracking

```python
def compute_availability_features(player_box: pd.DataFrame) -> pd.DataFrame:
    """Flag games where key rotation players (top 8 by avg MIN) were missing."""
    df = player_box.copy()
    df.columns = df.columns.str.lower()
    df["did_not_play"] = df["did_not_play"].fillna(False)
    df["ejected"] = df["ejected"].fillna(False)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)

    # Identify top-8 minute earners per team across the season
    season_mins = df.groupby(["team_id", "athlete_id"])["minutes"].mean()
    top8 = season_mins.groupby("team_id").nlargest(8).reset_index(level=0, drop=True)
    top8_ids = set(top8.index.get_level_values("athlete_id"))

    results = []
    for (gid, tid), gdf in df.groupby(["game_id", "team_id"]):
        rotation = gdf[gdf["athlete_id"].isin(top8_ids)]
        results.append({
            "game_id": gid,
            "team_id": tid,
            "rotation_dnp_count": rotation["did_not_play"].sum(),
            "rotation_ejected_count": rotation["ejected"].sum(),
        })
    return pd.DataFrame(results)
```

---

## 5. Win Probability & Clutch Modeling

### 5a. ESPN Win Probability as a Free Benchmark

```python
def fetch_espn_win_probability(game_id: str) -> pd.DataFrame:
    """
    Fetch ESPN's pre-computed win probability for a completed/live game.

    Returns DataFrame with columns: play_id, seconds_left, home_win_pct.
    """
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
        f"summary?event={game_id}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    wp_data = data.get("winprobability", [])
    rows = []
    for wp in wp_data:
        rows.append({
            "play_id": wp.get("playId"),
            "seconds_left": wp.get("secondsLeft"),
            "home_win_pct": wp.get("homeWinPercentage"),
        })
    return pd.DataFrame(rows)
```

### 5b. Build Your Own Pre-Game Win Probability

Use the PBP-derived features + existing team features to train a calibrated
regression that outputs a true probability:

```python
# In scripts/train_models.py — add a WinProb regressor

from sklearn.isotonic import IsotonicRegression

def train_pregame_winprob(X_train, y_train, base_model):
    """
    Train a pre-game win probability model using isotonic calibration
    on top of the ensemble's raw output.

    base_model : already-trained ensemble from model_utils
    """
    raw_probs = base_model.predict_proba(X_train)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, y_train)
    return calibrator
```

### 5c. Clutch Performance Features

```python
def compute_clutch_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Extract clutch-time performance per team per game.

    Clutch = last 5 minutes of Q4+ when score differential <= 5.
    """
    df = pbp.copy()
    df.columns = df.columns.str.lower()

    # Filter to clutch situations
    clutch = df[
        (df["period"] >= 4) &
        (df["start_game_seconds_remaining"] <= 300)
    ].copy()

    # Approximate score differential from play text
    # (home_score and away_score columns in some PBP formats)
    if {"home_score", "away_score"}.issubset(clutch.columns):
        clutch["score_diff"] = abs(
            pd.to_numeric(clutch["home_score"], errors="coerce") -
            pd.to_numeric(clutch["away_score"], errors="coerce")
        )
        clutch = clutch[clutch["score_diff"] <= 5]

    features = []
    for (gid, tid), gdf in clutch.groupby(["game_id", "team_id"]):
        scoring = gdf[gdf["scoring_play"] == True]
        features.append({
            "game_id": gid,
            "team_id": tid,
            "clutch_pts": scoring["score_value"].sum() if len(scoring) else 0,
            "clutch_plays": len(gdf),
            "clutch_fg_pct": (
                len(gdf[gdf["scoring_play"] == True]) /
                max(len(gdf[gdf["shooting_play"] == True]), 1)
            ),
        })
    return pd.DataFrame(features)
```

---

## 6. Shot-Chart / Spatial Features

The PBP parquet has `coordinate_x` and `coordinate_y` per shooting play.
ESPN uses a coordinate system where the court is ~50 x ~94, with the basket
near `(25, 0)` for the relevant half-court.

### 6a. Expected Field Goal Percentage (xFG%)

Train a league-wide model that predicts make/miss from shot location,
then compute each team's shot quality relative to expected.

```python
from sklearn.ensemble import GradientBoostingClassifier

def train_xfg_model(pbp_all: pd.DataFrame) -> GradientBoostingClassifier:
    """
    Train a shot-quality model: P(make) given (x, y, shot_type, period).
    
    Input: multi-season PBP with coordinate_x, coordinate_y, scoring_play, shooting_play.
    """
    shots = pbp_all[pbp_all["shooting_play"] == True].dropna(
        subset=["coordinate_x", "coordinate_y"]
    ).copy()

    shots["shot_dist"] = np.sqrt(
        (shots["coordinate_x"] - 25) ** 2 + shots["coordinate_y"] ** 2
    )
    shots["is_three"] = (shots["shot_dist"] >= 22).astype(int)
    shots["made"] = shots["scoring_play"].astype(int)

    X = shots[["coordinate_x", "coordinate_y", "shot_dist", "is_three", "period"]]
    y = shots["made"]

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8
    )
    model.fit(X, y)
    return model


def compute_team_xfg_diff(
    pbp_season: pd.DataFrame, xfg_model: GradientBoostingClassifier
) -> pd.DataFrame:
    """
    For each team-game, compute:
      - xFG%: average predicted make rate based on shot locations
      - actual_FG%: actual make rate
      - shot_quality_diff: actual - expected  (positive = they make hard shots)
    """
    shots = pbp_season[pbp_season["shooting_play"] == True].dropna(
        subset=["coordinate_x", "coordinate_y"]
    ).copy()

    shots["shot_dist"] = np.sqrt(
        (shots["coordinate_x"] - 25) ** 2 + shots["coordinate_y"] ** 2
    )
    shots["is_three"] = (shots["shot_dist"] >= 22).astype(int)
    shots["made"] = shots["scoring_play"].astype(int)

    X = shots[["coordinate_x", "coordinate_y", "shot_dist", "is_three", "period"]]
    shots["xFG"] = xfg_model.predict_proba(X)[:, 1]

    result = shots.groupby(["game_id", "team_id"]).agg(
        xFG_pct=("xFG", "mean"),
        actual_FG_pct=("made", "mean"),
        shot_count=("made", "count"),
    ).reset_index()

    result["shot_quality_diff"] = result["actual_FG_pct"] - result["xFG_pct"]
    return result
```

### 6b. Opponent Shot Quality Allowed

Same idea but **from the defensive perspective** — what quality of shots
does a team *allow*? A low opponent xFG% = good defense.

```python
def compute_opponent_xfg(
    pbp_season: pd.DataFrame, xfg_model, team_id_col: str = "team_id"
) -> pd.DataFrame:
    """
    Compute the shot quality that each team ALLOWS their opponents.
    """
    shots = pbp_season[pbp_season["shooting_play"] == True].dropna(
        subset=["coordinate_x", "coordinate_y"]
    ).copy()

    shots["shot_dist"] = np.sqrt(
        (shots["coordinate_x"] - 25) ** 2 + shots["coordinate_y"] ** 2
    )
    shots["is_three"] = (shots["shot_dist"] >= 22).astype(int)

    X = shots[["coordinate_x", "coordinate_y", "shot_dist", "is_three", "period"]]
    shots["xFG"] = xfg_model.predict_proba(X)[:, 1]

    # Group by OPPONENT (i.e. the other team_id in the game)
    # We need game_id + both sides to identify opponent
    # For simplicity, aggregate at game_id level, then join back
    game_agg = shots.groupby(["game_id", team_id_col]).agg(
        opp_xFG_allowed=("xFG", "mean"),
        opp_paint_shot_pct=("shot_dist", lambda x: (x < 8).mean()),
    ).reset_index()

    return game_agg
```

---

## 7. Hustle & Tracking Stats

These require per-game `nba_api` calls (hustle/tracking box scores aren't in
the hoopR parquet) but are enormously predictive.

### Available via `nba_api`

| Stat | Endpoint | Column |
|------|----------|--------|
| Deflections | `boxscorehustlev2` | `deflections` |
| Loose balls recovered | `boxscorehustlev2` | `loose_balls_recovered` |
| Screen assists | `boxscorehustlev2` | `screen_assists` |
| Contested 2PT shots | `boxscorehustlev2` | `contested_shots_2pt` |
| Contested 3PT shots | `boxscorehustlev2` | `contested_shots_3pt` |
| Charges drawn | `boxscorehustlev2` | `charges_drawn` |
| Distance (miles) | `boxscoreplayertrackv3` | `dist_miles` |
| Avg speed | `boxscoreplayertrackv3` | `avg_speed` |
| Touches | `boxscoreplayertrackv3` | `touches` |
| Paint touches | `boxscoreplayertrackv3` | `paint_touches` |
| Time of possession | `boxscoreplayertrackv3` | `time_of_poss` |

```python
# utils/hustle_features.py

from nba_api.stats.endpoints import boxscorehustlev2
import time

def fetch_hustle_stats(game_id: str) -> pd.DataFrame:
    """Fetch hustle stats for a single game. Rate-limited."""
    time.sleep(0.7)
    hustle = boxscorehustlev2.BoxScoreHustleV2(game_id=game_id)
    team_stats = hustle.get_data_frames()[1]  # Team-level hustle
    return team_stats


HUSTLE_FEATURES = [
    "deflections", "loose_balls_recovered", "screen_assists",
    "contested_shots", "charges_drawn",
]

# Add to FEATURE_COLS for the model:
HUSTLE_GAME_FEATURES = [
    "home_deflections_L10",        "away_deflections_L10",
    "home_contested_shots_L10",    "away_contested_shots_L10",
    "home_loose_balls_L10",        "away_loose_balls_L10",
    "deflections_diff",
    "contested_shots_diff",
]
```

> **Note:** Hustle stats require one API call per game. For historical backfill,
> batch during `train_models.py` and cache to parquet.

---

## 8. Lineup & Matchup Intelligence

### 8a. Five-Man Lineup Net Rating

```python
from nba_api.stats.endpoints import leaguedashlineups

def fetch_top_lineups(team_id: int, season: str = "2025-26", top_n: int = 5) -> pd.DataFrame:
    """
    Fetch the top N most-used 5-man lineups for a team and their net ratings.
    """
    time.sleep(0.7)
    lineups = leaguedashlineups.LeagueDashLineups(
        team_id_nullable=team_id,
        season=season,
        measure_type_detailed_defense="Base",
        group_quantity=5,
    )
    df = lineups.get_data_frames()[0]
    df = df.sort_values("MIN", ascending=False).head(top_n)
    return df[["GROUP_NAME", "MIN", "W_PCT", "NET_RATING", "PLUS_MINUS"]]
```

### 8b. Player On/Off Differential

The `nba_teamplayeronoffdetails` endpoint reveals each player's impact:

```python
from nba_api.stats.endpoints import teamplayeronoffdetails

def fetch_on_off_impact(team_id: int, season: str = "2025-26") -> pd.DataFrame:
    """
    Get on-court vs off-court net rating differential for each player.
    High differential = indispensable player (if injured, big impact).
    """
    time.sleep(0.7)
    result = teamplayeronoffdetails.TeamPlayerOnOffDetails(
        team_id=team_id, season=season
    )
    on_court = result.get_data_frames()[0]   # PlayersOnCourtTeamPlayerOnOffDetails
    off_court = result.get_data_frames()[1]  # PlayersOffCourtTeamPlayerOnOffDetails

    merged = on_court[["VS_PLAYER_NAME", "NET_RATING"]].rename(
        columns={"NET_RATING": "on_court_net_rating"}
    ).merge(
        off_court[["VS_PLAYER_NAME", "NET_RATING"]].rename(
            columns={"NET_RATING": "off_court_net_rating"}
        ),
        on="VS_PLAYER_NAME",
    )
    merged["on_off_diff"] = merged["on_court_net_rating"] - merged["off_court_net_rating"]
    return merged.sort_values("on_off_diff", ascending=False)
```

> **Modeling idea:** When a player with `on_off_diff > 8` is on the injury
> report, apply a penalty to the team's predicted win probability proportional
> to their differential.

---

## 9. Synergy Play-Type Features

The NBA Stats API has Synergy play-type data — how often teams run each
offensive action and how efficient they are at it.

```python
from nba_api.stats.endpoints import synergyplaytypes

def fetch_team_play_types(season: str = "2025-26") -> pd.DataFrame:
    """
    Fetch Synergy play-type frequencies and efficiency for all teams.

    Play types: Transition, Isolation, PRBallHandler, PRRollman,
    Postup, Spotup, Handoff, Cut, OffScreen, OffRebound, Misc
    """
    time.sleep(0.7)
    synergy = synergyplaytypes.SynergyPlayTypes(
        season=season,
        play_type_nullable="",
        player_or_team_abbreviation="T",
        type_grouping_nullable="offensive",
    )
    return synergy.get_data_frames()[0]
```

**Feature ideas from play types:**

| Feature | Description | Why it helps |
|---------|-------------|--------------|
| `transition_freq` | % of plays that are transition | Fast teams exploit slow opponents |
| `iso_ppp` | Points-per-possession on isolations | Quality of shot creation |
| `pnr_freq` | PnR ball handler frequency | Most common NBA action |
| `postup_ppp_diff` | Your post-up PPP vs opponent's post defense PPP | Matchup-specific edge |
| `spot_up_freq` | Spot-up 3PT frequency | Floor spacing indicator |

---

## 10. ESPN Betting Endpoint

ESPN exposes betting lines for free — **no API key required**. This can
supplement or replace The Odds API for consensus lines.

```python
def fetch_espn_lines_for_date(date_str: str) -> pd.DataFrame:
    """
    Pull opening lines, spreads, and totals from ESPN's public API.

    Parameters
    ----------
    date_str : str
        Format: YYYYMMDD

    Returns
    -------
    DataFrame with: game_id, home_team, away_team, spread, over_under, provider
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        home = away = None
        for team_info in comp.get("competitors", []):
            if team_info.get("homeAway") == "home":
                home = team_info.get("team", {}).get("abbreviation")
            else:
                away = team_info.get("team", {}).get("abbreviation")

        for odds in comp.get("odds", []):
            rows.append({
                "game_id": event["id"],
                "home_team": home,
                "away_team": away,
                "provider": odds.get("provider", {}).get("name"),
                "spread": odds.get("spread"),
                "over_under": odds.get("overUnder"),
                "home_ml": odds.get("homeTeamOdds", {}).get("moneyLine"),
                "away_ml": odds.get("awayTeamOdds", {}).get("moneyLine"),
            })
    return pd.DataFrame(rows)
```

> **This replaces 1 Odds API call per day with $0 cost.** Consider using this
> as the primary source and The Odds API only as a fallback for sportsbook-
> specific lines.

---

## 11. Implementation Roadmap

### Phase 1 — Quick Wins (no retraining needed, immediate value)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | Add `utils/hoopr_fetcher.py` — download team_box parquets | 1 hr | Unlocks all Phase 2 features |
| 2 | Add ESPN betting endpoint to `data_fetcher.py` | 30 min | Free odds data, reduce Odds API usage to 0 |
| 3 | Display ESPN win probability curve on Game Predictions page | 1 hr | Cool visualization, zero model change |
| 4 | Show bench scoring % on Team Stats page | 30 min | New insight from player_box starter flag |

### Phase 2 — Feature Expansion (requires retraining)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 5 | Integrate `fast_break_points`, `points_in_paint`, `turnover_points`, `lead_percentage` as rolling features | 2 hr | High — these are strong predictors |
| 6 | Add PBP-derived `clutch_scoring_plays`, `transition_scoring_pct` | 3 hr | Medium-high — captures late-game strength |
| 7 | Add `bench_scoring_pct`, `rotation_dnp_count` per game | 2 hr | Medium — roster depth signal |
| 8 | Build xFG% model from shot coordinates | 4 hr | High — shot quality > raw FG% |
| 9 | Add features to `FEATURE_COLS_GAME_EXTENDED` and retrain | 1 hr | Required after adding new features |

### Phase 3 — Advanced Modeling

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 10 | Train xFG model on 5+ seasons of PBP data | 4 hr | Foundation for shot quality features |
| 11 | Backfill hustle stats for 2021-22 through present | 6 hr | Deflections/contested shots are elite predictors |
| 12 | Player on/off penalty for injuries | 3 hr | Dynamic adjustment when stars sit |
| 13 | Synergy play-type matchup features | 4 hr | Team-vs-team specific edges |
| 14 | Build live in-game win probability model using PBP | 8 hr | Flagship feature |

### Phase 4 — Visualization & UX

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 15 | Shot chart heatmap on Team Stats page | 3 hr | Beautiful, engaging |
| 16 | Lineup net rating explorer | 2 hr | Deep analysis tool |
| 17 | Play-type breakdown radar chart | 2 hr | Unique matchup insight |
| 18 | Win probability chart for completed games | 2 hr | Post-game analysis |

---

## Appendix: New Feature Columns for model_utils.py

When ready to integrate, add these to `FEATURE_COLS_GAME_EXTENDED`:

```python
# model_utils.py — append to FEATURE_COLS_GAME_EXTENDED

HOOPR_FEATURE_COLS = [
    # Team box extras (Phase 2)
    "home_fast_break_points_L10",    "away_fast_break_points_L10",
    "home_points_in_paint_L10",      "away_points_in_paint_L10",
    "home_turnover_points_L10",      "away_turnover_points_L10",
    "home_lead_percentage_L10",      "away_lead_percentage_L10",
    "paint_diff_L10",                # home - away points_in_paint_L10
    "lead_pct_diff_L10",             # home - away lead_percentage_L10
    "fastbreak_diff_L10",            # home - away fast_break_points_L10

    # PBP-derived (Phase 2-3)
    "home_clutch_scoring_L10",       "away_clutch_scoring_L10",
    "home_transition_pct_L10",       "away_transition_pct_L10",
    "home_avg_shot_distance_L10",    "away_avg_shot_distance_L10",
    "home_pct_paint_shots_L10",      "away_pct_paint_shots_L10",

    # Shot quality (Phase 3)
    "home_xFG_pct_L10",             "away_xFG_pct_L10",
    "home_shot_quality_diff_L10",    "away_shot_quality_diff_L10",
    "xFG_diff",                      # home - away

    # Roster depth (Phase 2)
    "home_bench_scoring_pct_L10",    "away_bench_scoring_pct_L10",
    "home_rotation_dnp_count",       "away_rotation_dnp_count",
    "bench_scoring_diff",

    # Hustle (Phase 3)
    "home_deflections_L10",          "away_deflections_L10",
    "home_contested_shots_L10",      "away_contested_shots_L10",
    "deflections_diff",
    "contested_diff",
]
```

---

## Appendix: Quick-Start Script

Run this to download all hoopR data and verify everything works:

```python
# scripts/fetch_hoopr_data.py

"""Download sportsdataverse parquets for all configured seasons."""

import sys
sys.path.insert(0, ".")
from utils.hoopr_fetcher import load_hoopr_parquet

SEASONS = list(range(2022, 2027))  # 2021-22 through 2025-26

def main():
    for dataset in ["pbp", "team_box", "player_box", "schedule"]:
        for season in SEASONS:
            try:
                df = load_hoopr_parquet(dataset, season, force=True)
                print(f"  {dataset} {season}: {len(df):,} rows, {len(df.columns)} cols")
            except Exception as e:
                print(f"  {dataset} {season}: FAILED — {e}")

if __name__ == "__main__":
    main()
```

Expected output:
```
  pbp 2022: ~600,000 rows, 64 cols
  pbp 2023: ~620,000 rows, 64 cols
  ...
  team_box 2025: ~2,400 rows, 59 cols
  player_box 2025: ~24,000 rows, 57 cols
```

---

*Last updated: auto-generated from hoopR docs review*
