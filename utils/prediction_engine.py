"""
Prediction pipeline for game outcomes and DraftKings Pick 6 props.

Used by Streamlit pages to generate daily predictions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from utils.data_fetcher import (
    get_today_scoreboard,
    get_league_game_log_cached,
    get_team_estimated_metrics_cached,
    get_player_game_log_cached,
    get_implied_probs,
    get_all_teams,
    CURRENT_SEASON,
)
from utils.feature_engine import (
    build_training_dataset,
    build_game_feature_vector,
    engineer_team_features,
    PROP_STAT_MAP,
    GAME_FEATURE_NAMES,
)
from utils.model_utils import (
    load_models,
    ensemble_predict_proba,
    get_model_features,
    EloSystem,
    MODEL_DIR,
    FEATURE_COLS_GAME,
    load_totals_model,
    FEATURE_COLS_TOTALS,
)

# DraftKings Pick 6 payout multipliers
PAYOUT_TABLE: dict[int, int] = {2: 3, 3: 5, 4: 10, 5: 20, 6: 40}


# ── Game Outcome Utilities ─────────────────────────────────────────────────────

def win_prob_to_spread(home_win_prob: float) -> float:
    """
    Convert home team win probability to approximate point spread.
    Based on logistic calibration against historical closing lines.
    """
    p = np.clip(home_win_prob, 0.01, 0.99)
    return round(-13.0 * np.log((1.0 - p) / p), 1)


def assign_confidence_tier(
    model_prob: float,
    market_prob: float | None = None,
) -> str:
    """
    Assign High / Medium / Low confidence tier.

    High   : model >= 65% AND edge vs market >= 5%  (or no market and >= 65%)
    Medium : model >= 57%  OR  edge >= 2%
    Low    : everything else
    """
    edge = abs(model_prob - market_prob) if market_prob is not None else None
    if model_prob >= 0.65 and (edge is None or edge >= 0.05):
        return "High"
    if model_prob >= 0.57 or (edge is not None and edge >= 0.02):
        return "Medium"
    return "Low"


# ── Main Game Prediction Pipeline ─────────────────────────────────────────────

def predict_today_games(
    models: dict | None = None,
    elo: "EloSystem | None" = None,
    odds_df: pd.DataFrame | None = None,
    game_date: str | None = None,
) -> pd.DataFrame:
    """
    Run the full game prediction pipeline for a given date (defaults to today).

    Returns a DataFrame with one row per game containing:
      home_team, away_team, home_win_prob, away_win_prob,
      predicted_spread, confidence, edge, elo_prob, ml_prob
    """
    # Load models and Elo if not supplied
    if models is None:
        models = load_models()
    if elo is None:
        elo_path = MODEL_DIR / "elo_system.pkl"
        elo = EloSystem.load(elo_path) if elo_path.exists() else None

    # Build team features from current season log
    game_log = get_league_game_log_cached(CURRENT_SEASON)
    if game_log.empty:
        return pd.DataFrame()

    game_log["GAME_DATE"] = pd.to_datetime(game_log["GAME_DATE"])
    team_feat: dict[int, pd.DataFrame] = {}
    for tid, grp in game_log.groupby("TEAM_ID"):
        team_feat[int(tid)] = engineer_team_features(grp.sort_values("GAME_DATE"))

    # Get today's games
    games_header, _ = get_today_scoreboard(game_date)
    if games_header.empty:
        return pd.DataFrame()

    # Market implied probabilities
    market_map: dict[str, float] = {}
    if odds_df is not None and not odds_df.empty:
        implied = get_implied_probs(odds_df)
        for _, r in implied.iterrows():
            market_map[r["game_id"]] = float(r["home_prob"])

    # Team name/abbreviation → ID lookup
    teams_df = get_all_teams()
    name_to_id = dict(zip(teams_df["full_name"], teams_df["id"]))
    abbr_to_id = dict(zip(teams_df["abbreviation"], teams_df["id"]))
    id_to_name = dict(zip(teams_df["id"], teams_df["full_name"]))

    def resolve_id(name: str, abbr: str) -> int | None:
        return name_to_id.get(name) or abbr_to_id.get(abbr)

    results = []
    for _, game in games_header.iterrows():
        home_name = game.get("HOME_TEAM_NAME", game.get("HOME_TEAM_ABBREVIATION", ""))
        away_name = game.get("VISITOR_TEAM_NAME", game.get("VISITOR_TEAM_ABBREVIATION", ""))
        home_abbr = game.get("HOME_TEAM_ABBREVIATION", "")
        away_abbr = game.get("VISITOR_TEAM_ABBREVIATION", "")
        game_id   = str(game.get("GAME_ID", ""))

        # Prefer direct team IDs from ScoreboardV3; fall back to name/abbr lookup
        raw_home_id = game.get("HOME_TEAM_ID")
        raw_away_id = game.get("VISITOR_TEAM_ID")
        if raw_home_id and not pd.isna(raw_home_id):
            home_id = int(raw_home_id)
        else:
            home_id = resolve_id(home_name, home_abbr)
        if raw_away_id and not pd.isna(raw_away_id):
            away_id = int(raw_away_id)
        else:
            away_id = resolve_id(away_name, away_abbr)

        if home_id is None or away_id is None:
            continue

        # Resolve display names from static data if not in scoreboard
        if not home_name or home_name == home_abbr:
            home_name = id_to_name.get(home_id, home_abbr)
        if not away_name or away_name == away_abbr:
            away_name = id_to_name.get(away_id, away_abbr)

        # Elo probability
        elo_prob = elo.win_probability(home_id, away_id, True) if elo else 0.5

        # ML model probability
        ml_prob = 0.5
        hf = team_feat.get(home_id)
        af = team_feat.get(away_id)
        if hf is not None and af is not None and models:
            home_row = hf.sort_values("GAME_DATE").iloc[-1]
            away_row = af.sort_values("GAME_DATE").iloc[-1]
            fv = build_game_feature_vector(home_row, away_row)
            avail = [c for c in FEATURE_COLS_GAME if c in fv.index]
            X = pd.DataFrame([fv[avail].fillna(0)])
            try:
                ml_prob = float(ensemble_predict_proba(models, X)[0])
            except Exception:
                ml_prob = elo_prob

        # Blend Elo + ML (0.25 / 0.75 if models trained, 1.0 Elo otherwise)
        final_prob = (0.75 * ml_prob + 0.25 * elo_prob) if models else elo_prob

        market_prob = market_map.get(game_id)
        edge = round(final_prob - market_prob, 3) if market_prob else None

        results.append({
            "game_id":          game_id,
            "home_team":        home_name or home_abbr,
            "away_team":        away_name or away_abbr,
            "home_team_id":     home_id,
            "away_team_id":     away_id,
            "home_win_prob":    round(final_prob, 3),
            "away_win_prob":    round(1.0 - final_prob, 3),
            "predicted_spread": win_prob_to_spread(final_prob),
            "market_home_prob": round(market_prob, 3) if market_prob else None,
            "edge":             edge,
            "confidence":       assign_confidence_tier(final_prob, market_prob),
            "elo_prob":         round(elo_prob, 3),
            "ml_prob":          round(ml_prob, 3),
        })

    return pd.DataFrame(results)


# ── Over/Under (Totals) Prediction ────────────────────────────────────────────

def predict_total_points(
    home_row: pd.Series,
    away_row: pd.Series,
    total_model=None,
    consensus_line: float | None = None,
) -> dict:
    """
    Predict total points for a single game and compute over/under signal.

    Parameters
    ----------
    home_row / away_row : Latest feature row from engineer_team_features()
    total_model         : Fitted XGBRegressor from load_totals_model()
    consensus_line      : Sportsbook consensus total (from sbrscrape)

    Returns dict with:
        predicted_total, consensus_line, direction ('OVER'/'UNDER'/'—'),
        margin (predicted_total − consensus_line), confidence (0-1)
    """
    if total_model is None:
        total_model = load_totals_model()
    if total_model is None:
        return {
            "predicted_total": None,
            "consensus_line": consensus_line,
            "direction": "—",
            "margin": None,
            "confidence": None,
        }

    from utils.feature_engine import build_game_feature_vector
    fv = build_game_feature_vector(home_row, away_row)

    # Add consensus line as a feature (NaN if unavailable)
    fv["total_consensus"] = consensus_line if consensus_line is not None else float("nan")

    avail = [c for c in FEATURE_COLS_TOTALS if c in fv.index]
    X = pd.DataFrame([fv[avail].fillna(0)])

    try:
        predicted_total = float(total_model.predict(X)[0])
    except Exception:
        return {
            "predicted_total": None,
            "consensus_line": consensus_line,
            "direction": "—",
            "margin": None,
            "confidence": None,
        }

    result: dict = {
        "predicted_total": round(predicted_total, 1),
        "consensus_line":  consensus_line,
        "direction":       "—",
        "margin":          None,
        "confidence":      None,
    }

    if consensus_line is not None and consensus_line > 0:
        margin = predicted_total - consensus_line
        result["margin"]    = round(margin, 1)
        result["direction"] = "OVER" if margin > 0 else "UNDER"
        # Confidence proportional to magnitude of margin (3 pts ≈ moderate)
        result["confidence"] = round(min(abs(margin) / 6.0, 1.0), 2)

    return result


# ── Player Prop Predictions ────────────────────────────────────────────────────

def predict_player_prop(
    player_id: int,
    stat_cat: str,
    line: float,
    season: str = CURRENT_SEASON,
    opp_def_rating: float = 110.0,
    pace: float = 100.0,
    is_home: int = 1,
    rest_days: int = 2,
    is_b2b: int = 0,
) -> dict:
    """
    Predict a single player prop (more / less) using a statistical model.

    Uses the player's recent game log to estimate a distribution and computes
    P(stat > line) using a Normal CDF.

    Parameters
    ----------
    stat_cat : one of PTS, REB, AST, 3PM, PRA, STL, BLK, STL+BLK
    line     : the DraftKings prop line
    """
    stat_col = PROP_STAT_MAP.get(stat_cat.upper(), stat_cat)
    game_log = get_player_game_log_cached(player_id, season)
    if game_log.empty:
        return {"error": "No game log data available for this player"}

    from utils.feature_engine import engineer_player_features

    df = engineer_player_features(game_log)
    df = df.sort_values("GAME_DATE")

    if stat_col not in df.columns:
        return {"error": f"Stat '{stat_col}' not found in game log"}

    values = df[stat_col].dropna().values
    if len(values) == 0:
        return {"error": "No valid stat values found"}

    # Weighted recency: last 10 games for mean, full log for std
    recent = values[-10:] if len(values) >= 10 else values
    mu_raw = float(np.mean(recent))
    sigma  = float(np.std(values[-20:] if len(values) >= 20 else values))
    sigma  = max(sigma, 0.5)  # floor prevents zero std

    # Contextual adjustments
    home_factor  = 1.02 if is_home else 0.98
    pace_factor  = 0.90 + 0.10 * (pace / 100.0)   # neutral = 1.0 at pace 100
    b2b_factor   = 0.94 if is_b2b else 1.0
    def_factor   = 1.0 + (110.0 - opp_def_rating) * 0.003  # better opp def → lower output

    mu_adj = mu_raw * home_factor * pace_factor * b2b_factor * def_factor

    # P(over)
    p_over = float(1.0 - stats.norm.cdf(line, loc=mu_adj, scale=sigma))
    p_over = float(np.clip(p_over, 0.01, 0.99))
    p_under = 1.0 - p_over

    confidence = abs(p_over - 0.5) * 2.0  # 0 (coin flip) → 1 (certainty)
    direction  = "MORE" if p_over >= 0.5 else "LESS"

    over_rate_all  = float(np.mean(values > line)) if len(values) > 0 else 0.5
    over_rate_l10  = float(np.mean(recent > line)) if len(recent) > 0 else 0.5

    return {
        "player_id":       player_id,
        "stat":            stat_cat,
        "line":            line,
        "predicted_value": round(mu_adj, 1),
        "over_probability":  round(p_over, 3),
        "under_probability": round(p_under, 3),
        "confidence":      round(confidence, 3),
        "direction":       direction,
        "season_avg":      round(float(np.mean(values)), 1),
        "last_5_avg":      round(float(np.mean(values[-5:])), 1) if len(values) >= 5 else None,
        "last_10_avg":     round(float(np.mean(recent)), 1),
        "over_rate_season": round(over_rate_all, 3),
        "over_rate_l10":   round(over_rate_l10, 3),
        "std":             round(sigma, 2),
    }


def confidence_label(confidence: float) -> str:
    """Human-readable confidence tier for a prop."""
    if confidence > 0.40:
        return "Strong"
    if confidence > 0.25:
        return "Moderate"
    if confidence > 0.10:
        return "Weak"
    return "Skip"


# ── Pick 6 Entry Builder ───────────────────────────────────────────────────────

def build_pick6_entry(
    props: list[dict],
    n_picks: int = 5,
    risk: str = "balanced",
) -> dict:
    """
    Select the optimal Pick 6 entry from a list of prop predictions.

    Each prop dict must contain:
      player_id, player_name, stat, line, over_probability,
      confidence, direction, game_id (optional)

    Parameters
    ----------
    risk : 'conservative' → max 3 picks | 'balanced' → 4-5 | 'aggressive' → 6
    """
    if risk == "conservative":
        n_picks = min(n_picks, 3)
    elif risk == "aggressive":
        n_picks = 6

    # Sort by confidence (highest first)
    ranked = sorted(props, key=lambda p: p.get("confidence", 0), reverse=True)

    selected = []
    seen_players: set[int] = set()
    for prop in ranked:
        if len(selected) >= n_picks:
            break
        pid = prop.get("player_id")
        if pid in seen_players:
            continue
        seen_players.add(pid)
        selected.append(prop)

    if not selected:
        return {"picks": [], "combined_probability": 0.0, "expected_value": -1.0, "warnings": []}

    # Probability that each pick is correct
    pick_probs = [
        p["over_probability"] if p["direction"] == "MORE" else p["under_probability"]
        for p in selected
    ]
    combined_prob = float(np.prod(pick_probs))
    payout  = PAYOUT_TABLE.get(len(selected), 1)
    ev      = round(combined_prob * payout - 1.0, 3)

    # Correlation warnings: ≥2 picks from the same game
    game_counts: dict[str, int] = {}
    for p in selected:
        gid = str(p.get("game_id", "unknown"))
        game_counts[gid] = game_counts.get(gid, 0) + 1
    warnings = [
        f"⚠️ {cnt} picks from the same game (correlated risk)"
        for gid, cnt in game_counts.items()
        if cnt >= 2 and gid != "unknown"
    ]

    return {
        "picks":                selected,
        "n_picks":              len(selected),
        "combined_probability": round(combined_prob, 4),
        "payout_multiplier":    payout,
        "expected_value":       ev,
        "warnings":             warnings,
        "risk_profile":         risk,
    }
