"""
Model training script.

Run this after scripts/fetch_historical.py to train game outcome models
on the last 5 seasons of NBA data.

Usage:
    python scripts/train_models.py

Output:
    models/logistic_game_latest.pkl
    models/xgboost_game_latest.pkl
    models/lightgbm_game_latest.pkl
    models/random_forest_game_latest.pkl
    models/elo_system.pkl
    models/eval_metrics.json
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.data_fetcher import get_league_game_log, HISTORICAL_SEASONS
from utils.feature_engine import build_training_dataset
from utils.model_utils import (
    train_ensemble,
    EloSystem,
    evaluate_model,
    ensemble_predict_proba,
    get_model_features,
    save_models,
    MODEL_DIR,
    FEATURE_COLS_GAME,
)


def log(msg: str):
    print(f"[train_models] {msg}", flush=True)


# ── Step 1: Load data ──────────────────────────────────────────────────────────

def load_all_seasons() -> pd.DataFrame:
    """Load and concatenate all 5 league game logs."""
    dfs = []
    for season in HISTORICAL_SEASONS:
        log(f"Loading {season}...")
        df = get_league_game_log(season)
        if not df.empty:
            df["SEASON"] = season
            dfs.append(df)
        else:
            log(f"  WARNING: No data for {season}. Run scripts/fetch_historical.py first.")
    if not dfs:
        raise RuntimeError("No game log data found. Run scripts/fetch_historical.py first.")
    combined = pd.concat(dfs, ignore_index=True)
    log(f"Total games loaded: {len(combined):,} rows across {len(dfs)} seasons.\n")
    return combined


# ── Step 2: Train Elo ──────────────────────────────────────────────────────────

def train_elo(raw: pd.DataFrame) -> EloSystem:
    """Build Elo ratings by replaying all historical games in chronological order."""
    log("Training Elo rating system...")
    raw = raw.copy()
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw["IS_HOME"] = raw["MATCHUP"].apply(lambda m: 1 if "vs." in str(m) else 0)

    home_games = raw[raw["IS_HOME"] == 1].copy()
    away_games = raw[raw["IS_HOME"] == 0].copy()

    paired = home_games.merge(
        away_games[["GAME_ID", "TEAM_ID", "PTS"]].rename(
            columns={"TEAM_ID": "AWAY_TEAM_ID", "PTS": "AWAY_PTS"}
        ),
        on="GAME_ID",
        how="inner",
    ).rename(columns={"TEAM_ID": "HOME_TEAM_ID", "PTS": "HOME_PTS"})

    elo = EloSystem(k=20.0, home_advantage=100.0, mov_scale=True, season_carryover=0.75)
    elo.fit(paired[["GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_PTS", "AWAY_PTS", "SEASON"]])

    path = elo.save()
    log(f"  ✓ Elo saved → {path}")
    log(f"  Top 5 teams by Elo:")
    top = elo.get_all_ratings().head(5)
    for _, r in top.iterrows():
        log(f"    Team {int(r['team_id'])}: {r['elo']:.0f}")
    return elo


# ── Step 3: Build feature dataset ─────────────────────────────────────────────

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    log("Building feature dataset (this may take 1-2 minutes)...")
    df = build_training_dataset(raw)
    df = df.dropna(subset=["TARGET"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    log(f"  ✓ Feature dataset: {len(df):,} games")
    return df


# ── Step 4: Train & evaluate ───────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Train-test split on temporal boundary (2024-25 season as held-out test).
    Returns (models, metrics).
    """
    CUTOFF = pd.Timestamp("2024-10-01")
    train_df = df[df["GAME_DATE"] < CUTOFF]
    test_df  = df[df["GAME_DATE"] >= CUTOFF]

    log(f"Train: {len(train_df):,} games | Test: {len(test_df):,} games")
    log(f"  Training window: {train_df['GAME_DATE'].min().date()} → {train_df['GAME_DATE'].max().date()}")
    log(f"  Test  window:    {test_df['GAME_DATE'].min().date()} → {test_df['GAME_DATE'].max().date()}")

    X_train, feature_cols = get_model_features(train_df)
    y_train = train_df["TARGET"]

    log("\nTraining models...")
    models = train_ensemble(X_train, y_train)

    # Evaluate on held-out test set
    if not test_df.empty:
        X_test, _ = get_model_features(test_df, feature_cols)
        y_test = test_df["TARGET"]
        probs  = ensemble_predict_proba(models, X_test)
        metrics = evaluate_model(y_test, probs)
        log(f"\n  Test set results:")
        log(f"    Accuracy:    {metrics['accuracy']:.1%}")
        log(f"    Log Loss:    {metrics['log_loss']:.4f}")
        log(f"    Brier Score: {metrics['brier_score']:.4f}")

        # Accuracy by confidence tier
        test_df = test_df.copy()
        test_df["prob"] = probs
        test_df["conf"] = test_df["prob"].apply(
            lambda p: "High" if p >= 0.65 or p <= 0.35 else
                      "Medium" if p >= 0.57 or p <= 0.43 else "Low"
        )
        for tier in ["High", "Medium", "Low"]:
            sub = test_df[test_df["conf"] == tier]
            if not sub.empty:
                acc = float(np.mean((sub["prob"] >= 0.5).astype(int) == sub["TARGET"]))
                log(f"    {tier:6s} confidence ({len(sub):4d} games): {acc:.1%}")

        metrics["eval_date"]   = datetime.now().isoformat()
        metrics["train_games"] = int(len(train_df))
        metrics["test_games"]  = int(len(test_df))
        metrics["feature_cols"] = feature_cols
    else:
        metrics = {"note": "No test data available yet"}

    return models, metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("NBA Predictions — Model Training Pipeline")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60 + "\n")

    start = time.time()

    # 1. Load
    raw = load_all_seasons()

    # 2. Elo
    elo = train_elo(raw)

    # 3. Features
    df = build_features(raw)

    # 4. Train + Evaluate
    models, metrics = train_and_evaluate(df)

    # 5. Save models
    log("\nSaving models...")
    paths = save_models(models, suffix="latest")
    for name, path in paths.items():
        log(f"  ✓ {name} → {path}")

    # 6. Save metrics
    metrics_path = MODEL_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "feature_cols"}, f, indent=2)
    log(f"  ✓ Metrics → {metrics_path}")

    elapsed = time.time() - start
    log(f"\n✓ Training complete in {elapsed / 60:.1f} minutes.")
    log("Run 'streamlit run predictions.py' to launch the app.")


if __name__ == "__main__":
    main()
