"""
Model training script.

Run this after scripts/fetch_historical.py to train game outcome models
on the last 9 seasons of NBA data (2017-18 through 2025-26).

Usage:
    python scripts/train_models.py

Output:
    models/logistic_game_latest.pkl        — base Logistic Regression
    models/xgboost_game_latest.pkl         — base XGBoost
    models/lightgbm_game_latest.pkl        — base LightGBM
    models/random_forest_game_latest.pkl   — base Random Forest
    models/logistic_game_cal_latest.pkl    — isotonic-calibrated Logistic
    models/xgboost_game_cal_latest.pkl     — isotonic-calibrated XGBoost
    models/lightgbm_game_cal_latest.pkl    — isotonic-calibrated LightGBM
    models/random_forest_game_cal_latest.pkl
    models/totals_latest.pkl               — XGBoost total-points regression
    models/elo_system.pkl                  — Elo ratings
    models/eval_metrics.json               — accuracy / calibration / CV stats
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
    # Calibration
    calibrate_models,
    save_calibrated_models,
    # Totals model
    train_totals_model,
    save_totals_model,
    evaluate_totals_model,
    FEATURE_COLS_TOTALS,
    # Walk-forward CV
    walk_forward_eval,
)


def log(msg: str):
    print(f"[train_models] {msg}", flush=True)


# ── Step 1: Load data ──────────────────────────────────────────────────────────

def load_all_seasons() -> pd.DataFrame:
    """Load and concatenate all 9 league game logs (2017-18 → 2025-26)."""
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

def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Temporal train / calibration / test split.

    - Train  (≈80%): everything before 2023-24 season
    - Cal    (≈10%): 2023-24 season (used to fit isotonic calibration)
    - Test   (≈10%): 2024-25 season (held-out evaluation)

    Returns (base_models, calibrated_models, metrics).
    """
    TRAIN_CUTOFF = pd.Timestamp("2023-10-01")
    CAL_CUTOFF   = pd.Timestamp("2024-10-01")

    train_df = df[df["GAME_DATE"] < TRAIN_CUTOFF]
    cal_df   = df[(df["GAME_DATE"] >= TRAIN_CUTOFF) & (df["GAME_DATE"] < CAL_CUTOFF)]
    test_df  = df[df["GAME_DATE"] >= CAL_CUTOFF]

    log(f"Train: {len(train_df):,} games | Cal: {len(cal_df):,} games | Test: {len(test_df):,} games")
    if not train_df.empty:
        log(f"  Training window: {train_df['GAME_DATE'].min().date()} → {train_df['GAME_DATE'].max().date()}")
    if not test_df.empty:
        log(f"  Test  window:    {test_df['GAME_DATE'].min().date()} → {test_df['GAME_DATE'].max().date()}")

    X_train, feature_cols = get_model_features(train_df)
    y_train = train_df["TARGET"]

    log("\nTraining base models...")
    base_models = train_ensemble(X_train, y_train)

    # ── Calibration ───────────────────────────────────────────────────────────
    if not cal_df.empty:
        log("\nCalibrating models (isotonic regression on 2023-24 hold-out)...")
        X_cal, _ = get_model_features(cal_df, feature_cols)
        y_cal    = cal_df["TARGET"]
        cal_models = calibrate_models(base_models, X_cal, y_cal)
        log("  ✓ Calibration complete.")
    else:
        log("  ⚠ No calibration data — using uncalibrated models.")
        cal_models = base_models

    # ── Test evaluation ───────────────────────────────────────────────────────
    metrics: dict = {}
    if not test_df.empty:
        X_test, _ = get_model_features(test_df, feature_cols)
        y_test = test_df["TARGET"]

        # Evaluate base ensemble
        base_probs = ensemble_predict_proba(base_models, X_test)
        base_metrics = evaluate_model(y_test, base_probs)
        log(f"\n  Base model test results:")
        log(f"    Accuracy:    {base_metrics['accuracy']:.1%}")
        log(f"    Log Loss:    {base_metrics['log_loss']:.4f}")
        log(f"    Brier Score: {base_metrics['brier_score']:.4f}")

        # Evaluate calibrated ensemble
        cal_probs = ensemble_predict_proba(cal_models, X_test)
        cal_metrics = evaluate_model(y_test, cal_probs)
        log(f"\n  Calibrated model test results:")
        log(f"    Accuracy:    {cal_metrics['accuracy']:.1%}")
        log(f"    Log Loss:    {cal_metrics['log_loss']:.4f}  (lower = better calibration)")
        log(f"    Brier Score: {cal_metrics['brier_score']:.4f}")

        # Accuracy by confidence tier (calibrated)
        test_df = test_df.copy()
        test_df["prob"] = cal_probs
        test_df["conf"] = test_df["prob"].apply(
            lambda p: "High" if p >= 0.65 or p <= 0.35 else
                      "Medium" if p >= 0.57 or p <= 0.43 else "Low"
        )
        for tier in ["High", "Medium", "Low"]:
            sub = test_df[test_df["conf"] == tier]
            if not sub.empty:
                acc = float(np.mean((sub["prob"] >= 0.5).astype(int) == sub["TARGET"]))
                log(f"    {tier:6s} confidence ({len(sub):4d} games): {acc:.1%}")

        metrics = {
            **cal_metrics,
            "base_accuracy":    base_metrics["accuracy"],
            "base_log_loss":    base_metrics["log_loss"],
            "base_brier_score": base_metrics["brier_score"],
            "eval_date":    datetime.now().isoformat(),
            "train_games":  int(len(train_df)),
            "cal_games":    int(len(cal_df)),
            "test_games":   int(len(test_df)),
            "feature_cols": feature_cols,
        }
    else:
        metrics = {"note": "No test data available yet"}

    return base_models, cal_models, metrics


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

    # 4. Walk-forward CV (5-fold TimeSeriesSplit on full feature dataset)
    log("\nRunning walk-forward cross-validation (5 folds)...")
    try:
        cv_df = walk_forward_eval(df, n_splits=5)
        if not cv_df.empty:
            cv_acc  = float(cv_df["accuracy"].mean())
            cv_ll   = float(cv_df["log_loss"].mean())
            cv_bs   = float(cv_df["brier_score"].mean())
            log(f"  CV mean accuracy:    {cv_acc:.1%}")
            log(f"  CV mean log loss:    {cv_ll:.4f}")
            log(f"  CV mean Brier score: {cv_bs:.4f}")
            cv_results = {
                "cv_mean_accuracy":    round(cv_acc, 4),
                "cv_mean_log_loss":    round(cv_ll, 4),
                "cv_mean_brier_score": round(cv_bs, 4),
                "cv_folds":            cv_df.to_dict(orient="records"),
            }
        else:
            cv_results = {}
    except Exception as e:
        log(f"  ⚠ Walk-forward CV failed: {e}")
        cv_results = {}

    # 5. Train + Evaluate (with calibration)
    base_models, cal_models, metrics = train_and_evaluate(df)
    if cv_results:
        metrics.update(cv_results)

    # 6. Train Over/Under totals regression model
    log("\nTraining Over/Under totals regression model...")
    try:
        # Add TOTAL_PTS if not present
        if "TOTAL_PTS" not in df.columns and "HOME_PTS" in df.columns and "AWAY_PTS" in df.columns:
            df["TOTAL_PTS"] = df["HOME_PTS"] + df["AWAY_PTS"]
        # Fill optional total_consensus column with NaN if missing
        if "total_consensus" not in df.columns:
            df["total_consensus"] = np.nan

        totals_train = df[df["GAME_DATE"] < pd.Timestamp("2024-10-01")]
        totals_test  = df[df["GAME_DATE"] >= pd.Timestamp("2024-10-01")]
        totals_model = train_totals_model(totals_train)
        if not totals_test.empty:
            tm = evaluate_totals_model(totals_model, totals_test)
            log(f"  Totals MAE: {tm.get('totals_mae', '?'):.2f} pts")
            if tm.get("totals_dir_accuracy") is not None:
                log(f"  Directional accuracy vs line: {tm['totals_dir_accuracy']:.1%}")
            metrics.update(tm)
        totals_path = save_totals_model(totals_model)
        log(f"  ✓ Totals model → {totals_path}")
    except Exception as e:
        log(f"  ⚠ Totals model training failed: {e}")

    # 7. Save base models
    log("\nSaving models...")
    paths = save_models(base_models, suffix="latest")
    for name, path in paths.items():
        log(f"  ✓ {name} → {path}")

    # 8. Save calibrated models
    cal_paths = save_calibrated_models(cal_models, suffix="latest")
    for name, path in cal_paths.items():
        log(f"  ✓ {name} (calibrated) → {path}")

    # 9. Save metrics
    metrics_path = MODEL_DIR / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "feature_cols"}, f, indent=2)
    log(f"  ✓ Metrics → {metrics_path}")

    elapsed = time.time() - start
    log(f"\n✓ Training complete in {elapsed / 60:.1f} minutes.")
    log("Run 'streamlit run predictions.py' to launch the app.")


if __name__ == "__main__":
    main()
