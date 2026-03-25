"""
ML model definitions, Elo rating system, training helpers, and persistence.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Feature columns used for game outcome models
FEATURE_COLS_GAME = [
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

# Extended feature set — requires scraped external data + retraining.
# Activate with: python scripts/train_models.py --extended-features
FEATURE_COLS_GAME_EXTENDED = FEATURE_COLS_GAME + [
    "home_REST_W_PCT",     "away_REST_W_PCT",
    "home_NBS_SAR",        "away_NBS_SAR",
    "home_NBS_eDIFF",      "away_NBS_eDIFF",
    "home_NBS_CONS",       "away_NBS_CONS",
    "rest_w_pct_diff",
    "sar_diff",
    "efficiency_diff_ext",
]

# Odds-aware feature set — requires historical odds parquet + retraining.
# Activate with: python scripts/train_models.py --odds-features
FEATURE_COLS_GAME_ODDS = FEATURE_COLS_GAME_EXTENDED + [
    "implied_prob_home",
    "implied_prob_away",
    "spread_consensus",
    "total_consensus",
    "odds_disagreement_ml",
    "odds_disagreement_total",
]


# ── Elo Rating System ──────────────────────────────────────────────────────────

class EloSystem:
    """
    NBA Elo rating system with home-court advantage and margin-of-victory scaling.

    Inspired by FiveThirtyEight's NBA Elo model.
    """

    def __init__(
        self,
        k: float = 20.0,
        home_advantage: float = 100.0,
        mov_scale: bool = True,
        season_carryover: float = 0.75,
        initial_rating: float = 1500.0,
    ):
        self.k = k
        self.home_advantage = home_advantage
        self.mov_scale = mov_scale
        self.season_carryover = season_carryover
        self.initial_rating = initial_rating
        self.ratings: dict[int, float] = {}

    def get_rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.initial_rating)

    def win_probability(self, team_a_id: int, team_b_id: int, a_is_home: bool = True) -> float:
        """Probability that team A beats team B."""
        ra = self.get_rating(team_a_id) + (self.home_advantage if a_is_home else 0)
        rb = self.get_rating(team_b_id) + (self.home_advantage if not a_is_home else 0)
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def _mov_multiplier(self, margin: float, winner_elo_diff: float) -> float:
        """FiveThirtyEight margin-of-victory multiplier."""
        return np.log(abs(margin) + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))

    def update(
        self,
        home_team_id: int,
        away_team_id: int,
        home_score: int,
        away_score: int,
    ):
        """Update ratings after a completed game."""
        p_home = self.win_probability(home_team_id, away_team_id, a_is_home=True)
        home_win = int(home_score > away_score)
        margin = abs(home_score - away_score)

        k = self.k
        if self.mov_scale and margin > 0:
            elo_diff = self.get_rating(home_team_id) - self.get_rating(away_team_id)
            winner_diff = elo_diff if home_win else -elo_diff
            k *= self._mov_multiplier(margin, winner_diff)

        delta = k * (home_win - p_home)
        self.ratings[home_team_id] = self.get_rating(home_team_id) + delta
        self.ratings[away_team_id] = self.get_rating(away_team_id) - delta

    def new_season(self):
        """Regress all ratings toward the mean at season start."""
        for tid in self.ratings:
            self.ratings[tid] = (
                self.ratings[tid] * self.season_carryover
                + self.initial_rating * (1.0 - self.season_carryover)
            )

    def fit(self, games_df: pd.DataFrame) -> "EloSystem":
        """
        Fit Elo ratings on historical game data.

        Parameters
        ----------
        games_df : DataFrame with columns:
            GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID, HOME_PTS, AWAY_PTS
            Optional: SEASON (to trigger new_season() regression between seasons)
        """
        df = games_df.sort_values("GAME_DATE").copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        prev_season = None

        for _, row in df.iterrows():
            season = str(row.get("SEASON", ""))
            if season and prev_season and season != prev_season:
                self.new_season()
            prev_season = season
            self.update(
                int(row["HOME_TEAM_ID"]),
                int(row["AWAY_TEAM_ID"]),
                int(row["HOME_PTS"]),
                int(row["AWAY_PTS"]),
            )
        return self

    def get_all_ratings(self) -> pd.DataFrame:
        """Return current ratings as a DataFrame sorted descending."""
        return (
            pd.DataFrame.from_dict(
                {"team_id": list(self.ratings.keys()), "elo": list(self.ratings.values())}
            )
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or MODEL_DIR / "elo_system.pkl")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EloSystem":
        path = Path(path or MODEL_DIR / "elo_system.pkl")
        return joblib.load(path)


# ── Feature extraction ─────────────────────────────────────────────────────────

def get_model_features(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Extract feature matrix X from training DataFrame. Returns (X, cols_used)."""
    cols = feature_cols or FEATURE_COLS_GAME
    available = [c for c in cols if c in df.columns]
    X = df[available].fillna(0).astype(float)
    return X, available


# ── Model builders ─────────────────────────────────────────────────────────────

def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_ensemble(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train all four base models. Returns {name: fitted_model}."""
    models = {}
    print("  Training Logistic Regression...", flush=True)
    models["logistic"] = train_logistic_regression(X, y)
    print("  Training XGBoost...", flush=True)
    models["xgboost"] = train_xgboost(X, y)
    print("  Training LightGBM...", flush=True)
    models["lightgbm"] = train_lightgbm(X, y)
    print("  Training Random Forest...", flush=True)
    models["random_forest"] = train_random_forest(X, y)
    return models


# ── Ensemble prediction ────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "logistic":      0.15,
    "xgboost":       0.35,
    "lightgbm":      0.35,
    "random_forest": 0.15,
}


def ensemble_predict_proba(
    models: dict,
    X: pd.DataFrame,
    weights: dict | None = None,
) -> np.ndarray:
    """Weighted average of base model home-win probabilities."""
    w = weights or DEFAULT_WEIGHTS
    probs = np.zeros(len(X))
    total = 0.0
    for name, model in models.items():
        wt = w.get(name, 0.25)
        try:
            p = model.predict_proba(X)[:, 1]
            probs += wt * p
            total += wt
        except Exception:
            pass
    return probs / total if total > 0 else probs


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Return accuracy, log-loss, and Brier score."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "log_loss":    round(float(log_loss(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def walk_forward_eval(
    df: pd.DataFrame,
    n_splits: int = 5,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    TimeSeriesSplit walk-forward evaluation.
    Returns a DataFrame with per-fold accuracy, log_loss, brier_score.
    """
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    X, cols = get_model_features(df, feature_cols)
    y = df["TARGET"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if len(y_te) == 0:
            continue
        models = train_ensemble(X_tr, y_tr)
        probs = ensemble_predict_proba(models, X_te)
        metrics = evaluate_model(y_te, probs)
        metrics["fold"] = fold + 1
        metrics["n_test"] = len(y_te)
        records.append(metrics)

    return pd.DataFrame(records)


# ── Feature importance ─────────────────────────────────────────────────────────

def get_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """
    Extract feature importances from XGBoost, LightGBM, or RandomForest.
    Returns DataFrame(feature, importance) sorted descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        # Pipeline (Logistic Regression)
        clf = model.named_steps.get("clf")
        if hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return pd.DataFrame(columns=["feature", "importance"])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(importances), len(feature_cols))
    return (
        pd.DataFrame({"feature": feature_cols[:n], "importance": importances[:n]})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def save_models(models: dict, suffix: str = "latest") -> dict:
    """Save each model to models/{name}_game_{suffix}.pkl. Returns paths."""
    saved = {}
    for name, model in models.items():
        path = MODEL_DIR / f"{name}_game_{suffix}.pkl"
        joblib.dump(model, path)
        saved[name] = str(path)
    return saved


def load_models(suffix: str = "latest") -> dict:
    """Load all saved base models. Returns {name: model}."""
    models = {}
    for name in ["logistic", "xgboost", "lightgbm", "random_forest"]:
        path = MODEL_DIR / f"{name}_game_{suffix}.pkl"
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models


def load_eval_metrics() -> dict:
    """Load the most recent evaluation metrics JSON."""
    path = MODEL_DIR / "eval_metrics.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}
