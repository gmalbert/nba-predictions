"""
Model Performance page.

Shows model accuracy, calibration, backtesting results, and feature
importance to provide transparency into prediction quality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from utils.data_fetcher import (
    get_league_game_log_cached,
    CURRENT_SEASON,
    HISTORICAL_SEASONS,
)
from utils.feature_engine import build_training_dataset
from utils.model_utils import (
    load_models,
    load_eval_metrics,
    EloSystem,
    get_model_features,
    ensemble_predict_proba,
    evaluate_model,
    get_feature_importance,
    MODEL_DIR,
    FEATURE_COLS_GAME,
)
from footer import add_betting_oracle_footer


NBA_BLUE = "#1D428A"
NBA_RED  = "#C8102E"
GREEN    = "#16a34a"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("data_files/logo.png", width=120)
    st.markdown("---")
    model_choice = st.selectbox("Model", ["Ensemble", "XGBoost", "LightGBM", "Logistic Regression", "Random Forest"])
    season_eval  = st.selectbox("Evaluation Season", HISTORICAL_SEASONS[::-1], index=0)
    st.markdown("---")
    st.info("Models are retrained via `scripts/train_models.py`. Last eval metrics are shown below.")


# ── Main ───────────────────────────────────────────────────────────────────────

st.title("📈 Model Performance")
st.caption("Accuracy, calibration, and feature importance for the game outcome models.")

# Load metrics
metrics = load_eval_metrics()
models  = load_models()
elo_path = MODEL_DIR / "elo_system.pkl"

if not metrics and not models:
    st.warning(
        "No trained models found. Run `python scripts/fetch_historical.py` "
        "then `python scripts/train_models.py` to generate predictions.",
        icon="⚠️",
    )
    st.stop()

# ── Accuracy dashboard ─────────────────────────────────────────────────────────

st.subheader("📊 Summary Metrics")
if metrics:
    eval_date = metrics.get("eval_date", "Unknown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",    f"{metrics.get('accuracy', 0):.1%}")
    m2.metric("Log Loss",    f"{metrics.get('log_loss', 0):.4f}")
    m3.metric("Brier Score", f"{metrics.get('brier_score', 0):.4f}")
    m4.metric("Train Games", f"{metrics.get('train_games', 0):,}")
    st.caption(f"Last evaluated: {eval_date}")
else:
    st.info("No evaluation metrics found. Train models to see results.")

st.markdown("---")

# ── Calibration plot ───────────────────────────────────────────────────────────

st.subheader("🎯 Model Calibration")
st.caption("A well-calibrated model's predicted probabilities match actual win rates.")

game_log = get_league_game_log_cached(season_eval)
if game_log.empty:
    st.info(f"No game log for {season_eval}. Run fetch_historical.py to download data.")
else:
    with st.spinner("Computing calibration..."):
        try:
            train_df = build_training_dataset(game_log)
            if not train_df.empty and models:
                X, _ = get_model_features(train_df)
                y    = train_df["TARGET"]
                probs = ensemble_predict_proba(models, X)

                # Bin into deciles
                bins    = np.linspace(0, 1, 11)
                bin_idx = np.digitize(probs, bins) - 1
                cal_rows = []
                for b in range(10):
                    mask = bin_idx == b
                    if mask.sum() >= 5:
                        cal_rows.append({
                            "Predicted Prob (mid)": (bins[b] + bins[b+1]) / 2,
                            "Actual Win Rate":       float(y[mask].mean()),
                            "Count": int(mask.sum()),
                        })
                cal_df = pd.DataFrame(cal_rows)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", line=dict(dash="dash", color="gray", width=1),
                    name="Perfect calibration",
                ))
                fig.add_trace(go.Scatter(
                    x=cal_df["Predicted Prob (mid)"],
                    y=cal_df["Actual Win Rate"],
                    mode="lines+markers",
                    marker=dict(size=cal_df["Count"] / cal_df["Count"].max() * 20 + 5,
                                color=NBA_BLUE),
                    line=dict(color=NBA_BLUE, width=2),
                    name="Ensemble model",
                    text=cal_df["Count"].apply(lambda c: f"{c} games"),
                    hovertemplate="Pred: %{x:.0%}<br>Actual: %{y:.0%}<br>%{text}",
                ))
                fig.update_layout(
                    xaxis=dict(title="Predicted Probability", tickformat=".0%", range=[0, 1]),
                    yaxis=dict(title="Actual Win Rate", tickformat=".0%", range=[0, 1]),
                    height=400, margin=dict(l=10, r=10, t=30, b=30),
                    legend=dict(x=0.05, y=0.95),
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Could not compute calibration: {e}")

st.markdown("---")

# ── Feature Importance ─────────────────────────────────────────────────────────

st.subheader("🔬 Feature Importance")
model_name_map = {
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic",
}

if model_choice != "Ensemble" and models:
    key = model_name_map.get(model_choice)
    model = models.get(key)
    if model:
        if not game_log.empty:
            try:
                train_df = build_training_dataset(game_log)
                _, feature_cols = get_model_features(train_df)
            except Exception:
                feature_cols = FEATURE_COLS_GAME
        else:
            feature_cols = FEATURE_COLS_GAME

        fi_df = get_feature_importance(model, feature_cols)
        if not fi_df.empty:
            top_n = min(20, len(fi_df))
            fig_fi = px.bar(
                fi_df.head(top_n),
                x="importance", y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#e0e7ff", NBA_BLUE],
                height=max(300, top_n * 22),
            )
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=30, b=10),
                coloraxis_showscale=False,
            )
            fig_fi.update_traces(marker_line_width=0)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")
    else:
        st.info(f"{model_choice} model not found. Train models first.")
elif model_choice == "Ensemble" and models:
    # Average importance across tree-based models
    all_fi = []
    for k in ["xgboost", "lightgbm", "random_forest"]:
        m = models.get(k)
        if m:
            fi = get_feature_importance(m, FEATURE_COLS_GAME)
            if not fi.empty:
                fi = fi.rename(columns={"importance": k})
                all_fi.append(fi.set_index("feature"))
    if all_fi:
        merged = pd.concat(all_fi, axis=1).fillna(0)
        merged["avg_importance"] = merged.mean(axis=1)
        merged = merged.sort_values("avg_importance", ascending=False).reset_index()
        top_n = min(20, len(merged))
        fig_ens = px.bar(
            merged.head(top_n),
            x="avg_importance", y="feature", orientation="h",
            color="avg_importance",
            color_continuous_scale=["#e0e7ff", NBA_BLUE],
            height=max(300, top_n * 22),
        )
        fig_ens.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_ens, use_container_width=True)

st.markdown("---")

# ── Backtesting results ────────────────────────────────────────────────────────

st.subheader("🕐 Season-by-Season Backtest")
st.caption("Ensemble model accuracy evaluated on each season independently.")

if models and len(HISTORICAL_SEASONS) > 1:
    backtest_rows = []
    backtest_cols = st.columns(min(5, len(HISTORICAL_SEASONS)))

    for i, season in enumerate(HISTORICAL_SEASONS):
        with backtest_cols[i % len(backtest_cols)]:
            with st.spinner(f"{season}..."):
                try:
                    slog = get_league_game_log_cached(season)
                    if not slog.empty:
                        sdf = build_training_dataset(slog)
                        if not sdf.empty:
                            X_s, _ = get_model_features(sdf)
                            y_s    = sdf["TARGET"]
                            p_s    = ensemble_predict_proba(models, X_s)
                            m_s    = evaluate_model(y_s, p_s)
                            st.metric(
                                label=season,
                                value=f"{m_s['accuracy']:.1%}",
                                delta=f"LL {m_s['log_loss']:.3f}",
                            )
                            backtest_rows.append({"Season": season, **m_s})
                except Exception:
                    pass

    if backtest_rows:
        bt_df = pd.DataFrame(backtest_rows)
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_df["Season"], y=bt_df["accuracy"],
            mode="lines+markers",
            line=dict(color=NBA_BLUE, width=2),
            marker=dict(size=8),
            name="Accuracy",
        ))
        fig_bt.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Coin flip (50%)")
        fig_bt.update_layout(
            xaxis_title="Season", yaxis_title="Accuracy",
            yaxis=dict(tickformat=".0%", range=[0.45, 0.80]),
            height=300, margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_bt, use_container_width=True)
else:
    st.info("Train models to see backtest results.")

add_betting_oracle_footer()
