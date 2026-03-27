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
    CURRENT_SEASON,
    HISTORICAL_SEASONS,
)
from utils.feature_engine import build_training_dataset, get_training_dataset
from utils.model_utils import (
    load_models,
    load_calibrated_models,
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
    st.image("data_files/logo.png", width=200)
    st.markdown("---")
    model_choice = st.selectbox("Model", ["Ensemble", "XGBoost", "LightGBM", "Logistic Regression", "Random Forest"])
    season_eval  = st.selectbox("Evaluation Season", HISTORICAL_SEASONS[::-1], index=0)
    st.markdown("---")


# ── Main ───────────────────────────────────────────────────────────────────────

st.title("📈 Model Performance")
st.caption("Accuracy, calibration, and feature importance for the game outcome models.")

# Load metrics
metrics = load_eval_metrics()
models  = load_models()
elo_path = MODEL_DIR / "elo_system.pkl"

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

try:
    train_df = get_training_dataset(season_eval)
    if train_df.empty:
        st.info(f"No data for {season_eval}. Run fetch_historical.py to download data.")
    elif models:
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
        st.plotly_chart(fig, width='stretch')
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
        try:
            train_df = get_training_dataset(season_eval)
            _, feature_cols = get_model_features(train_df) if not train_df.empty else (None, FEATURE_COLS_GAME)
        except Exception:
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
            st.plotly_chart(fig_fi, width='stretch')
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
        st.plotly_chart(fig_ens, width='stretch')

st.markdown("---")

# ── Backtesting results ────────────────────────────────────────────────────────

st.subheader("🕐 Season-by-Season Backtest")
st.caption("Ensemble model accuracy evaluated on each season independently.")

if models and len(HISTORICAL_SEASONS) > 1:
    backtest_rows = []
    backtest_cols = st.columns(min(5, len(HISTORICAL_SEASONS)))

    for i, season in enumerate(HISTORICAL_SEASONS):
        with backtest_cols[i % len(backtest_cols)]:
            try:
                sdf = get_training_dataset(season)
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
        st.plotly_chart(fig_bt, width='stretch')
else:
    st.info("Train models to see backtest results.")

st.markdown("---")

# ── ROI by Confidence Tier ─────────────────────────────────────────────────────

st.subheader("💰 Simulated ROI by Confidence Tier")
st.caption(
    "Flat-bet $100 on every home-team prediction within each tier on the evaluation season. "
    "No real sportsbook lines are used — assumes −110 odds (standard juice) as a worst-case baseline."
)

if models:
    try:
        train_df_roi = get_training_dataset(season_eval)
        if not train_df_roi.empty:
            cal_roi_models = load_calibrated_models()
            _used_models = cal_roi_models if cal_roi_models else models
            X_roi, _ = get_model_features(train_df_roi)
            probs_roi = ensemble_predict_proba(_used_models, X_roi)

            train_df_roi = train_df_roi.copy()
            train_df_roi["prob"]   = probs_roi
            train_df_roi["pred"]   = (probs_roi >= 0.5).astype(int)
            train_df_roi["correct"]= (train_df_roi["pred"] == train_df_roi["TARGET"]).astype(int)
            train_df_roi["conf"]   = train_df_roi["prob"].apply(
                lambda p: "High"   if p >= 0.65 or p <= 0.35 else
                          "Medium" if p >= 0.57 or p <= 0.43 else "Low"
            )

            # Flat-bet at −110 (pay $110 to win $100)
            def _flat_bet_roi(sub: pd.DataFrame) -> float:
                n = len(sub)
                if n == 0:
                    return 0.0
                wins = sub["correct"].sum()
                pnl  = wins * 100.0 - (n - wins) * 110.0
                return round(pnl / (n * 110.0) * 100, 1)   # ROI %

            roi_rows = []
            for tier in ["High", "Medium", "Low"]:
                sub = train_df_roi[train_df_roi["conf"] == tier]
                roi_rows.append({
                    "Tier":    tier,
                    "Games":   len(sub),
                    "Wins":    int(sub["correct"].sum()),
                    "Acc %":   round(float(sub["correct"].mean()) * 100, 1) if len(sub) > 0 else 0,
                    "ROI %":   _flat_bet_roi(sub),
                })
            roi_df = pd.DataFrame(roi_rows)

            colors_roi = {"High": GREEN, "Medium": "#d97706", "Low": "#6b7280"}
            fig_roi = px.bar(
                roi_df,
                x="Tier", y="ROI %",
                color="Tier",
                color_discrete_map=colors_roi,
                text="ROI %",
                labels={"ROI %": "Simulated ROI (%)"},
                height=300,
            )
            fig_roi.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_roi.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_roi.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_roi, width='stretch')
            st.dataframe(roi_df, hide_index=True, width='stretch')
    except Exception as e:
        st.info(f"Could not compute ROI simulation: {e}")
else:
    st.info("Train models and select an evaluation season to see ROI simulation.")

st.markdown("---")

# ── Cumulative P&L Chart ───────────────────────────────────────────────────────

st.subheader("📈 Cumulative P&L (Flat $100 @ −110)")
st.caption(
    "Simulated cumulative profit/loss across all predictions on the evaluation season, "
    "sorted chronologically. Bet direction = home if prob ≥ 50%, else away."
)

if models:
    try:
        # Reuse train_df_roi if already computed above, otherwise re-compute
        if "train_df_roi" not in dir() or train_df_roi.empty:
            train_df_roi = get_training_dataset(season_eval)
            cal_roi_models = load_calibrated_models()
            _used_models = cal_roi_models if cal_roi_models else models
            X_roi, _ = get_model_features(train_df_roi)
            probs_roi = ensemble_predict_proba(_used_models, X_roi)
            train_df_roi = train_df_roi.copy()
            train_df_roi["prob"]    = probs_roi
            train_df_roi["pred"]    = (probs_roi >= 0.5).astype(int)
            train_df_roi["correct"] = (train_df_roi["pred"] == train_df_roi["TARGET"]).astype(int)

        pnl_df = train_df_roi.copy().sort_values("GAME_DATE").reset_index(drop=True)
        pnl_df["pnl_game"] = pnl_df["correct"].apply(lambda c: 100.0 if c else -110.0)
        pnl_df["cum_pnl"]  = pnl_df["pnl_game"].cumsum()

        break_even_line = 0
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=pnl_df.index,
            y=pnl_df["cum_pnl"],
            mode="lines",
            line=dict(color=NBA_BLUE, width=2),
            name="Cumulative P&L",
            hovertemplate="Game %{x}<br>Cum P&L: $%{y:,.0f}",
        ))
        fig_pnl.add_hline(y=break_even_line, line_dash="dash", line_color="gray",
                          annotation_text="Break-even")
        fig_pnl.update_layout(
            xaxis_title="Game #",
            yaxis_title="Cumulative P&L ($)",
            yaxis=dict(tickprefix="$"),
            height=300,
            margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_pnl, width='stretch')
        final_pnl = float(pnl_df["cum_pnl"].iloc[-1]) if not pnl_df.empty else 0
        st.caption(
            f"Season final P&L: **${final_pnl:+,.0f}** over {len(pnl_df):,} games "
            f"(flat $100 bets, −110 odds assumed)."
        )
    except Exception as e:
        st.info(f"Could not compute P&L chart: {e}")
else:
    st.info("Train models and select an evaluation season to see the P&L chart.")

add_betting_oracle_footer()
