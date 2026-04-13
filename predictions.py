import streamlit as st
import pandas as pd
from datetime import datetime
from footer import add_betting_oracle_footer

# --- Page Configuration (called ONCE here; sub-pages must NOT call set_page_config) ---
st.set_page_config(
    page_title="Betting Baseline - NBA Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

NBA_BLUE = "#1D428A"
NBA_RED  = "#C8102E"
CONF_COLORS = {"High": "#16a34a", "Medium": "#d97706", "Low": "#6b7280"}


def _prob_bar_html(home_prob: float, home: str, away: str) -> str:
    """Inline HTML win-probability bar (no Plotly overhead on a landing page)."""
    hp = round(home_prob * 100)
    ap = 100 - hp
    return (
        f'<div style="display:flex;height:22px;border-radius:6px;overflow:hidden;font-size:0.75rem;font-weight:600">'
        f'<div style="width:{hp}%;background:{NBA_BLUE};color:white;display:flex;align-items:center;justify-content:center">{hp}%</div>'
        f'<div style="width:{ap}%;background:{NBA_RED};color:white;display:flex;align-items:center;justify-content:center">{ap}%</div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#888;margin-top:2px">'
        f'<span>{home}</span><span>{away}</span></div>'
    )


def _conf_badge(tier: str) -> str:
    c = CONF_COLORS.get(tier, "#6b7280")
    return f'<span style="background:{c};color:white;padding:1px 9px;border-radius:10px;font-size:0.72rem;font-weight:700">{tier}</span>'


def home_page():
    """Dashboard-style landing page."""
    from utils.data_fetcher import (
        get_today_predictions,
        get_multi_book_odds,
        get_best_lines,
        expected_value,
        kelly_criterion,
        CURRENT_SEASON,
    )
    from utils.model_utils import load_eval_metrics, load_totals_model
    from utils.prediction_engine import predict_total_points
    from utils.feature_engine import engineer_team_features
    from utils.data_fetcher import get_league_game_log_cached

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_left, hdr_right = st.columns([1, 4])
    with hdr_left:
        st.image("data_files/logo.png", width=130)
    with hdr_right:
        st.markdown(
            f"<h1 style='margin-bottom:0'>Betting Baseline</h1>"
            f"<p style='color:#888;margin-top:2px'>NBA Predictions · {CURRENT_SEASON} · "
            f"{datetime.today().strftime('%A, %B %d, %Y')}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Load today's predictions (cached; no spinner to keep it snappy) ───────
    try:
        preds_df = get_today_predictions()
    except Exception:
        preds_df = pd.DataFrame()

    metrics = load_eval_metrics()

    # ── Hero metrics row ──────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    total_games = len(preds_df) if not preds_df.empty else 0
    high_conf   = int((preds_df["confidence"] == "High").sum()) if not preds_df.empty else 0
    med_conf    = int((preds_df["confidence"] == "Medium").sum()) if not preds_df.empty else 0
    avg_conv    = (
        preds_df["home_win_prob"].clip(upper=0.99).apply(lambda p: max(p, 1 - p)).mean()
        if not preds_df.empty else 0.0
    )
    accuracy    = metrics.get("accuracy", None) if metrics else None

    m1.metric("Today's Games",      total_games)
    m2.metric("High Confidence",    high_conf,   delta=None)
    m3.metric("Medium Confidence",  med_conf,    delta=None)
    m4.metric("Avg Conviction",     f"{avg_conv:.0%}" if avg_conv else "—")
    m5.metric(
        "Model Accuracy",
        f"{accuracy:.1%}" if accuracy else "—",
        help="Ensemble accuracy on held-out games. Train via scripts/train_models.py.",
    )

    st.markdown("---")

    # ── Today's games + top picks ─────────────────────────────────────────────
    if preds_df.empty:
        st.info("No games found for today, or the NBA API is unavailable. Check back later.")
    else:
        left_col, right_col = st.columns([3, 2], gap="large")

        # ── Left: game cards ──────────────────────────────────────────────────
        with left_col:
            st.markdown(f"### 🏀 Today's Matchups ({total_games})")

            for _, g in preds_df.iterrows():
                home = g.get("home_team", "Home")
                away = g.get("away_team", "Away")
                hp   = float(g.get("home_win_prob", 0.5))
                conf = g.get("confidence", "Medium")
                spread = g.get("predicted_spread", None)

                spread_str = ""
                if spread is not None:
                    try:
                        spread_val = float(spread)
                        fav  = home if spread_val < 0 else away
                        spread_str = f"· {fav} -{abs(spread_val):.1f}"
                    except (TypeError, ValueError):
                        pass

                with st.container(border=True):
                    row_l, row_r = st.columns([5, 2])
                    with row_l:
                        st.markdown(
                            f"**{away}** @ **{home}** {spread_str}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(_prob_bar_html(hp, home, away), unsafe_allow_html=True)
                    with row_r:
                        st.markdown(
                            f'<div style="text-align:right;padding-top:8px">{_conf_badge(conf)}</div>',
                            unsafe_allow_html=True,
                        )
                        fav_label = home if hp >= 0.5 else away
                        st.caption(f"Pick: {fav_label}")

        # ── Right: best bets ──────────────────────────────────────────────────
        with right_col:
            st.markdown("### 🏆 Best Bets Today")

            # Load models + odds (all cached — fast on subsequent renders)
            import numpy as np
            totals_model = load_totals_model()
            multi_odds_raw  = get_multi_book_odds()
            best_lines_list = get_best_lines(multi_odds_raw) if multi_odds_raw else []

            def _tk(name: str) -> str:
                return name.split()[-1].lower()

            full_odds_lkp  = {_tk(g.get("home_team", "")): g for g in multi_odds_raw}
            best_lines_lkp = {_tk(bl["home_team"]): bl for bl in best_lines_list}

            # Build team feature map for O/U (needs game log; cached separately)
            @st.cache_data(ttl=3600)
            def _home_feat_map(season: str) -> dict:
                gl = get_league_game_log_cached(season)
                if gl.empty:
                    return {}
                result: dict = {}
                for tid, grp in gl.groupby("TEAM_ID"):
                    feats = engineer_team_features(grp.sort_values("GAME_DATE"))
                    if not feats.empty:
                        result[int(tid)] = feats.sort_values("GAME_DATE").iloc[-1]
                return result

            team_feat_map = _home_feat_map(CURRENT_SEASON) if totals_model is not None else {}

            # ── Score every game for each market ─────────────────────────────
            ml_picks, spread_picks, ou_picks = [], [], []

            for _, g in preds_df.iterrows():
                home = g.get("home_team", "Home")
                away = g.get("away_team", "Away")
                hp   = float(g.get("home_win_prob", 0.5))
                ap   = 1.0 - hp
                conf = g.get("confidence", "Low")
                pred_spread = g.get("predicted_spread")
                hid  = g.get("home_team_id")
                aid  = g.get("away_team_id")
                hk   = _tk(home)
                bl   = best_lines_lkp.get(hk)
                go   = full_odds_lkp.get(hk)

                # ─ ML EV ────────────────────────────────────────────────────
                if bl:
                    bhml = bl.get("best_home_ml")
                    baml = bl.get("best_away_ml")
                    if bhml is not None:
                        ev = expected_value(hp, bhml)
                        ml_picks.append({
                            "game": f"{away} @ {home}", "pick": f"{home} ML",
                            "odds": bhml, "book": bl.get("best_home_ml_book", ""),
                            "ev": ev, "prob": hp, "conf": conf, "score": ev,
                        })
                    if baml is not None:
                        ev = expected_value(ap, baml)
                        ml_picks.append({
                            "game": f"{away} @ {home}", "pick": f"{away} ML",
                            "odds": baml, "book": bl.get("best_away_ml_book", ""),
                            "ev": ev, "prob": ap, "conf": conf, "score": ev,
                        })

                # ─ Spread cover ─────────────────────────────────────────────
                if go and pred_spread is not None:
                    try:
                        sp_vals = [v for v in (go.get("home_spread") or {}).values() if v is not None]
                        if sp_vals:
                            cons_sp = float(np.mean(sp_vals))
                            edge = float(pred_spread) + cons_sp  # pos = home covers
                            if edge > 0:
                                pick = f"{home} {cons_sp:+.1f}"
                                cover_who = home
                            else:
                                pick = f"{away} +{abs(cons_sp):.1f}"
                                cover_who = away
                            spread_picks.append({
                                "game": f"{away} @ {home}", "pick": pick,
                                "cover": cover_who, "edge": abs(edge),
                                "pred": float(pred_spread), "line": cons_sp,
                                "conf": conf, "score": abs(edge),
                            })
                    except (TypeError, ValueError):
                        pass

                # ─ O/U ──────────────────────────────────────────────────────
                if totals_model is not None and hid and aid:
                    h_feat = team_feat_map.get(int(hid))
                    a_feat = team_feat_map.get(int(aid))
                    if h_feat is not None and a_feat is not None:
                        tot_vals = [v for v in (go or {}).get("total", {}).values() if v is not None]
                        cons_line = float(np.mean(tot_vals)) if tot_vals else None
                        ou = predict_total_points(h_feat, a_feat, totals_model, cons_line)
                        if ou.get("direction") in ("OVER", "UNDER") and ou.get("confidence"):
                            ou_picks.append({
                                "game": f"{away} @ {home}",
                                "pick": f"{ou['direction']} {cons_line:.1f}" if cons_line else ou["direction"],
                                "direction": ou["direction"],
                                "pred": ou.get("predicted_total"),
                                "line": ou.get("consensus_line"),
                                "margin": ou.get("margin"),
                                "conf_pct": ou["confidence"],
                                "conf": conf, "score": ou["confidence"],
                            })

            # ── Sort and pick best of each type ──────────────────────────────
            best_bets: list[dict] = []

            if ml_picks:
                ml_picks.sort(key=lambda x: x["score"], reverse=True)
                bm = ml_picks[0]
                best_bets.append({**bm, "type": "💵 Moneyline", "color_start": "#0f2e6b", "color_end": "#1D428A",
                                  "headline": bm["pick"],
                                  "subline": f"EV: ${bm['ev']:+.2f}/100 · {bm['odds']:+d} @ {bm['book']}",
                                  "badge": "Positive EV ✅" if bm["ev"] > 0 else "Negative EV",
                                  "good": bm["ev"] > 0})

            if spread_picks:
                spread_picks.sort(key=lambda x: x["score"], reverse=True)
                bs = spread_picks[0]
                best_bets.append({**bs, "type": "📐 Spread", "color_start": "#064e3b", "color_end": "#065f46",
                                  "headline": bs["pick"],
                                  "subline": f"Model: {bs['pred']:+.1f} · Line: {bs['line']:+.1f} · Edge: {bs['edge']:.1f}",
                                  "badge": "Covers ✅", "good": True})

            if ou_picks:
                ou_picks.sort(key=lambda x: x["score"], reverse=True)
                bo = ou_picks[0]
                icon = "📈" if bo["direction"] == "OVER" else "📉"
                best_bets.append({**bo, "type": "🎯 Over/Under", "color_start": "#78350f", "color_end": "#92400e",
                                  "headline": f"{icon} {bo['pick']}",
                                  "subline": (
                                      f"Pred: {bo['pred']:.1f} pts · {bo['margin']:+.1f} vs line · {bo['conf_pct']:.0%} conf"
                                      if bo.get("pred") and bo.get("margin") is not None
                                      else f"Pred: {bo['pred']:.1f} pts · {bo['conf_pct']:.0%} conf"
                                  ),
                                  "badge": f"{bo['conf_pct']:.0%} confidence", "good": True})

            # Fallback: high-confidence ML when no live odds
            if not best_bets:
                high_df = preds_df[preds_df["confidence"] == "High"]
                for _, g in high_df.iterrows():
                    home = g.get("home_team", "Home")
                    away = g.get("away_team", "Away")
                    hp   = float(g.get("home_win_prob", 0.5))
                    fav  = home if hp >= 0.5 else away
                    prob = max(hp, 1 - hp)
                    best_bets.append({
                        "type": "💵 Moneyline", "game": f"{away} @ {home}",
                        "headline": f"{fav} ML", "subline": f"Win prob: {prob:.0%}",
                        "badge": "High Confidence", "good": True,
                        "color_start": "#0f2e6b", "color_end": "#1D428A",
                    })

            # ── Render best-bet cards ─────────────────────────────────────────
            if not best_bets:
                st.info("No picks available yet — check back when today's lines are posted.")
            else:
                for bet in best_bets:
                    badge_color = "#16a34a" if bet.get("good") else "#dc2626"
                    st.markdown(
                        f'<div style="background:linear-gradient(90deg,{bet["color_start"]} 0%,{bet["color_end"]} 100%);'
                        f'border-radius:10px;padding:12px 16px;margin-bottom:10px;color:white">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<span style="font-size:0.72rem;opacity:0.75;font-weight:600;letter-spacing:0.05em">{bet["type"].upper()}</span>'
                        f'<span style="background:{badge_color};border-radius:8px;padding:1px 8px;font-size:0.68rem;font-weight:700">{bet["badge"]}</span>'
                        f'</div>'
                        f'<div style="font-size:0.72rem;opacity:0.65;margin-top:4px">{bet["game"]}</div>'
                        f'<div style="font-size:1.1rem;font-weight:800;margin-top:4px">{bet["headline"]}</div>'
                        f'<div style="font-size:0.8rem;opacity:0.85;margin-top:3px">{bet["subline"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    st.markdown("---")

    # ── Navigation tiles ──────────────────────────────────────────────────────
    st.markdown("### Explore")
    nc1, nc2, nc3, nc4, nc5, nc6 = st.columns(6)
    tiles = [
        ("🏀", "Predictions", "Win probabilities, spreads & matchup analysis", "pages/1_Game_Predictions.py"),
        ("🎯", "Pick 6",           "DK Pick 6 player prop builder",                "pages/2_Pick_6.py"),
        ("🏆", "Standings",        "Full conference & division standings",           "pages/3_Standings.py"),
        ("📊", "Team Stats",       "Advanced team metrics & trend charts",          "pages/4_Team_Stats.py"),
        ("👤", "Player Stats",     "Per-player dashboards & rolling averages",      "pages/5_Player_Stats.py"),
        ("🔬", "Models","Accuracy, calibration & feature importance",    "pages/6_Model_Performance.py"),
    ]
    for col, (icon, title, desc, path) in zip([nc1, nc2, nc3, nc4, nc5, nc6], tiles):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div style="text-align:center;font-size:1.6rem;padding-top:4px">{icon}</div>',
                    unsafe_allow_html=True,
                )
                st.page_link(path, label=f"**{title}**")
                st.caption(desc)

    # Betting Oracle footer component
    add_betting_oracle_footer()


# --- Navigation ---
pg = st.navigation(
    {
        "": [
            st.Page(home_page, title="Home", icon="🏠", default=True),
        ],
        "Predictions": [
            st.Page("pages/1_Game_Predictions.py", title="Game Predictions", icon="🏀"),
            st.Page("pages/2_Pick_6.py",           title="Pick 6",           icon="🎯"),
        ],
        "Stats": [
            st.Page("pages/3_Standings.py",         title="Standings",         icon="🏆"),
            st.Page("pages/4_Team_Stats.py",        title="Team Stats",        icon="📊"),
            st.Page("pages/5_Player_Stats.py",      title="Player Stats",      icon="🏃"),
        ],
        "Models": [
            st.Page("pages/6_Model_Performance.py", title="Model Performance", icon="📈"),
        ],
    }
)
pg.run()
