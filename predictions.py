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
        get_standings,
        CURRENT_SEASON,
    )
    from utils.model_utils import load_eval_metrics

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

        # ── Right: top picks + standings ──────────────────────────────────────
        with right_col:
            high_df = preds_df[preds_df["confidence"] == "High"]
            if not high_df.empty:
                st.markdown("### ⭐ Top Picks")
                for _, g in high_df.iterrows():
                    home = g.get("home_team", "Home")
                    away = g.get("away_team", "Away")
                    hp   = float(g.get("home_win_prob", 0.5))
                    fav  = home if hp >= 0.5 else away
                    fav_prob = max(hp, 1 - hp)
                    spread = g.get("predicted_spread", None)
                    spread_str = ""
                    if spread is not None:
                        try:
                            spread_val = float(spread)
                            spread_str = f"  ·  -{abs(spread_val):.1f}"
                        except (TypeError, ValueError):
                            pass
                    st.markdown(
                        f'<div style="background:linear-gradient(90deg,#0f2e6b 0%,#1D428A 100%);'
                        f'border-radius:10px;padding:10px 14px;margin-bottom:8px;color:white">'
                        f'<div style="font-size:0.8rem;opacity:0.7">{away} @ {home}</div>'
                        f'<div style="font-size:1.05rem;font-weight:700">{fav}{spread_str}</div>'
                        f'<div style="font-size:0.85rem;margin-top:2px">Win prob: <b>{fav_prob:.0%}</b>'
                        f'&nbsp;&nbsp;{_conf_badge("High")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Standings preview ──────────────────────────────────────────────
            st.markdown("### 📊 Standings")
            try:
                standings = get_standings(CURRENT_SEASON)
            except Exception:
                standings = pd.DataFrame()

            if not standings.empty:
                cols_want = ["TeamCity", "TeamName", "Conference", "PlayoffRank", "WINS", "LOSSES", "WinPCT"]
                avail = [c for c in cols_want if c in standings.columns]
                if avail:
                    s = standings[avail].copy()
                    s["Team"] = s.get("TeamCity", "") + " " + s.get("TeamName", "")
                    rank_col = "PlayoffRank" if "PlayoffRank" in s.columns else None
                    conf_col = "Conference"  if "Conference"  in s.columns else None

                    east_tab, west_tab = st.tabs(["East", "West"])
                    for tab, conf_val in [(east_tab, "East"), (west_tab, "West")]:
                        with tab:
                            sub = s[s[conf_col] == conf_val] if conf_col else s
                            if rank_col:
                                sub = sub.sort_values(rank_col)
                            disp_cols = ["Team", "WINS", "LOSSES", "WinPCT"]
                            disp_cols = [c for c in disp_cols if c in sub.columns]
                            show = sub[disp_cols].head(15).copy().reset_index(drop=True)
                            show.index = show.index + 1
                            col_cfg = {
                                "Team":   st.column_config.TextColumn("Team", width="medium"),
                                "WINS":   st.column_config.NumberColumn("W",  format="%d", width="small"),
                                "LOSSES": st.column_config.NumberColumn("L",  format="%d", width="small"),
                            }
                            if "WinPCT" in show.columns:
                                show["WinPCT"] = show["WinPCT"].astype(float)
                                col_cfg["WinPCT"] = st.column_config.ProgressColumn(
                                    "Win %",
                                    format="%.3f",
                                    min_value=0.0,
                                    max_value=1.0,
                                    width="medium",
                                )
                            st.dataframe(
                                show,
                                column_config=col_cfg,
                                width="stretch",
                                hide_index=False,
                                height=430,
                            )
            else:
                st.caption("Standings unavailable.")

    st.markdown("---")

    # ── Navigation tiles ──────────────────────────────────────────────────────
    st.markdown("### Explore")
    nc1, nc2, nc3, nc4, nc5 = st.columns(5)
    tiles = [
        ("🏀", "Game Predictions", "Win probabilities, spreads & matchup analysis", "pages/1_Game_Predictions.py"),
        ("🎯", "Pick 6",           "DK Pick 6 player prop builder",                "pages/2_Pick_6.py"),
        ("📈", "Team Stats",       "Advanced team metrics & trend charts",          "pages/3_Team_Stats.py"),
        ("👤", "Player Stats",     "Per-player dashboards & rolling averages",      "pages/4_Player_Stats.py"),
        ("🔬", "Model Performance","Accuracy, calibration & feature importance",    "pages/5_Model_Performance.py"),
    ]
    for col, (icon, title, desc, path) in zip([nc1, nc2, nc3, nc4, nc5], tiles):
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
            st.Page("pages/3_Team_Stats.py",       title="Team Stats",       icon="📊"),
            st.Page("pages/4_Player_Stats.py",     title="Player Stats",     icon="🏃"),
        ],
        "Models": [
            st.Page("pages/5_Model_Performance.py", title="Model Performance", icon="📈"),
        ],
    }
)
pg.run()
