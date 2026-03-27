"""
Game Predictions page.

Shows win probabilities, predicted spreads/totals, and matchup analysis
for every NBA game on the selected date.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from utils.data_fetcher import (
    get_league_game_log_cached,
    get_league_team_stats_cached,
    get_team_estimated_metrics_cached,
    get_injury_report,
    get_all_teams,
    get_nbastuffer_teamstats,
    get_nbastuffer_refstats,
    get_today_referee_assignments,
    get_today_predictions,
    build_ref_lookup,
    get_multi_book_odds,
    get_best_lines,
    expected_value,
    kelly_criterion,
    BOOK_LABELS,
    CURRENT_SEASON,
)
from utils.prediction_engine import predict_today_games, win_prob_to_spread, predict_total_points
from utils.model_utils import load_models, load_totals_model, EloSystem, MODEL_DIR
from utils.feature_engine import engineer_team_features
from utils.feature_engine import enrich_with_nbastuffer_team
from footer import add_betting_oracle_footer



# ── Helpers ────────────────────────────────────────────────────────────────────

CONFIDENCE_COLORS = {"High": "#16a34a", "Medium": "#d97706", "Low": "#6b7280"}
NBA_BLUE = "#1D428A"
NBA_RED  = "#C8102E"


@st.cache_resource
def _load_models():
    return load_models()


@st.cache_resource
def _load_elo():
    elo_path = MODEL_DIR / "elo_system.pkl"
    return EloSystem.load(elo_path) if elo_path.exists() else None


@st.cache_resource
def _load_totals_model():
    return load_totals_model()


@st.cache_data(ttl=3600)
def _build_team_features_map(season: str) -> dict[int, pd.Series]:
    """Build {team_id: latest_feature_row} for all teams. Cached for 1 hour."""
    game_log = get_league_game_log_cached(season)
    if game_log.empty:
        return {}
    result: dict[int, pd.Series] = {}
    for tid, grp in game_log.groupby("TEAM_ID"):
        feats = engineer_team_features(grp.sort_values("GAME_DATE"))
        if not feats.empty:
            result[int(tid)] = feats.sort_values("GAME_DATE").iloc[-1]
    return result


def confidence_badge(tier: str) -> str:
    color = CONFIDENCE_COLORS.get(tier, "#6b7280")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600">{tier}</span>'


def prob_bar(home_prob: float, home_name: str, away_name: str) -> go.Figure:
    """Horizontal stacked bar showing home/away win probability."""
    fig = go.Figure()
    fig.add_bar(x=[home_prob * 100], y=[""], orientation="h",
                marker_color=NBA_BLUE, name=home_name, text=f"{home_prob:.0%}",
                textposition="inside", insidetextanchor="middle")
    fig.add_bar(x=[(1 - home_prob) * 100], y=[""], orientation="h",
                marker_color=NBA_RED, name=away_name, text=f"{1-home_prob:.0%}",
                textposition="inside", insidetextanchor="middle")
    fig.update_layout(
        barmode="stack", height=60, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def radar_chart(home_stats: dict, away_stats: dict, home: str, away: str) -> go.Figure:
    cats = ["Off Rating", "Def Rating (inv)", "Pace", "3PT%", "Rebound%", "AST Ratio"]
    home_vals = [
        home_stats.get("OFF_RATING", 110),
        120 - home_stats.get("DEF_RATING", 110),
        home_stats.get("PACE", 100),
        home_stats.get("FG3_PCT", 0.35) * 100,
        home_stats.get("REB_PCT", 0.5) * 100,
        home_stats.get("AST_RATIO", 15),
    ]
    away_vals = [
        away_stats.get("OFF_RATING", 110),
        120 - away_stats.get("DEF_RATING", 110),
        away_stats.get("PACE", 100),
        away_stats.get("FG3_PCT", 0.35) * 100,
        away_stats.get("REB_PCT", 0.5) * 100,
        away_stats.get("AST_RATIO", 15),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_vals + [home_vals[0]], theta=cats + [cats[0]],
                                  fill="toself", name=home, line_color=NBA_BLUE, opacity=0.6))
    fig.add_trace(go.Scatterpolar(r=away_vals + [away_vals[0]], theta=cats + [cats[0]],
                                  fill="toself", name=away, line_color=NBA_RED, opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), height=350,
                      margin=dict(l=20, r=20, t=20, b=20), legend=dict(x=0.5, y=-0.05, xanchor="center"))
    return fig


def _fmt_american(odds: int | float | None) -> str:
    if odds is None:
        return "—"
    return f"+{int(odds)}" if odds >= 0 else str(int(odds))


def _ev_badge(ev: float) -> str:
    if ev > 0:
        return f'<span style="background:#16a34a;color:white;padding:1px 8px;border-radius:10px;font-size:0.75rem;font-weight:700">+EV ${ev:+.2f}</span>'
    return f'<span style="background:#dc2626;color:white;padding:1px 8px;border-radius:10px;font-size:0.75rem;font-weight:700">−EV ${ev:.2f}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("data_files/logo.png", width=200)
    st.markdown("---")
    selected_date = st.date_input("Game Date", value=datetime.today())
    conf_filter   = st.selectbox("Confidence Filter", ["All", "High", "Medium", "Low"])
    show_injuries = st.checkbox("Show injury report", value=False)
    show_odds     = st.checkbox("Show multi-book odds", value=True)
    st.markdown("---")
    st.caption(f"Season: {CURRENT_SEASON}")

game_date_str = selected_date.strftime("%m/%d/%Y")

# ── Main Content ───────────────────────────────────────────────────────────────

st.title("🏀 Game Predictions")
st.caption(f"Predictions for {selected_date.strftime('%A, %B %d, %Y')}")

# Load models (quiet on missing)
models       = _load_models()
elo          = _load_elo()
totals_model = _load_totals_model()

with st.spinner("Loading predictions..."):
    try:
        preds_df = get_today_predictions(game_date_mmddyyyy=game_date_str)
    except Exception as e:
        st.error(f"Could not generate predictions: {e}")
        preds_df = pd.DataFrame()

if preds_df.empty:
    st.info("No games found for this date, or the NBA API is unavailable.")
else:
    if conf_filter != "All":
        preds_df = preds_df[preds_df["confidence"] == conf_filter]
        if preds_df.empty:
            st.info(f"No {conf_filter} confidence games today.")
            st.stop()

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    high_cnt   = len(preds_df[preds_df["confidence"] == "High"])
    med_cnt    = len(preds_df[preds_df["confidence"] == "Medium"])
    avg_prob   = preds_df["home_win_prob"].clip(upper=0.99).apply(lambda p: max(p, 1-p)).mean()
    col1.metric("Total Games", len(preds_df))
    col2.metric("High Confidence", high_cnt)
    col3.metric("Medium Confidence", med_cnt)
    col4.metric("Avg Conviction", f"{avg_prob:.0%}")

    st.markdown("---")

    # Game cards
    teams_df = get_all_teams()
    abbr_map  = dict(zip(teams_df["id"], teams_df["abbreviation"]))

    # Team stats for radar charts
    team_stats_df = get_league_team_stats_cached(CURRENT_SEASON)
    team_stats_map: dict = {}
    if not team_stats_df.empty:
        for _, r in team_stats_df.iterrows():
            team_stats_map[r.get("TEAM_ID", r.get("TEAM_NAME", ""))] = r.to_dict()

    # nbastuffer advanced team stats for contextual enrichment
    nbs_team_df = get_nbastuffer_teamstats(CURRENT_SEASON)
    nbs_ref_lookup = build_ref_lookup(CURRENT_SEASON)

    # Today's referee assignments keyed by NBA_GAME_ID (same format as predictions)
    today_ref_df = get_today_referee_assignments()
    _ref_lookup: dict[str, pd.DataFrame] = {}
    if not today_ref_df.empty and "NBA_GAME_ID" in today_ref_df.columns:
        today_ref_df = today_ref_df.copy()
        today_ref_df["NBA_GAME_ID"] = today_ref_df["NBA_GAME_ID"].astype(str)
        for _gid, _grp in today_ref_df[today_ref_df["NBA_GAME_ID"] != ""].groupby("NBA_GAME_ID"):
            _ref_lookup[str(_gid)] = _grp

    # Multi-book odds — keyed by (home_team_fragment, away_team_fragment)
    multi_odds_raw   = get_multi_book_odds() if show_odds else []
    best_lines_list  = get_best_lines(multi_odds_raw) if multi_odds_raw else []
    # Build a lookup: normalized home team name → best_lines entry
    def _team_key(name: str) -> str:
        return name.split()[-1].lower()  # last word of team name
    best_lines_lookup: dict[str, dict] = {
        _team_key(bl["home_team"]): bl for bl in best_lines_list
    }
    # Full game dict lookup for odds table
    full_odds_lookup: dict[str, dict] = {
        _team_key(g.get("home_team", "")): g for g in multi_odds_raw
    }

    # Pre-build team feature rows once (cached) — used for O/U predictions
    team_feat_map = _build_team_features_map(CURRENT_SEASON) if totals_model is not None else {}

    for _, game in preds_df.iterrows():
        home  = game["home_team"]
        away  = game["away_team"]
        hprob = game["home_win_prob"]
        aprob = game["away_win_prob"]
        conf  = game["confidence"]
        spread = game["predicted_spread"]
        edge   = game.get("edge")

        with st.expander(f"**{away}  @  {home}**  |  {hprob:.0%} / {aprob:.0%}", expanded=True):
            # ── Pre-compute odds context (shared across all card sections) ─────
            home_key  = _team_key(home)
            game_odds = full_odds_lookup.get(home_key)
            bl_entry  = best_lines_lookup.get(home_key)

            # Consensus spread & total from sportsbooks
            _spreads_raw = (game_odds or {}).get("home_spread") or {}
            _spread_vals = [v for v in _spreads_raw.values() if v is not None]
            cons_spread  = float(np.mean(_spread_vals)) if _spread_vals else None

            _totals_raw = (game_odds or {}).get("total") or {}
            _total_vals = [v for v in _totals_raw.values() if v is not None]
            cons_line   = float(np.mean(_total_vals)) if _total_vals else None

            # O/U prediction
            hid = game.get("home_team_id")
            aid = game.get("away_team_id")
            ou_result: dict = {}
            if totals_model is not None and hid and aid:
                h_feat = team_feat_map.get(int(hid))
                a_feat = team_feat_map.get(int(aid))
                if h_feat is not None and a_feat is not None:
                    ou_result = predict_total_points(
                        h_feat, a_feat,
                        total_model=totals_model,
                        consensus_line=cons_line,
                    )

            # EV for home / away moneyline
            best_hml = bl_entry.get("best_home_ml") if bl_entry else None
            best_aml = bl_entry.get("best_away_ml") if bl_entry else None
            home_ev  = expected_value(hprob, best_hml) if best_hml is not None else None
            away_ev  = expected_value(aprob, best_aml) if best_aml is not None else None

            lc, rc = st.columns([3, 2])

            with lc:
                # Win probability bar
                fig = prob_bar(hprob, home, away)
                st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Home Win %",  f"{hprob:.0%}")
                m2.metric("Away Win %",  f"{aprob:.0%}")
                m3.metric("Spread",      f"{'Home' if spread < 0 else 'Away'} {abs(spread):.1f}")
                if edge is not None:
                    m4.metric("Edge vs Market", f"{edge:+.1%}")

                st.markdown(
                    f"Confidence: {confidence_badge(conf)}",
                    unsafe_allow_html=True,
                )
                if game.get("elo_prob"):
                    st.caption(
                        f"Elo: {game['elo_prob']:.0%}  |  ML: {game.get('ml_prob', 0):.0%}"
                    )

            with rc:
                # Radar chart if team stats available
                if hid and aid and team_stats_map:
                    hs = team_stats_map.get(hid, {})
                    as_ = team_stats_map.get(aid, {})
                    est = get_team_estimated_metrics_cached(CURRENT_SEASON)
                    if not est.empty:
                        hest = est[est["TEAM_ID"] == hid]
                        aest = est[est["TEAM_ID"] == aid]
                        if not hest.empty:
                            hs.update(hest.iloc[0].to_dict())
                        if not aest.empty:
                            as_.update(aest.iloc[0].to_dict())
                    fig2 = radar_chart(hs, as_, home, away)
                    st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})

            # ── Betting Signals ────────────────────────────────────────────────
            _has_spread = cons_spread is not None
            _has_ou     = bool(ou_result.get("predicted_total"))
            _has_ev     = home_ev is not None or away_ev is not None

            if _has_spread or _has_ou or _has_ev:
                st.markdown("---")
                st.caption("🎯 Betting Signals")
                bs1, bs2, bs3 = st.columns(3)

                # 1. Spread cover
                with bs1:
                    if _has_spread:
                        # predicted_spread: positive = home wins by X pts
                        # cons_spread: negative = home is favorite by |X| (market convention)
                        # home covers when predicted margin exceeds the spread
                        home_covers = (spread + cons_spread) > 0
                        cover_icon  = "✅" if home_covers else "❌"
                        cover_label = f"{home} Covers" if home_covers else f"{home} No Cover"
                        st.metric(
                            label=f"Spread Cover (line: {home} {cons_spread:+.1f})",
                            value=f"{cover_icon} {cover_label}",
                            delta=f"Model {spread:+.1f} | Line {cons_spread:+.1f}",
                            delta_color="normal" if home_covers else "inverse",
                            help="Whether model's predicted margin beats the consensus spread.",
                        )
                    else:
                        st.metric("Spread Cover", "No line available")

                # 2. Over/Under direction
                with bs2:
                    ou_td = ou_result.get("predicted_total")
                    if ou_td is not None:
                        ou_dir   = ou_result.get("direction", "—")
                        ou_line  = ou_result.get("consensus_line")
                        ou_conf  = ou_result.get("confidence")
                        ou_icon  = "📈" if ou_dir == "OVER" else "📉" if ou_dir == "UNDER" else ""
                        _margin  = ou_result.get("margin")
                        st.metric(
                            label=f"O/U (line: {ou_line:.1f})" if ou_line else "Over/Under",
                            value=f"{ou_icon} {ou_dir}",
                            delta=(
                                f"Pred {ou_td:.1f} ({_margin:+.1f} vs line)"
                                if ou_line and _margin is not None
                                else f"Pred {ou_td:.1f} pts"
                            ),
                            delta_color="normal" if ou_dir == "OVER" else "inverse" if ou_dir == "UNDER" else "off",
                            help=f"Confidence: {ou_conf:.0%}" if ou_conf is not None else None,
                        )
                    else:
                        st.metric("Over/Under", "—", help="Totals model unavailable or no line")

                # 3. Best moneyline value
                with bs3:
                    if _has_ev:
                        _ev_map: dict = {}
                        if home_ev is not None:
                            _ev_map[home] = (home_ev, best_hml, bl_entry.get("best_home_ml_book", ""))
                        if away_ev is not None:
                            _ev_map[away] = (away_ev, best_aml, bl_entry.get("best_away_ml_book", ""))
                        best_t = max(_ev_map, key=lambda t: _ev_map[t][0])
                        bev, bodds, bbook = _ev_map[best_t]
                        ev_icon = "✅" if bev > 0 else "❌"
                        st.metric(
                            label=f"Best ML Value ({_fmt_american(bodds)} @ {bbook})" if bbook else f"Best ML Value ({_fmt_american(bodds)})",
                            value=f"{ev_icon} {best_t}",
                            delta=f"${bev:+.2f} per $100",
                            delta_color="normal" if bev > 0 else "inverse",
                            help="Expected value per $100 wagered on the best available moneyline.",
                        )
                    else:
                        st.metric("ML Value", "—", help="Live odds unavailable")

            # ── nbastuffer contextual metrics row ──────────────────────────────
            if not nbs_team_df.empty:
                home_last = home.split()[-1]
                away_last = away.split()[-1]
                h_nbs = enrich_with_nbastuffer_team(home_last, nbs_team_df)
                a_nbs = enrich_with_nbastuffer_team(away_last, nbs_team_df)
                if not (np.isnan(h_nbs["NBS_SAR"]) and np.isnan(a_nbs["NBS_SAR"])):
                    st.markdown("---")
                    st.caption("📊 nbastuffer Advanced Ratings")
                    nc1, nc2, nc3, nc4, nc5, nc6 = st.columns(6)
                    def _fmt(v, fmt=".2f"):
                        return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "—"
                    nc1.metric(f"{home} SAR",    _fmt(h_nbs["NBS_SAR"]))
                    nc2.metric(f"{home} eDIFF",  _fmt(h_nbs["NBS_eDIFF"]))
                    nc3.metric(f"{home} eWIN%",  _fmt(h_nbs["NBS_eWIN_PCT"], ".3f"))
                    nc4.metric(f"{away} SAR",    _fmt(a_nbs["NBS_SAR"]))
                    nc5.metric(f"{away} eDIFF",  _fmt(a_nbs["NBS_eDIFF"]))
                    nc6.metric(f"{away} eWIN%",  _fmt(a_nbs["NBS_eWIN_PCT"], ".3f"))

            # ── Referee crew assignments (ESPN / nba_official + nbastuffer cross stats)
            game_refs = _ref_lookup.get(str(game.get("game_id", "")), pd.DataFrame())
            with st.expander("👨‍⚖️ Game Officials", expanded=False):
                if game_refs.empty:
                    st.info("Referee crew not yet posted for this game.  Check again closer to tip-off.")
                else:
                    game_refs = game_refs.sort_values(["ORDER"])
                    display_cols = ["ORDER", "REFEREE", "SOURCE", "CALLED_FOULS_PER_GAME", "HOME_WIN_PCT", "FOUL_PCT_ROAD", "FOUL_PCT_HOME", "FOUL_DIFFERENTIAL", "EXPERIENCE_YEARS"]
                    df_show = game_refs[display_cols].fillna("—")
                    df_show["ORDER"] = df_show["ORDER"].astype(int)
                    df_show = df_show.rename(columns={
                        "ORDER": "Position",
                        "REFEREE": "Official",
                        "SOURCE": "Source",
                        "CALLED_FOULS_PER_GAME": "Fouls/Game",
                        "HOME_WIN_PCT": "Home Win%",
                        "FOUL_PCT_ROAD": "Foul% vs Road",
                        "FOUL_PCT_HOME": "Foul% vs Home",
                        "FOUL_DIFFERENTIAL": "Foul Diff",
                        "EXPERIENCE_YEARS": "Experience (yrs)",
                    })
                    st.dataframe(df_show, hide_index=True, width='stretch')

            # ── Multi-book odds comparison ──────────────────────────────────────
            if show_odds:
                if game_odds or bl_entry:
                    st.markdown("---")
                    with st.expander("📊 Multi-Book Odds & Expected Value", expanded=False):
                        # Build comparison table
                        home_mls   = (game_odds or {}).get("home_ml") or {}
                        away_mls   = (game_odds or {}).get("away_ml") or {}
                        spreads    = (game_odds or {}).get("home_spread") or {}
                        totals     = (game_odds or {}).get("total") or {}
                        over_odds  = (game_odds or {}).get("over_odds") or {}
                        under_odds = (game_odds or {}).get("under_odds") or {}

                        table_rows = []
                        from utils.data_fetcher import SBRSCRAPE_BOOKS
                        for bk in SBRSCRAPE_BOOKS:
                            hml = home_mls.get(bk)
                            aml = away_mls.get(bk)
                            if hml is None and aml is None:
                                continue
                            table_rows.append({
                                "Book":       BOOK_LABELS.get(bk, bk),
                                "Home ML":    _fmt_american(hml),
                                "Away ML":    _fmt_american(aml),
                                "Spread":     _fmt_american(spreads.get(bk)),
                                "Total":      f"{totals.get(bk):.1f}" if totals.get(bk) else "—",
                                "Over Odds":  _fmt_american(over_odds.get(bk)),
                                "Under Odds": _fmt_american(under_odds.get(bk)),
                            })

                        if table_rows:
                            odds_tbl_df = pd.DataFrame(table_rows)
                            # Highlight best home ML (most positive) and best away ML
                            numeric_hml = {BOOK_LABELS.get(bk, bk): v for bk, v in home_mls.items() if v is not None}
                            numeric_aml = {BOOK_LABELS.get(bk, bk): v for bk, v in away_mls.items() if v is not None}
                            best_home_book = max(numeric_hml, key=numeric_hml.get) if numeric_hml else None
                            best_away_book = max(numeric_aml, key=numeric_aml.get) if numeric_aml else None

                            def _highlight_best(row):
                                styles = [""] * len(row)
                                cols   = list(row.index)
                                if best_home_book and row["Book"] == best_home_book:
                                    styles[cols.index("Home ML")] = "background-color: #bbf7d0; font-weight: bold"
                                if best_away_book and row["Book"] == best_away_book:
                                    styles[cols.index("Away ML")] = "background-color: #bbf7d0; font-weight: bold"
                                return styles

                            st.dataframe(
                                odds_tbl_df.style.apply(_highlight_best, axis=1),
                                hide_index=True,
                                width='stretch',
                            )

                        # EV & Kelly section — uses values pre-computed above
                        if bl_entry and best_hml is not None:
                            st.markdown("##### 💰 Expected Value (vs Best Available Line)")
                            ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
                            ev_c1.metric(
                                f"{home} ML ({_fmt_american(best_hml)} @ {bl_entry.get('best_home_ml_book','')})",
                                f"${home_ev:+.2f} / $100",
                                delta="Positive EV ✅" if home_ev > 0 else "Negative EV ❌",
                                delta_color="normal" if home_ev > 0 else "inverse",
                            )
                            home_kelly = kelly_criterion(hprob, best_hml)
                            ev_c2.metric(
                                "Kelly Criterion (Home)",
                                f"{home_kelly * 25:.1f}% bankroll",
                                help="Quarter-Kelly (÷4) recommended. Full Kelly shown ×0.25.",
                            )
                            if best_aml is not None:
                                ev_c3.metric(
                                    f"{away} ML ({_fmt_american(best_aml)} @ {bl_entry.get('best_away_ml_book','')})",
                                    f"${away_ev:+.2f} / $100",
                                    delta="Positive EV ✅" if away_ev > 0 else "Negative EV ❌",
                                    delta_color="normal" if away_ev > 0 else "inverse",
                                )
                                away_kelly = kelly_criterion(aprob, best_aml)
                                ev_c4.metric(
                                    "Kelly Criterion (Away)",
                                    f"{away_kelly * 25:.1f}% bankroll",
                                    help="Quarter-Kelly (÷4) recommended.",
                                )

                        elif show_odds:
                            st.caption("_No live odds available from sbrscrape or The Odds API for this game._")

# Injury report — rendered regardless of whether predictions loaded
if show_injuries:
    st.markdown("---")
    st.subheader("Current Injury Report")
    with st.spinner("Fetching injury data..."):
        injuries_df = get_injury_report()
    if injuries_df.empty:
        st.info("No injury data available.")
    else:
        inj_show = injuries_df.sort_values(["team", "status"]).rename(columns={
            "team":        "Team",
            "player_name": "Player",
            "position":    "Position",
            "status":      "Status",
            "description": "Description",
        })
        st.dataframe(inj_show, hide_index=True, width='stretch')

add_betting_oracle_footer()
