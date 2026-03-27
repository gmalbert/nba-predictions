"""
Team Stats page.

Interactive team statistics explorer: leaderboards, comparison radar charts,
rolling trend lines, and advanced metrics for all 30 teams.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.data_fetcher import (
    get_league_team_stats_cached,
    get_team_estimated_metrics_cached,
    get_league_game_log_cached,
    get_standings,
    get_all_teams,
    get_nbastuffer_teamstats,
    get_databallr_teams,
    CURRENT_SEASON,
    HISTORICAL_SEASONS,
)
from utils.feature_engine import engineer_team_features
from footer import add_betting_oracle_footer


NBA_BLUE = "#1D428A"
NBA_RED  = "#C8102E"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("data_files/logo.png", width=200)
    st.markdown("---")
    all_teams_df = get_all_teams()
    team_options = sorted(all_teams_df["full_name"].tolist())
    team_select = st.multiselect(
        "Select Teams (up to 4)",
        team_options,
        default=team_options[:2],
        max_selections=4,
    )
    season_select = st.selectbox("Season", HISTORICAL_SEASONS[::-1], index=0)
    rolling_window = st.select_slider(
        "Rolling Window (games)", options=[5, 10, 15, 20], value=10
    )
    st.markdown("---")
    view_mode = st.radio("View", ["League Rankings", "Team Trends", "Advanced Metrics", "Standings", "External Advanced"])


# ── Load data ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_team_stats(season: str):
    return get_league_team_stats_cached(season)


@st.cache_data(ttl=3600)
def load_estimated_metrics(season: str):
    return get_team_estimated_metrics_cached(season)


@st.cache_data(ttl=3600)
def load_game_log(season: str):
    return get_league_game_log_cached(season)


st.title("📊 Team Stats")

team_stats = load_team_stats(season_select)
est_metrics = load_estimated_metrics(season_select)

if team_stats.empty:
    st.warning(f"No team stats data for {season_select}. Run `scripts/fetch_historical.py` first.")
    st.stop()

# Build team ID lookups
all_teams_df_local = get_all_teams()
name_to_id = dict(zip(all_teams_df_local["full_name"], all_teams_df_local["id"]))
id_to_name = dict(zip(all_teams_df_local["id"], all_teams_df_local["full_name"]))
id_to_abbr  = dict(zip(all_teams_df_local["id"], all_teams_df_local["abbreviation"]))


# ── View: League Rankings ──────────────────────────────────────────────────────

if view_mode == "League Rankings":
    st.subheader(f"📋 League Rankings — {season_select}")

    stat_tabs = st.tabs(["Offense", "Defense", "Three Point", "Rebounding", "Misc"])

    off_cols  = ["TEAM_NAME", "PTS", "FGM", "FGA", "FG_PCT", "FTM", "FTA", "AST"]
    def_cols  = ["TEAM_NAME", "OPP_PTS", "OPP_FGM", "OPP_FGA", "OPP_FG_PCT", "STL", "BLK", "TOV"]
    three_cols = ["TEAM_NAME", "FG3M", "FG3A", "FG3_PCT"]
    reb_cols  = ["TEAM_NAME", "OREB", "DREB", "REB"]
    misc_cols = ["TEAM_NAME", "PLUS_MINUS", "W", "L", "W_PCT"]

    for tab_obj, col_set, sort_col, ascending in zip(
        stat_tabs,
        [off_cols, def_cols, three_cols, reb_cols, misc_cols],
        ["PTS", "OPP_PTS", "FG3_PCT", "REB", "PLUS_MINUS"],
        [False, True, False, False, False],
    ):
        with tab_obj:
            available = [c for c in col_set if c in team_stats.columns]
            sort_c = sort_col if sort_col in team_stats.columns else available[-1]
            display_df = team_stats[available].sort_values(sort_c, ascending=ascending).reset_index(drop=True)
            display_df.index += 1
            for col in display_df.select_dtypes("number").columns:
                display_df[col] = display_df[col].round(3)
            display_df = display_df.rename(columns={
                "TEAM_NAME": "Team", "PTS": "Pts", "FGM": "FGM", "FGA": "FGA",
                "FG_PCT": "FG%", "FTM": "FTM", "FTA": "FTA", "AST": "Ast",
                "OPP_PTS": "Opp Pts", "OPP_FGM": "Opp FGM", "OPP_FGA": "Opp FGA",
                "OPP_FG_PCT": "Opp FG%", "STL": "Stl", "BLK": "Blk", "TOV": "Tov",
                "FG3M": "3PM", "FG3A": "3PA", "FG3_PCT": "3P%",
                "OREB": "OReb", "DREB": "DReb", "REB": "Reb",
                "PLUS_MINUS": "+/-", "W": "W", "L": "L", "W_PCT": "Win%",
            })
            st.dataframe(display_df, width='stretch', hide_index=True)

    # Advanced metrics
    if not est_metrics.empty:
        st.markdown("---")
        st.subheader("⚡ Advanced Estimated Metrics")
        adv_display = est_metrics.copy()
        adv_display["TEAM_NAME"] = adv_display["TEAM_ID"].map(id_to_name)
        adv_cols = ["TEAM_NAME", "E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING", "E_PACE", "E_AST_RATIO"]
        available_adv = [c for c in adv_cols if c in adv_display.columns]
        adv_out = (
            adv_display[available_adv]
            .sort_values("E_NET_RATING" if "E_NET_RATING" in adv_display.columns else available_adv[-1], ascending=False)
            .reset_index(drop=True)
            .rename(columns={
                "TEAM_NAME": "Team", "E_OFF_RATING": "Off Rtg", "E_DEF_RATING": "Def Rtg",
                "E_NET_RATING": "Net Rtg", "E_PACE": "Pace", "E_AST_RATIO": "Ast Ratio",
            })
        )
        st.dataframe(adv_out, width='stretch', hide_index=True)


# ── View: Team Trends ──────────────────────────────────────────────────────────

elif view_mode == "Team Trends":
    if not team_select:
        st.info("Select at least one team in the sidebar.")
        st.stop()

    st.subheader(f"📈 Rolling {rolling_window}-Game Trends — {season_select}")
    game_log = load_game_log(season_select)

    if game_log.empty:
        st.info("Game log not available. Run fetch_historical.py first.")
        st.stop()

    game_log["GAME_DATE"] = pd.to_datetime(game_log["GAME_DATE"])
    stat_choice = st.selectbox("Stat to Plot", ["PLUS_MINUS", "PTS", "FG3_PCT", "AST", "REB", "TOV"])

    fig = go.Figure()
    colors = [NBA_BLUE, NBA_RED, "#16a34a", "#d97706"]
    for i, team_name in enumerate(team_select):
        tid = name_to_id.get(team_name)
        if tid is None:
            continue
        team_log = game_log[game_log["TEAM_ID"] == tid].sort_values("GAME_DATE")
        if team_log.empty or stat_choice not in team_log.columns:
            continue
        rolled = team_log[stat_choice].rolling(rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=team_log["GAME_DATE"],
            y=rolled,
            mode="lines",
            name=team_name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"{stat_choice} ({rolling_window}-game avg)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=30),
    )
    st.plotly_chart(fig, width='stretch')


# ── View: Advanced Metrics ─────────────────────────────────────────────────────

elif view_mode == "Advanced Metrics":
    if not team_select:
        st.info("Select at least one team in the sidebar.")
        st.stop()

    st.subheader(f"⚡ Advanced Metrics Comparison — {season_select}")
    if est_metrics.empty:
        st.info("Advanced metrics not available. Run fetch_historical.py first.")
        st.stop()

    cats = ["E_OFF_RATING", "E_DEF_RATING", "E_PACE", "E_AST_RATIO", "E_NET_RATING"]
    cats_available = [c for c in cats if c in est_metrics.columns]
    if not cats_available:
        st.info("Expected metrics columns not found.")
        st.stop()

    colors = [NBA_BLUE, NBA_RED, "#16a34a", "#d97706"]
    fig = go.Figure()
    for i, team_name in enumerate(team_select):
        tid = name_to_id.get(team_name)
        if tid is None:
            continue
        row = est_metrics[est_metrics["TEAM_ID"] == tid]
        if row.empty:
            continue
        vals = [float(row.iloc[0].get(c, 0)) for c in cats_available]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats_available + [cats_available[0]],
            fill="toself",
            name=team_name,
            line_color=colors[i % len(colors)],
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        height=450,
        margin=dict(l=20, r=20, t=30, b=30),
    )
    st.plotly_chart(fig, width='stretch')

    # Bar chart comparison
    st.markdown("---")
    bar_stat = st.selectbox("Bar Chart Stat", cats_available)
    bar_data = []
    for team_name in team_select:
        tid = name_to_id.get(team_name)
        if tid is None:
            continue
        row = est_metrics[est_metrics["TEAM_ID"] == tid]
        if not row.empty:
            bar_data.append({"Team": team_name, bar_stat: float(row.iloc[0].get(bar_stat, 0))})
    if bar_data:
        bar_df = pd.DataFrame(bar_data).sort_values(bar_stat, ascending=False)
        fig2 = px.bar(bar_df, x="Team", y=bar_stat, color="Team", height=350,
                      color_discrete_sequence=[NBA_BLUE, NBA_RED, "#16a34a", "#d97706"])
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, width='stretch')


# ── View: Standings ────────────────────────────────────────────────────────────

elif view_mode == "Standings":
    st.subheader(f"🏆 Standings — {season_select}")
    with st.spinner("Loading standings..."):
        standings = get_standings(season_select)
    if standings.empty:
        st.info("Standings not available.")
    else:
        disp_cols = [c for c in ["TeamName", "TeamCity", "Conference", "Division", "WINS", "LOSSES",
                                  "WinPCT", "HOME", "ROAD", "L10", "strCurrentStreak"] if c in standings.columns]
        for conf in ["East", "West"]:
            col = "Conference" if "Conference" in standings.columns else None
            if col:
                sub = standings[standings[col].str.contains(conf, case=False, na=False)][disp_cols]
            else:
                sub = standings[disp_cols]
            if not sub.empty:
                st.markdown(f"**{conf}ern Conference**")
                st.dataframe(
                    sub.reset_index(drop=True).rename(columns={
                        "TeamName": "Team", "TeamCity": "City", "Conference": "Conf",
                        "Division": "Division", "WINS": "W", "LOSSES": "L",
                        "WinPCT": "Win%", "HOME": "Home", "ROAD": "Away",
                        "L10": "Last 10", "strCurrentStreak": "Streak",
                    }),
                    hide_index=True, width='stretch'
                )




# ── View: External Advanced ────────────────────────────────────────────────────

elif view_mode == "External Advanced":
    st.subheader("🔬 External Advanced Metrics")

    ext_tab1, ext_tab2 = st.tabs(["nbastuffer Ratings", "databallr Efficiency"])

    with ext_tab1:
        st.caption("Data from nbastuffer.com — Schedule Adjusted Rating, Efficiency Differential, Consistency, etc.")
        nbs_season = st.selectbox(
            "Season", ["2024-25", "2023-24", "2022-23", "2021-22"], key="nbs_season"
        )
        nbs_split = st.selectbox(
            "Split", ["regular", "last5", "road", "home"], key="nbs_split"
        )
        nbs_df = get_nbastuffer_teamstats(nbs_season, split=nbs_split)
        if nbs_df.empty:
            st.info(
                f"No nbastuffer data for {nbs_season} / {nbs_split}. "
                "Run `python scripts/scrape_external.py --source nbastuffer` to populate."
            )
        else:
            display_cols = [c for c in [
                "TEAM", "GP", "SAR", "eDIFF", "CONS", "A4F", "WIN%", "eWIN%", "pWIN%",
                "PPG", "oPPG", "pDIFF", "PACE", "oEFF", "dEFF", "SoS", "W", "L"
            ] if c in nbs_df.columns]
            nbs_display = nbs_df[display_cols].copy()
            # Convert numeric columns from string to float
            for nc in [c for c in display_cols if c not in ("TEAM", "CONF", "DIVISION", "STRK", "SEASON", "SPLIT", "SOURCE")]:
                nbs_display[nc] = pd.to_numeric(nbs_display[nc], errors="coerce")
            sort_col = st.selectbox("Sort by", [c for c in ["SAR", "eDIFF", "WIN%", "eWIN%"] if c in display_cols], key="nbs_sort")
            nbs_display = nbs_display.sort_values(sort_col, ascending=False).reset_index(drop=True)
            nbs_display.index += 1
            st.dataframe(nbs_display, width='stretch')

            # Bar chart
            st.markdown("---")
            bar_metric = st.selectbox(
                "Bar Chart Metric",
                [c for c in ["SAR", "eDIFF", "CONS", "eWIN%", "WIN%", "PPG"] if c in display_cols],
                key="nbs_bar",
            )
            bar_fig = px.bar(
                nbs_display.reset_index().rename(columns={"index": "Rank"}),
                x="TEAM", y=bar_metric,
                color=bar_metric,
                color_continuous_scale="RdYlGn",
                height=400,
                title=f"{bar_metric} — {nbs_season} {nbs_split.title()}",
            )
            bar_fig.update_layout(
                xaxis_tickangle=-45,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=80),
            )
            st.plotly_chart(bar_fig, width='stretch')

    with ext_tab2:
        st.caption("Data from databallr.com — Offensive/Defensive Ratings and Efficiency Metrics (current season)")
        db_df = get_databallr_teams()
        if db_df.empty:
            st.info(
                "No databallr data found. "
                "Run `python scripts/scrape_external.py --source databallr` to populate."
            )
        else:
            db_df = db_df[db_df["Team"].notna() & (db_df["Team"] != "AVG")].copy()
            display_cols = [c for c in [
                "Rk", "Team", "ORTG", "DRTG", "Net", "OFF", "oTS", "oTOV", "ORB",
                "DEF", "dTS", "dTOV", "DRB", "NetEff", "NetPoss"
            ] if c in db_df.columns]
            sort_db = st.selectbox("Sort by", [c for c in ["Net", "ORTG", "DRTG", "NetEff"] if c in display_cols], key="db_sort")
            db_display = db_df[display_cols].copy()
            for nc in [c for c in display_cols if c not in ("Team", "Rk", "SEASON", "SOURCE", "PAGE_TYPE")]:
                db_display[nc] = pd.to_numeric(db_display[nc], errors="coerce")
            db_display = db_display.sort_values(sort_db, ascending=(sort_db == "DRTG")).reset_index(drop=True)
            db_display.index += 1
            st.dataframe(db_display, width='stretch')

            st.markdown("---")
            # Scatter: ORTG vs DRTG
            if "ORTG" in db_df.columns and "DRTG" in db_df.columns and "Net" in db_df.columns:
                scatter_df = db_df[display_cols].copy()
                for c in ["ORTG", "DRTG", "Net"]:
                    scatter_df[c] = pd.to_numeric(scatter_df[c], errors="coerce")
                fig_sc = px.scatter(
                    scatter_df.dropna(subset=["ORTG", "DRTG"]),
                    x="ORTG", y="DRTG",
                    text="Team",
                    color="Net",
                    color_continuous_scale="RdYlGn",
                    size_max=10,
                    title="ORTG vs DRTG (current season) — top-right = elite offense, bottom-left = elite defense",
                    height=500,
                )
                fig_sc.update_traces(textposition="top center", marker=dict(size=10))
                # Add reference lines at league avg
                avg_o = scatter_df["ORTG"].mean()
                avg_d = scatter_df["DRTG"].mean()
                fig_sc.add_hline(y=avg_d, line_dash="dot", line_color="gray", opacity=0.5)
                fig_sc.add_vline(x=avg_o, line_dash="dot", line_color="gray", opacity=0.5)
                fig_sc.update_layout(margin=dict(l=10, r=10, t=50, b=20))
                st.plotly_chart(fig_sc, width='stretch')


add_betting_oracle_footer()
