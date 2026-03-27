"""
Player Stats page.

Individual player analysis: game logs, rolling stat trends, splits,
and historical performance reference for Pick 6 research.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.data_fetcher import (
    get_player_game_log_cached,
    get_league_player_stats_cached,
    get_all_active_players,
    get_nbastuffer_playerstats,
    CURRENT_SEASON,
    HISTORICAL_SEASONS,
)
from utils.feature_engine import engineer_player_features
from footer import add_betting_oracle_footer


NBA_BLUE = "#1D428A"
NBA_RED  = "#C8102E"
GREEN    = "#16a34a"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("data_files/logo.png", width=200)
    st.markdown("---")
    all_players = get_all_active_players()
    player_names = sorted(all_players["full_name"].tolist())
    player_search = st.text_input("Search Player", placeholder="e.g. LeBron James")
    if player_search:
        filtered = [n for n in player_names if player_search.lower() in n.lower()]
        display_list = filtered if filtered else player_names
    else:
        display_list = player_names

    selected_name = st.selectbox("Select Player", display_list)
    season_select = st.selectbox("Season", HISTORICAL_SEASONS[::-1], index=0)
    log_length    = st.selectbox("Game Log Length", ["Last 5", "Last 10", "Last 20", "Full Season"])
    compare_mode  = st.checkbox("Compare Players", value=False)
    if compare_mode:
        compare_name = st.selectbox("Compare With", [n for n in player_names if n != selected_name])
    else:
        compare_name = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_player_id(name: str) -> int | None:
    row = all_players[all_players["full_name"] == name]
    return int(row.iloc[0]["id"]) if not row.empty else None


def n_games(choice: str) -> int | None:
    return {"Last 5": 5, "Last 10": 10, "Last 20": 20, "Full Season": None}.get(choice)


def rolling_chart(df: pd.DataFrame, stat_col: str, window: int = 5) -> go.Figure:
    df = df.sort_values("GAME_DATE")
    fig = go.Figure()
    if stat_col not in df.columns:
        return fig
    fig.add_bar(
        x=df["GAME_DATE"], y=df[stat_col],
        marker_color=NBA_BLUE, opacity=0.5, name="Per game",
    )
    rolled = df[stat_col].rolling(window, min_periods=1).mean()
    fig.add_scatter(
        x=df["GAME_DATE"], y=rolled,
        mode="lines", line=dict(color=NBA_RED, width=2),
        name=f"{window}-game avg",
    )
    fig.add_hline(
        y=float(df[stat_col].mean()),
        line_dash="dot", line_color=GREEN, line_width=1,
        annotation_text=f"Season avg {df[stat_col].mean():.1f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=30),
        xaxis_title="Date", yaxis_title=stat_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def hist_chart(values: np.ndarray, stat: str) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=values, nbinsx=15, marker_color=NBA_BLUE, opacity=0.75, name=stat)
    fig.add_vline(x=float(np.mean(values)), line_color=NBA_RED, line_width=2,
                  annotation_text=f"Avg {np.mean(values):.1f}", annotation_position="top right")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=30), showlegend=False,
                      xaxis_title=stat, yaxis_title="Games")
    return fig


def splits_chart(df: pd.DataFrame, stat_col: str) -> go.Figure:
    if stat_col not in df.columns:
        return go.Figure()
    groups = {}
    if "IS_HOME" in df.columns:
        groups["Home"] = float(df[df["IS_HOME"] == 1][stat_col].mean())
        groups["Away"] = float(df[df["IS_HOME"] == 0][stat_col].mean())
    if "IS_B2B" in df.columns:
        groups["B2B"]     = float(df[df["IS_B2B"] == 1][stat_col].mean())
        groups["Rest 2+"] = float(df[df["IS_B2B"] == 0][stat_col].mean())
    if "WL" in df.columns:
        groups["Win"]  = float(df[df["WL"] == "W"][stat_col].mean())
        groups["Loss"] = float(df[df["WL"] == "L"][stat_col].mean())
    labels = list(groups.keys())
    values = list(groups.values())
    colors = [NBA_BLUE if v >= float(df[stat_col].mean()) else NBA_RED for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.add_hline(y=float(df[stat_col].mean()), line_dash="dash", line_color=GREEN,
                  annotation_text="Season avg")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20),
                      yaxis_title=stat_col, showlegend=False)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

st.title("🏃 Player Stats")

player_id = get_player_id(selected_name)
if player_id is None:
    st.error("Player not found.")
    st.stop()

with st.spinner(f"Loading data for {selected_name}..."):
    try:
        game_log = get_player_game_log_cached(player_id, season_select)
    except Exception as e:
        st.error(f"Error fetching player data: {e}")
        st.stop()

if game_log.empty:
    st.info(f"No game log data for {selected_name} in {season_select}.")
    st.stop()

df = engineer_player_features(game_log).sort_values("GAME_DATE", ascending=False)
n = n_games(log_length)
df_display = df.head(n) if n else df
df_full    = df.sort_values("GAME_DATE")

# ── Player header ──────────────────────────────────────────────────────────────

h1, h2, h3, h4, h5, h6 = st.columns(6)
for col, stat in zip([h1, h2, h3, h4, h5, h6], ["PTS", "REB", "AST", "FG3M", "STL", "MIN"]):
    if stat in df_full.columns:
        season_avg = df_full[stat].mean()
        l5_avg     = df_full[stat].tail(5).mean()
        delta      = l5_avg - season_avg
        col.metric(f"{stat} (season avg)", f"{season_avg:.1f}", delta=f"{delta:+.1f} (L5)")

st.caption(f"{season_select}  |  {len(df_full)} games played")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_log, tab_trends, tab_splits, tab_props, tab_nbs = st.tabs(
    ["Game Log", "Trends", "Splits", "Prop Research", "nbastuffer"]
)

with tab_log:
    log_cols = [c for c in ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST",
                              "FG3M", "FG_PCT", "FT_PCT", "STL", "BLK", "TOV", "PLUS_MINUS"]
                if c in df_display.columns]
    st.dataframe(
        df_display[log_cols].rename(columns={
            "GAME_DATE": "Date", "MATCHUP": "Matchup", "WL": "W/L", "MIN": "Min",
            "PTS": "Pts", "REB": "Reb", "AST": "Ast", "FG3M": "3PM",
            "FG_PCT": "FG%", "FT_PCT": "FT%", "STL": "Stl", "BLK": "Blk",
            "TOV": "Tov", "PLUS_MINUS": "+/-",
        }),
        hide_index=True, width='stretch'
    )

with tab_trends:
    stat_to_plot = st.selectbox("Stat", ["PTS", "REB", "AST", "FG3M", "PRA", "MIN", "PLUS_MINUS"],
                                 key="trend_stat")
    window_size  = st.slider("Rolling window", 3, 15, 5, key="trend_window")
    if stat_to_plot in df_full.columns:
        st.plotly_chart(rolling_chart(df_full, stat_to_plot, window_size),
                        width='stretch', config={"displayModeBar": False})
    # Distribution
    if stat_to_plot in df_full.columns:
        st.plotly_chart(hist_chart(df_full[stat_to_plot].dropna().values, stat_to_plot),
                        width='stretch', config={"displayModeBar": False})

with tab_splits:
    split_stat = st.selectbox("Stat", ["PTS", "REB", "AST", "FG3M", "PLUS_MINUS"], key="split_stat")
    if split_stat in df_full.columns:
        st.plotly_chart(splits_chart(df_full, split_stat),
                        width='stretch', config={"displayModeBar": False})
    # Numeric splits table
    splits_rows = []
    for label, mask in [("Home", df_full.get("IS_HOME", pd.Series(dtype=int)) == 1),
                         ("Away", df_full.get("IS_HOME", pd.Series(dtype=int)) == 0),
                         ("B2B",  df_full.get("IS_B2B",  pd.Series(dtype=int)) == 1),
                         ("Rest", df_full.get("IS_B2B",  pd.Series(dtype=int)) == 0),
                         ("Win",  df_full.get("WL", pd.Series(dtype=str)) == "W"),
                         ("Loss", df_full.get("WL", pd.Series(dtype=str)) == "L")]:
        sub = df_full[mask]
        if not sub.empty and split_stat in sub.columns:
            splits_rows.append({
                "Split": label, "Games": len(sub),
                "Avg": round(sub[split_stat].mean(), 1),
                "Std": round(sub[split_stat].std(), 1),
                "Max": round(sub[split_stat].max(), 1),
                "Min": round(sub[split_stat].min(), 1),
            })
    if splits_rows:
        st.dataframe(pd.DataFrame(splits_rows), hide_index=True, width='stretch')

with tab_props:
    st.markdown("**Historical performance vs common prop lines:**")
    prop_stat = st.selectbox("Stat", ["PTS", "REB", "AST", "FG3M", "PRA", "STL_BLK"], key="prop_stat_tab")
    if prop_stat in df_full.columns:
        season_avg  = df_full[prop_stat].mean()
        prop_rows = []
        for offset in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            line = round(season_avg + offset, 1)
            if line < 0:
                continue
            over_rate = float((df_full[prop_stat] > line).mean())
            prop_rows.append({
                "Line": line,
                "Over Rate (season)": f"{over_rate:.0%}",
                "Over Rate (L10)": f"{float((df_full[prop_stat].tail(10) > line).mean()):.0%}",
                "Edge vs 50/50": f"{(over_rate - 0.5):+.0%}",
            })
        st.dataframe(pd.DataFrame(prop_rows), hide_index=True, width='stretch')

with tab_nbs:
    st.caption("Advanced player stats from nbastuffer.com (TS%, USG%, PRA, ORTG/DRTG)")
    nbs_player_df = get_nbastuffer_playerstats(season_select)
    if nbs_player_df.empty:
        st.info(
            f"No nbastuffer player data for {season_select}. "
            "Run `python scripts/scrape_external.py --source nbastuffer` to populate."
        )
    else:
        # Match by last name (case-insensitive)
        last_name = selected_name.split()[-1].lower()
        name_col = next((c for c in ["NAME", "PLAYER", "name"] if c in nbs_player_df.columns), None)
        if name_col:
            mask = nbs_player_df[name_col].astype(str).str.lower().str.contains(last_name, na=False)
            player_nbs = nbs_player_df[mask]
        else:
            player_nbs = pd.DataFrame()

        if not player_nbs.empty:
            row = player_nbs.iloc[0]
            st.subheader(f"nbastuffer: {row.get(name_col, selected_name)}")
            stat_map = {
                "USG%": "Usage %", "TO%": "Turnover %", "TS%": "True Shooting %",
                "eFG%": "Effective FG %", "P+R+A": "Pts+Reb+Ast",
                "PpG": "Points/G", "RpG": "Rebounds/G", "ApG": "Assists/G",
                "SpG": "Steals/G", "BpG": "Blocks/G", "TOpG": "TO/G",
                "ORtg": "Off Rating", "DRtg": "Def Rating",
                "MpG": "Min/G", "GP": "Games", "FT%": "FT %", "3P%": "3P %",
            }
            grid_cols = [c for c in stat_map if c in nbs_player_df.columns]
            rows_of_4 = [grid_cols[i:i+4] for i in range(0, len(grid_cols), 4)]
            for row_cols in rows_of_4:
                cols = st.columns(len(row_cols))
                for ci, stat in zip(cols, row_cols):
                    val = row.get(stat, "—")
                    try:
                        val = f"{float(val):.3f}" if "%" in stat else f"{float(val):.1f}"
                    except (TypeError, ValueError):
                        val = str(val)
                    ci.metric(stat_map.get(stat, stat), val)
        else:
            st.info(f"No nbastuffer data found for '{selected_name}' in {season_select}.")

        # Full league leaderboard in this tab
        st.markdown("---")
        st.markdown("**League leaderboard (nbastuffer)**")
        nbs_sort_stat = st.selectbox(
            "Sort by",
            [c for c in ["PpG", "P+R+A", "USG%", "TS%", "eFG%", "ORtg", "DRtg", "ApG", "RpG"] if c in nbs_player_df.columns],
            key="nbs_player_sort",
        )
        display_player_cols = [c for c in [
            name_col, "TEAM", "POS", "GP", "MpG", "PpG", "RpG", "ApG", "USG%", "TO%",
            "TS%", "eFG%", "P+R+A", "ORtg", "DRtg"
        ] if c and c in nbs_player_df.columns]
        nbs_leaderboard = (
            nbs_player_df[display_player_cols]
            .sort_values(nbs_sort_stat, ascending=False, key=lambda x: pd.to_numeric(x, errors="coerce"))
            .reset_index(drop=True)
        )
        nbs_leaderboard.index += 1
        st.dataframe(nbs_leaderboard.head(50), width='stretch')

# ── Compare mode ───────────────────────────────────────────────────────────────

if compare_mode and compare_name:
    cid = get_player_id(compare_name)
    if cid:
        st.divider()
        st.subheader(f"⚖️ Comparison: {selected_name} vs {compare_name}")
        with st.spinner(f"Loading {compare_name}..."):
            clog  = get_player_game_log_cached(cid, season_select)
        if not clog.empty:
            cdf = engineer_player_features(clog)
            compare_stats = ["PTS", "REB", "AST", "FG3M", "STL", "MIN"]
            comp_rows = []
            for stat in compare_stats:
                v1 = df_full[stat].mean() if stat in df_full.columns else np.nan
                v2 = cdf[stat].mean()     if stat in cdf.columns else np.nan
                comp_rows.append({
                    "Stat": stat,
                    selected_name: round(v1, 1) if not np.isnan(v1) else "–",
                    compare_name:  round(v2, 1) if not np.isnan(v2) else "–",
                })
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width='stretch')

add_betting_oracle_footer()
