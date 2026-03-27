"""
Standings page.

Full NBA standings with conference, division, and streak breakdowns.
"""

import streamlit as st
import pandas as pd

from utils.data_fetcher import (
    get_standings,
    CURRENT_SEASON,
    HISTORICAL_SEASONS,
)
from footer import add_betting_oracle_footer


NBA_BLUE = "#1D428A"
NBA_RED  = "#C8102E"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("data_files/logo.png", width=200)
    st.markdown("---")
    season_select = st.selectbox("Season", HISTORICAL_SEASONS[::-1], index=0)
    view_mode = st.radio("View", ["Conference", "Division"])

# ── Load ───────────────────────────────────────────────────────────────────────

st.title("🏆 Standings")
st.caption(f"Season: {season_select}")

with st.spinner("Loading standings..."):
    standings = get_standings(season_select)

if standings.empty:
    st.warning(
        "Standings unavailable. The NBA API may be unreachable. "
        "Run `python scripts/fetch_historical.py` to cache data."
    )
    add_betting_oracle_footer()
    st.stop()

# ── Column normalization ───────────────────────────────────────────────────────

ALL_COLS = [
    "TeamCity", "TeamName", "Conference", "Division",
    "PlayoffRank", "WINS", "LOSSES", "WinPCT",
    "HOME", "ROAD", "L10", "strCurrentStreak",
    "ConferenceGamesBack", "DivisionGamesBack",
]
avail = [c for c in ALL_COLS if c in standings.columns]
s = standings[avail].copy()

if "TeamCity" in s.columns and "TeamName" in s.columns:
    s.insert(0, "Team", s["TeamCity"] + " " + s["TeamName"])
elif "TeamName" in s.columns:
    s.insert(0, "Team", s["TeamName"])

rename_map = {
    "WINS":                "W",
    "LOSSES":              "L",
    "WinPCT":              "Win%",
    "HOME":                "Home",
    "ROAD":                "Away",
    "L10":                 "Last 10",
    "strCurrentStreak":    "Streak",
    "PlayoffRank":         "Seed",
    "ConferenceGamesBack": "GB (Conf)",
    "DivisionGamesBack":   "GB (Div)",
}
s = s.rename(columns=rename_map)

display_cols = [c for c in ["Team", "W", "L", "Win%", "Home", "Away", "Last 10", "Streak", "Seed", "GB (Conf)", "GB (Div)"] if c in s.columns]

conf_col  = "Conference" if "Conference" in s.columns else None
div_col   = "Division"   if "Division"   in s.columns else None
seed_col  = "Seed"       if "Seed"       in s.columns else None


def _build_conf_df(conf_val: str) -> pd.DataFrame:
    if conf_col:
        df = s[s[conf_col].str.contains(conf_val, case=False, na=False)].copy()
    else:
        df = s.copy()
    if seed_col and seed_col in df.columns:
        df = df.sort_values(seed_col)
    df = df[display_cols].reset_index(drop=True)
    df.index = df.index + 1
    return df


def _col_config(df: pd.DataFrame) -> dict:
    cfg: dict = {
        "Team": st.column_config.TextColumn("Team", width="large"),
    }
    for c in ["W", "L", "Seed"]:
        if c in df.columns:
            cfg[c] = st.column_config.NumberColumn(c, format="%d", width="small")
    if "Win%" in df.columns:
        cfg["Win%"] = st.column_config.ProgressColumn(
            "Win%", format="%.3f", min_value=0.0, max_value=1.0, width="medium"
        )
    return cfg


# ── Conference view ────────────────────────────────────────────────────────────

if view_mode == "Conference":
    east_tab, west_tab = st.tabs(["🔵 Eastern Conference", "🔴 Western Conference"])

    with east_tab:
        east_df = _build_conf_df("East")
        if east_df.empty:
            st.info("No data available.")
        else:
            st.dataframe(east_df, column_config=_col_config(east_df), use_container_width=True)

    with west_tab:
        west_df = _build_conf_df("West")
        if west_df.empty:
            st.info("No data available.")
        else:
            st.dataframe(west_df, column_config=_col_config(west_df), use_container_width=True)

# ── Division view ──────────────────────────────────────────────────────────────

elif view_mode == "Division":
    if div_col is None or div_col not in s.columns:
        st.info("Division data not available.")
    else:
        divisions = s[div_col].dropna().unique().tolist()
        # Group by conference so East divisions come first
        if conf_col and conf_col in s.columns:
            east_divs = sorted(
                [d for d in divisions if s.loc[s[div_col] == d, conf_col].str.contains("East", case=False, na=False).any()]
            )
            west_divs = sorted(
                [d for d in divisions if d not in east_divs]
            )
            ordered = east_divs + west_divs
        else:
            ordered = sorted(divisions)

        cols_per_row = 3
        for i in range(0, len(ordered), cols_per_row):
            row_divs = ordered[i:i + cols_per_row]
            cols = st.columns(len(row_divs))
            for col, div in zip(cols, row_divs):
                with col:
                    div_df = s[s[div_col] == div].copy()
                    if seed_col and seed_col in div_df.columns:
                        div_df = div_df.sort_values(seed_col)
                    div_df = div_df[display_cols].reset_index(drop=True)
                    div_df.index = div_df.index + 1
                    st.markdown(f"**{div}**")
                    st.dataframe(
                        div_df,
                        column_config=_col_config(div_df),
                        use_container_width=True,
                        height=min(220, 36 + len(div_df) * 35),
                    )

add_betting_oracle_footer()
