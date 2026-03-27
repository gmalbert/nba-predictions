"""
Pick 6 Analysis page — NBA DraftKings Pick 6

Models DraftKings Pick 6 player props and builds recommended entries
for today's board based on historical performance and matchup context.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.data_fetcher import (
    get_player_game_log_cached,
    get_all_active_players,
    get_league_player_stats_cached,
    CURRENT_SEASON,
)
from utils.feature_engine import engineer_player_features, PROP_STAT_MAP
from utils.prediction_engine import predict_player_prop
from footer import add_betting_oracle_footer

# ── Constants ─────────────────────────────────────────────────────────────────

STAT_CATEGORIES = ["PTS", "REB", "AST", "3PM", "PRA", "STL", "BLK", "STL+BLK"]

# Secondary stat shown alongside the main stat in the game log (col_name, display_label)
_SECONDARY = {
    "PTS":     ("FG3M", "3PM"),
    "REB":     ("FG3M", "3PM"),
    "AST":     ("TOV",  "TOV"),
    "3PM":     ("PTS",  "PTS"),
    "PRA":     ("MIN",  "MIN"),
    "STL":     ("AST",  "AST"),
    "BLK":     ("REB",  "REB"),
    "STL+BLK": ("MIN",  "MIN"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _df_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    h = len(df) * row_height + header_height + padding
    return min(h, max_height) if max_height else h


def _show_leaders(df, stat, label, top_n):
    """Display top-N players for a given per-game stat from league stats."""
    extra_cols = {
        "PTS": ["FG3M", "FG_PCT"],
        "REB": ["OREB", "DREB"],
        "AST": ["TOV", "AST_TOV"],
    }.get(stat, [])

    needed   = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", stat]
    all_cols = needed + [c for c in extra_cols if c in df.columns]
    avail    = [c for c in all_cols if c in df.columns]

    top = df[avail].sort_values(stat, ascending=False).head(top_n).copy()
    top.insert(0, "#", range(1, len(top) + 1))

    rename = {
        "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team",
        "GP": "GP", "MIN": "MIN", stat: label,
        "FG3M": "3PM", "FG_PCT": "FG%",
        "OREB": "OREB", "DREB": "DREB",
        "TOV": "TOV", "AST_TOV": "AST/TOV",
    }
    disp = top.rename(columns=rename)
    for c in disp.select_dtypes(include="float").columns:
        disp[c] = disp[c].round(1)

    st.dataframe(disp, width='stretch', hide_index=True, height=_df_height(disp))
    st.caption(f"Top {top_n} players by {label} · {CURRENT_SEASON} · Per Game")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.image("data_files/logo.png", width=200)
        st.markdown("---")
    
    st.title("🎯 DraftKings Pick 6 – NBA")
    st.markdown("---")

    # ── Load shared data ──────────────────────────────────────────────────────
    all_players = get_all_active_players()
    try:
        league_stats = get_league_player_stats_cached(CURRENT_SEASON)
    except Exception:
        league_stats = None

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⭐ Top Picks",
        "📊 DK Pick 6 Calculator",
        "🏀 Top Scorers",
        "📊 Top Rebounders",
        "🎯 Assists Leaders",
        "🔍 Player Search",
    ])

    # =========================================================================
    # TAB 1: Top Picks
    # =========================================================================
    with tab1:
        st.subheader("Top Player Prop Recommendations")

        if league_stats is None or league_stats.empty:
            st.warning("⚠️ League stats not available. Could not generate top picks.")
        else:
            # ── Filters ───────────────────────────────────────────────────────
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                pick_stat = st.selectbox(
                    "Prop Type",
                    ["PTS", "REB", "AST", "FG3M"],
                    key="top_picks_stat",
                )
            with fc2:
                min_gp = st.slider("Min Games Played", 10, 50, 20, key="top_picks_gp")
            with fc3:
                top_n_picks = st.slider("Players to show", 10, 50, 25, key="top_picks_n")
            with fc4:
                sort_opt = st.selectbox(
                    "Sort By",
                    ["Season Avg ↓", "Consistency ↓", "GP ↓"],
                    key="top_picks_sort",
                )

            st.markdown("---")

            # ── Build candidates table ─────────────────────────────────────
            stat_labels = {"PTS": "Points", "REB": "Rebounds", "AST": "Assists", "FG3M": "3-Pointers"}
            stat_label  = stat_labels.get(pick_stat, pick_stat)

            needed = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "W_PCT", pick_stat]
            # Add FG_PCT for PTS, extra context cols
            extra = {"PTS": ["FG_PCT", "FG3M"], "REB": ["OREB", "DREB"], "AST": ["TOV"], "FG3M": ["PTS"]}.get(pick_stat, [])
            avail_cols = [c for c in needed + extra if c in league_stats.columns]

            df_picks = (
                league_stats[avail_cols]
                .copy()
                .query("GP >= @min_gp")
            )

            if df_picks.empty:
                st.info(f"No players with ≥{min_gp} games. Lower the minimum games filter.")
            else:
                # Compute a DK-style suggested line (~85% of season avg rounded to nearest 0.5)
                df_picks["_avg"]  = pd.to_numeric(df_picks[pick_stat], errors="coerce")
                df_picks["_gp"]   = pd.to_numeric(df_picks["GP"], errors="coerce")
                df_picks["_wpct"] = pd.to_numeric(df_picks.get("W_PCT", 0.5), errors="coerce")
                df_picks = df_picks.dropna(subset=["_avg"]).copy()

                # Suggested DK line = floor to nearest .5 slightly below season avg
                df_picks["Sug. Line"] = (df_picks["_avg"] * 0.90).apply(
                    lambda x: round(x * 2) / 2  # round to nearest 0.5
                )

                # Tier: based on how far season avg is above suggested line
                def _tier(avg, line):
                    ratio = avg / line if line > 0 else 1.0
                    if ratio >= 1.20:   return "🔥 ELITE"
                    if ratio >= 1.12:   return "💪 STRONG"
                    if ratio >= 1.05:   return "✅ GOOD"
                    return "⚠️ LEAN"

                df_picks["Tier"] = df_picks.apply(
                    lambda r: _tier(r["_avg"], r["Sug. Line"]), axis=1
                )

                # Sort
                if sort_opt == "Season Avg ↓":
                    df_picks = df_picks.sort_values("_avg", ascending=False)
                elif sort_opt == "Consistency ↓":
                    df_picks = df_picks.sort_values("_wpct", ascending=False)
                else:
                    df_picks = df_picks.sort_values("_gp", ascending=False)

                df_picks = df_picks.head(top_n_picks).copy()
                df_picks.insert(0, "#", range(1, len(df_picks) + 1))

                # Confidence tier metrics
                t1, t2, t3 = st.columns(3)
                t1.metric("🔥 Elite",  len(df_picks[df_picks["Tier"] == "🔥 ELITE"]))
                t2.metric("💪 Strong", len(df_picks[df_picks["Tier"] == "💪 STRONG"]))
                t3.metric("✅ Good",   len(df_picks[df_picks["Tier"] == "✅ GOOD"]))

                st.markdown("---")

                # Build display df
                rename = {
                    "PLAYER_NAME":        "Player",
                    "TEAM_ABBREVIATION":  "Team",
                    "GP":   "GP",
                    "MIN":  "MIN",
                    "W_PCT": "W%",
                    pick_stat: stat_label,
                    "FG_PCT": "FG%",
                    "FG3M":   "3PM",
                    "OREB":   "OREB",
                    "DREB":   "DREB",
                    "TOV":    "TOV",
                    "PTS":    "PTS",
                }
                show = ["#", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", pick_stat,
                        "Sug. Line", "Tier"] + [c for c in extra if c in df_picks.columns]
                disp = df_picks[[c for c in show if c in df_picks.columns]].rename(columns=rename)
                for c in disp.select_dtypes(include="float").columns:
                    disp[c] = disp[c].round(1)

                st.dataframe(disp, width='stretch', hide_index=True,
                             height=_df_height(disp))

                st.caption(
                    f"Top {len(df_picks)} {stat_label} candidates · {CURRENT_SEASON} season · "
                    "Suggested Line = 90% of season avg rounded to nearest 0.5 · "
                    "Verify against actual DraftKings lines in the 📊 **DK Pick 6 Calculator** tab."
                )

                with st.expander("ℹ️ How to use this table"):
                    st.markdown("""
**Tier calculation** — based on how much the player's season average exceeds the suggested line:

| Tier | Ratio (avg ÷ line) | Meaning |
|------|-------------------|---------|
| 🔥 ELITE  | ≥ 1.20× | Player consistently smashes this type of line |
| 💪 STRONG | 1.12–1.20× | Very likely to hit |
| ✅ GOOD   | 1.05–1.12× | Solid play above breakeven |
| ⚠️ LEAN   | < 1.05× | No strong edge at this line level |

**Suggested Line** is a conservative estimate (~90% of season avg). The actual DraftKings
line may be higher or lower — always check the real line in the **📊 DK Pick 6 Calculator** tab.
                    """)

    # =========================================================================
    # TAB 2: DK Pick 6 Calculator
    # =========================================================================
    with tab2:
        st.subheader("📊 DraftKings Pick 6 – Line Comparison")
        st.markdown("Enter a player and their DraftKings Pick 6 line to get the model's prediction")

        input_col1, input_col2 = st.columns([2, 1])

        # ── Player search ─────────────────────────────────────────────────────
        with input_col1:
            player_search = st.text_input(
                "🔍 Search Player",
                placeholder="Type player name (e.g., LeBron James, Nikola Jokic)",
                key="dk_player_search",
            )

        selected_player_id   = None
        selected_player_full = None

        if player_search:
            search_lower = player_search.lower()
            matches = all_players[
                all_players["full_name"].str.lower().str.contains(search_lower, na=False)
            ].copy()

            if not matches.empty:
                matches["exact_match"] = (
                    matches["full_name"].str.lower() == search_lower
                )
                matches = matches.sort_values(
                    ["exact_match", "full_name"], ascending=[False, True]
                )
                player_opts = {
                    row["full_name"]: int(row["id"])
                    for _, row in matches.head(20).iterrows()
                }
                with input_col1:
                    selected_display = st.selectbox(
                        "Select Player",
                        list(player_opts.keys()),
                        key="dk_player_select",
                    )
                selected_player_id   = player_opts[selected_display]
                selected_player_full = selected_display
            else:
                with input_col1:
                    st.info("No players found. Try a different search.")

        # ── Stat category ─────────────────────────────────────────────────────
        with input_col2:
            stat_category = st.selectbox(
                "Stat Category",
                STAT_CATEGORIES,
                key="dk_stat_category",
            )

        # ── Line input + analysis ─────────────────────────────────────────────
        if selected_player_id and stat_category:
            dk_line = st.number_input(
                f"DraftKings Pick 6 Line for {stat_category}",
                min_value=0.5,
                max_value=200.0,
                value=20.5,
                step=0.5,
                key="dk_line_value",
                help="Enter the exact over/under line from DraftKings Pick 6",
            )

            with st.spinner("Loading player stats…"):
                try:
                    game_log = get_player_game_log_cached(selected_player_id, CURRENT_SEASON)
                except Exception:
                    game_log = pd.DataFrame()

            if game_log is not None and not game_log.empty:
                df = engineer_player_features(game_log).sort_values("GAME_DATE")
                stat_col = PROP_STAT_MAP.get(stat_category.upper(), stat_category)

                if stat_col not in df.columns:
                    st.info(f"No {stat_category} data available for this player this season.")
                else:
                    recent = df.tail(10)

                    last_3_avg  = float(recent.tail(3)[stat_col].mean())
                    last_5_avg  = float(recent.tail(5)[stat_col].mean())
                    last_10_avg = float(recent[stat_col].mean())
                    season_avg  = float(df[stat_col].mean())

                    total_games = len(recent)
                    games_over  = int((recent[stat_col] > dk_line).sum())
                    games_under = total_games - games_over

                    # ── Model prediction ──────────────────────────────────────
                    model_result = predict_player_prop(
                        player_id=selected_player_id,
                        stat_cat=stat_category,
                        line=dk_line,
                        season=CURRENT_SEASON,
                        is_home=1,
                    )

                    if "error" not in model_result:
                        prob_over         = model_result["over_probability"]
                        prob_under        = model_result["under_probability"]
                        prediction_source = "🤖 Machine Learning Model"
                    else:
                        prob_over         = (games_over + 1) / (total_games + 2)
                        prob_under        = 1.0 - prob_over
                        prediction_source = "📊 Historical"

                    if prob_over >= prob_under:
                        recommendation = "MORE"
                        confidence     = prob_over
                    else:
                        recommendation = "LESS"
                        confidence     = prob_under

                    tier = (
                        "🔥 ELITE"  if confidence >= 0.65 else
                        "💪 STRONG" if confidence >= 0.60 else
                        "✅ GOOD"   if confidence >= 0.55 else
                        "⚠️ LEAN"
                    )

                    # ── Gradient hero card ────────────────────────────────────
                    st.markdown(f"""
<div style="background: linear-gradient(135deg, #1D428A 0%, #C8102E 100%);
            padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
    <h2 style="margin: 0; font-size: 1.8rem;">{selected_player_full}</h2>
    <p style="margin: 0.5rem 0; font-size: 1.1rem; opacity: 0.9;">{stat_category}</p>
    <div style="display: flex; justify-content: space-between;
                align-items: center; margin-top: 1.5rem;">
        <div>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">DraftKings Line</p>
            <p style="margin: 0; font-size: 2.5rem; font-weight: bold;">{dk_line}</p>
        </div>
        <div style="text-align: right;">
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Recommendation</p>
            <p style="margin: 0; font-size: 2.5rem; font-weight: bold;">{recommendation}</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{tier}</p>
        </div>
    </div>
    <div style="margin-top: 1.5rem; padding-top: 1.5rem;
                border-top: 1px solid rgba(255,255,255,0.2);">
        <p style="margin: 0; font-size: 1rem;">
            Confidence: <strong>{confidence:.1%}</strong> ({prediction_source})
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
            Historical: {total_games} games ({games_over} over, {games_under} under)
        </p>
    </div>
</div>
                    """, unsafe_allow_html=True)

                    # ── Performance Analysis ──────────────────────────────────
                    st.markdown("### 📈 Performance Analysis")

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        st.metric(
                            "Last 3 Games Avg", f"{last_3_avg:.1f}",
                            f"{last_3_avg - dk_line:+.1f} vs line",
                            delta_color="normal",
                        )
                    with mc2:
                        st.metric(
                            "Last 5 Games Avg", f"{last_5_avg:.1f}",
                            f"{last_5_avg - dk_line:+.1f} vs line",
                            delta_color="normal",
                        )
                    with mc3:
                        st.metric(
                            "Last 10 Games Avg", f"{last_10_avg:.1f}",
                            f"{last_10_avg - dk_line:+.1f} vs line",
                            delta_color="normal",
                        )
                    with mc4:
                        st.metric(
                            "Season Average", f"{season_avg:.1f}",
                            f"{season_avg - dk_line:+.1f} vs line",
                            delta_color="normal",
                        )

                    # ── Recent Game Log ───────────────────────────────────────
                    st.markdown("### 📊 Recent Game Log")

                    sec_col, sec_label = _SECONDARY.get(stat_category, ("MIN", "MIN"))
                    build_cols = ["GAME_DATE", "MATCHUP", stat_col]
                    if sec_col in df.columns and sec_col != stat_col:
                        build_cols.append(sec_col)

                    gl_disp = recent[build_cols].copy()
                    gl_disp["Result"] = gl_disp[stat_col].apply(
                        lambda x: f"✅ OVER ({x:.1f})" if x > dk_line
                        else f"❌ UNDER ({x:.1f})"
                    )
                    rename_map = {
                        "GAME_DATE": "Date",
                        "MATCHUP":   "Opponent",
                        stat_col:    stat_category,
                    }
                    if sec_col in build_cols:
                        rename_map[sec_col] = sec_label

                    st.dataframe(
                        gl_disp.rename(columns=rename_map).sort_values(
                            "Date", ascending=False
                        ),
                        width='stretch',
                        hide_index=True,
                    )

                    # ── Hit Rate Analysis ─────────────────────────────────────
                    st.markdown("### 🎯 Hit Rate Analysis")

                    hc1, hc2, hc3 = st.columns(3)
                    with hc1:
                        l3 = recent.tail(3)
                        o3 = int((l3[stat_col] > dk_line).sum())
                        st.metric(
                            "Last 3 Games", f"{o3}/3 Over",
                            f"{o3/3*100:.0f}% hit rate" if len(l3) >= 3 else "—",
                        )
                    with hc2:
                        l5 = recent.tail(5)
                        o5 = int((l5[stat_col] > dk_line).sum())
                        st.metric(
                            "Last 5 Games", f"{o5}/5 Over",
                            f"{o5/5*100:.0f}% hit rate" if len(l5) >= 5 else "—",
                        )
                    with hc3:
                        r10 = (
                            f"{games_over/total_games*100:.0f}% hit rate"
                            if total_games > 0 else "—"
                        )
                        st.metric(
                            f"Last {total_games} Games",
                            f"{games_over}/{total_games} Over",
                            r10,
                        )

                    # ── How This Works ────────────────────────────────────────
                    with st.expander("ℹ️ How This Works"):
                        st.markdown(f"""
**{prediction_source}**

The model analyses this player's recent game log, fits a Normal distribution
to the last 10 games and applies contextual adjustments (home/away, pace, rest)
to estimate `P(stat > line)`.

**Confidence Tiers:**
- 🔥 **ELITE (≥65%)**: Highest confidence picks
- 💪 **STRONG (60–65%)**: Very confident picks
- ✅ **GOOD (55–60%)**: Solid picks above breakeven
- ⚠️ **LEAN (<55%)**: Lower confidence — use sparingly

**Pro Tips:**
- Recent form (L3 / L5) is weighted most heavily
- Compare the season average to the line for long-run context
- **MORE** = model expects player to *exceed* the line
- **LESS** = model expects player to *fall short* of the line
                        """)
            else:
                st.info("No game log data available for this player this season.")
        else:
            st.info("👆 Search for a player above to get started")
            with st.expander("📖 How to Use This Tool"):
                st.markdown("""
### Step-by-Step Guide:

1. **Search for a Player** — type the player's name in the search box
2. **Select from Results** — choose the correct player from the dropdown
3. **Pick a Stat Category** — PTS, REB, AST, 3PM, PRA, STL, BLK, or STL+BLK
4. **Enter the DraftKings Line** — input the exact over/under line from DK Pick 6
5. **Get Your Recommendation** — see the model's MORE / LESS prediction with confidence tier

### Example:

- Player: **Nikola Jokic**
- Stat: **PTS**
- DK Line: **26.5**
- Model says: **MORE 26.5** (💪 STRONG – 63% confidence)
                """)

    # =========================================================================
    # TAB 3: Top Scorers
    # =========================================================================
    with tab3:
        st.subheader(f"🏆 {CURRENT_SEASON} Season Leaders – Points")
        if league_stats is not None and not league_stats.empty:
            _show_leaders(league_stats, "PTS", "Points Per Game", 30)
        else:
            st.warning("⚠️ League stats not available.")

    # =========================================================================
    # TAB 4: Top Rebounders
    # =========================================================================
    with tab4:
        st.subheader(f"🏆 {CURRENT_SEASON} Season Leaders – Rebounds")
        if league_stats is not None and not league_stats.empty:
            _show_leaders(league_stats, "REB", "Rebounds Per Game", 30)
        else:
            st.warning("⚠️ League stats not available.")

    # =========================================================================
    # TAB 5: Assists Leaders
    # =========================================================================
    with tab5:
        st.subheader(f"🏆 {CURRENT_SEASON} Season Leaders – Assists")
        if league_stats is not None and not league_stats.empty:
            _show_leaders(league_stats, "AST", "Assists Per Game", 30)
        else:
            st.warning("⚠️ League stats not available.")

    # =========================================================================
    # TAB 6: Player Search
    # =========================================================================
    with tab6:
        st.subheader("Search Individual Player Stats")

        search_term6 = st.text_input(
            "Search for a player (type to filter)", "", key="player_search_filter"
        )
        player_list = all_players["full_name"].sort_values().tolist()
        pid_map     = all_players.set_index("full_name")["id"].to_dict()

        filtered6 = (
            [n for n in player_list if search_term6.lower() in n.lower()]
            if search_term6
            else player_list
        )

        if filtered6:
            selected_name6 = st.selectbox(
                "Select Player", filtered6, key="player_search_sel"
            )
            pid6 = int(pid_map.get(selected_name6, 0))

            if pid6:
                with st.spinner("Loading…"):
                    try:
                        gl6 = get_player_game_log_cached(pid6, CURRENT_SEASON)
                    except Exception:
                        gl6 = pd.DataFrame()

                if gl6 is not None and not gl6.empty:
                    df6 = engineer_player_features(gl6).sort_values(
                        "GAME_DATE", ascending=False
                    )
                    st.markdown(f"**{CURRENT_SEASON} Season – {selected_name6}**")
                    st.caption(f"{len(df6)} games played this season")

                    avg_cols  = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]
                    avail_avg = [c for c in avg_cols if c in df6.columns]
                    if avail_avg:
                        acols = st.columns(len(avail_avg))
                        for i, col in enumerate(avail_avg):
                            acols[i].metric(col, f"{df6[col].mean():.1f}")

                    st.markdown("---")

                    show_cols  = [
                        "GAME_DATE", "MATCHUP", "WL", "MIN",
                        "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV",
                    ]
                    show_avail = [c for c in show_cols if c in df6.columns]
                    st.dataframe(
                        df6[show_avail].rename(columns={
                            "GAME_DATE": "Date",
                            "MATCHUP":   "Opponent",
                            "WL":        "W/L",
                        }),
                        width='stretch',
                        hide_index=True,
                        height=_df_height(df6[show_avail]),
                    )
                else:
                    st.info("No game log data available for this player this season.")
        else:
            st.info("No players match your search.")

    add_betting_oracle_footer()


main()
