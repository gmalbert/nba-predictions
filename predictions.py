import streamlit as st
from footer import add_betting_oracle_footer

# --- Page Configuration (called ONCE here; sub-pages must NOT call set_page_config) ---
st.set_page_config(
    page_title="Betting Baseline - NBA Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def home_page():
    """Landing page / home content."""
    # --- Logo ---
    st.image("data_files/logo.png", width=200)

    # --- Title & Description ---
    st.title("Betting Baseline - NBA Predictions")
    st.subheader("Statistical Analysis & Game Predictions")

    st.markdown("""
Welcome to **Betting Baseline** — a data-driven platform for analyzing NBA games, 
predicting outcomes, and building smarter DraftKings Pick 6 entries.
""")

    st.divider()

    # --- Feature Overview ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Game Predictions")
        st.markdown("""
    - Win probabilities for every game
    - Predicted spreads & totals
    - Confidence-tiered picks
    - Key matchup factors
    """)

    with col2:
        st.markdown("### 🎯 Pick 6 Analysis")
        st.markdown("""
    - Player prop modeling (pts, reb, ast, 3PM)
    - Over/under probability for each prop
    - Correlation-aware entry builder
    - Expected value calculations
    """)

    with col3:
        st.markdown("### 📈 Stats & Trends")
        st.markdown("""
    - Team & player dashboards
    - Rolling averages & trend charts
    - Head-to-head comparisons
    - Splits analysis (home/away, rest days)
    """)

    st.divider()

    # --- Data & Methodology ---
    st.markdown("### How It Works")

    st.markdown("""
This platform combines data from the [NBA Stats API](https://github.com/swar/nba_api) 
with machine learning models to generate predictions for NBA games and player props.

**Data Sources:**
- **NBA API**: Real-time and historical game data, box scores, player stats, and league standings
- **Web Scraping**: Advanced metrics from Basketball Reference, injury reports, and lineup data
- **Odds Data**: Market lines for model calibration and edge detection

**Models:**
- Elo rating system for baseline team strength
- XGBoost & LightGBM for game outcome and player prop predictions
- Ensemble methods combining multiple model outputs
- Walk-forward validation to ensure realistic accuracy estimates

**Pick 6:**
- Individual models for each stat category (points, rebounds, assists, 3PM)
- Matchup-adjusted projections accounting for opponent defense
- Correlation analysis for smarter multi-pick entries
""")

    st.divider()

    # --- Quick Links ---
    st.markdown("### Quick Links")

    link_col1, link_col2, link_col3 = st.columns(3)

    with link_col1:
        st.markdown("🏀 [DraftKings Pick 6](https://pick6.draftkings.com/?sport=NBA)")

    with link_col2:
        st.markdown("📦 [nba_api GitHub](https://github.com/swar/nba_api)")

    with link_col3:
        st.markdown("💻 [Project Repo](https://github.com/gmalbert/nba-predictions)")

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
