# Streamlit Site Layout

## Status
- [x] Complete

This document describes the page structure, navigation, and UX design for the NBA Predictions Streamlit application.

---

## Site Architecture

```
predictions.py (Home / Landing Page)
├── pages/
│   ├── 1_Game_Predictions.py
│   ├── 2_Pick_6.py
│   ├── 3_Team_Stats.py
│   ├── 4_Player_Stats.py
│   └── 5_Model_Performance.py
```

Streamlit's native multi-page app structure is used. Files in `pages/` automatically appear in the sidebar navigation.

---

## Page Descriptions

### Home Page (`predictions.py`)

The landing page serves as the dashboard and entry point.

**Components:**
- **Logo & Title**: Site logo (200px width) + "NBA Predictions" header
- **Today's Games**: Cards showing today's matchups with quick win probability indicators
- **Quick Pick 6**: Top-confidence Pick 6 recommendations for today
- **Model Status**: Last updated timestamp, current model accuracy metrics
- **Recent Results**: How recent predictions performed (last 3 days)

**Layout:**
```
┌──────────────────────────────────────────────┐
│  [Logo]  NBA Predictions                     │
│  Statistical analyses & predictions          │
├──────────────────────────────────────────────┤
│                                              │
│  📊 Today's Games                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │ LAL @ BOS│ │ GSW @ MIL│ │ DEN @ PHX│     │
│  │ 45% / 55%│ │ 52% / 48%│ │ 60% / 40%│     │
│  └──────────┘ └──────────┘ └──────────┘     │
│                                              │
│  ┌─────────────────┐ ┌────────────────────┐  │
│  │ Quick Pick 6    │ │ Recent Results     │  │
│  │ • LeBron > 25.5 │ │ Yesterday: 4/5     │  │
│  │ • Tatum < 8.5r  │ │ 2 days ago: 3/5    │  │
│  │ • Jokic > 10.5a │ │ Season: 62% acc    │  │
│  └─────────────────┘ └────────────────────┘  │
└──────────────────────────────────────────────┘
```

---

### Game Predictions Page (`pages/1_Game_Predictions.py`)

Detailed game-by-game predictions with full analysis.

**Sidebar Controls:**
- Date picker (default: today)
- Conference filter (All / East / West)

**Main Content:**
- **Game Cards**: Each game gets an expandable card with:
  - Team logos and names
  - Win probability bar (horizontal, team colors)
  - Predicted spread and total
  - Key matchup factors (top 5 features driving the prediction)
  - Recent form for both teams (last 5 games W/L)
  - Head-to-head record this season
- **Confidence Indicator**: High / Medium / Low confidence badge per game
- **Historical Accuracy**: "Our model has been X% accurate on similar confidence games"

**Visualizations:**
- Win probability gauge chart (Plotly)
- Team comparison radar chart (off rating, def rating, pace, 3PT%, rebounding)
- Rolling net rating trend lines (Plotly line chart)

---

### Pick 6 Page (`pages/2_Pick_6.py`)

DraftKings Pick 6 analysis and recommendations.

**Sidebar Controls:**
- Date picker
- Number of picks (2-6)
- Risk tolerance slider (Conservative / Balanced / Aggressive)
- Stat category filter (Points, Rebounds, Assists, 3PM, All)

**Main Content:**
- **Recommended Picks Table**:
  | Player | Stat | Line | Prediction | Confidence | Direction |
  |--------|------|------|------------|------------|-----------|
  | LeBron | PTS  | 25.5 | 27.8      | 72%        | MORE ▲    |
- **Pick Builder**: Interactive tool to build custom Pick 6 entries
  - Select players from dropdown
  - See individual confidence scores
  - See combined probability of the entry
  - Correlation warnings (e.g., "Two players in the same game — correlated risk")
- **Today's Board Analysis**: Full analysis of every available prop on the Pick 6 board
- **Payout Calculator**: Expected value based on confidence and pick count

**Visualizations:**
- Player stat distribution histogram with prop line overlay
- Over/under rate chart (last N games)
- Matchup difficulty heatmap (player vs opponent defense)

---

### Team Stats Page (`pages/3_Team_Stats.py`)

Interactive team statistics explorer.

**Sidebar Controls:**
- Team selector (multi-select)
- Season selector
- Stat category tabs (Offense, Defense, Advanced, Four Factors)
- Rolling window slider (5 / 10 / 15 / 20 / Full Season)

**Main Content:**
- **Team Dashboard**: Selected team's key stats at a glance
- **Comparison Mode**: Side-by-side comparison of 2-4 teams
- **League Rankings**: Sortable table of all 30 teams by selected stat
- **Trend Charts**: Rolling averages over the season

**Visualizations:**
- Team stats radar chart (Plotly)
- Rolling average line charts (offensive rating, defensive rating, net rating)
- Shot chart / scoring distribution (paint vs mid-range vs 3PT)
- Four Factors bar chart comparison

---

### Player Stats Page (`pages/4_Player_Stats.py`)

Individual player analysis and prop research tool.

**Sidebar Controls:**
- Player search (typeahead)
- Season selector
- Compare players toggle
- Game log length (Last 5 / 10 / 20 / Full Season)

**Main Content:**
- **Player Card**: Photo, team, position, key season averages
- **Game Log Table**: Sortable, filterable game log
- **Stat Trends**: Rolling averages for major stat categories
- **Splits Analysis**: Home/Away, vs conference, days rest, back-to-back
- **Prop Research**: Historical performance vs common prop lines

**Visualizations:**
- Points/rebounds/assists rolling average line charts
- Game log scatter plot with trend line
- Splits comparison bar charts
- Distribution histogram for each stat category

---

### Model Performance Page (`pages/5_Model_Performance.py`)

Transparency and accountability — show how models are performing.

**Sidebar Controls:**
- Date range selector
- Model selector (if multiple models)
- Prediction type (Game Outcome / Spread / Total / Pick 6)

**Main Content:**
- **Accuracy Metrics Dashboard**:
  - Overall accuracy (%)
  - Accuracy by confidence tier (High / Medium / Low)
  - Log loss and Brier score
  - Calibration score
- **ROI Tracker**: If betting at model-recommended confidence thresholds
- **Backtesting Results**: Walk-forward validation performance by month
- **Feature Importance**: Top features driving predictions

**Visualizations:**
- Calibration plot (predicted probability vs actual win rate)
- Accuracy over time (rolling 30-day window)
- Confusion matrix heatmap
- Feature importance horizontal bar chart
- ROI cumulative line chart

---

## Shared UI Components

### Sidebar (Global)
- Site logo (smaller, ~100px)
- Navigation links (auto-generated by Streamlit multi-page)
- "Last Updated" timestamp
- Quick links: GitHub repo, DraftKings Pick 6

### Styling & Theme
- **Streamlit Config** (`.streamlit/config.toml`):
  ```toml
  [theme]
  primaryColor = "#1D428A"      # NBA blue
  backgroundColor = "#FFFFFF"
  secondaryBackgroundColor = "#F0F2F6"
  textColor = "#1D1D1D"
  font = "sans serif"
  ```
- Use `st.columns()` for responsive layouts
- Use `st.expander()` for detailed analysis sections
- Use `st.metric()` for key stats with delta indicators
- Use `st.tabs()` to organize related content without page navigation

---

## Streamlit Cloud Deployment Notes

### Configuration
- **Entry point**: `predictions.py` (root of repo)
- **Requirements**: `requirements.txt` in repo root
- **Python version**: Specify in `runtime.txt` if needed (e.g., `python-3.11`)

### Performance Optimization
- **`st.cache_data`**: Cache all API calls and data transformations with appropriate TTL
  - Live scores: `ttl=300` (5 minutes)
  - Daily stats: `ttl=3600` (1 hour)
  - Historical data: `ttl=86400` (24 hours)
- **`st.cache_resource`**: Cache model loading (load once per session)
- **Data Size**: Keep datasets small — Streamlit Cloud has ~1GB memory limit
- **Lazy Loading**: Only fetch data when a user navigates to a page

### Secrets Management
- Store API keys in Streamlit Cloud Secrets (`.streamlit/secrets.toml` for local dev)
- Access via `st.secrets["ODDS_API_KEY"]`
- Never commit secrets to git

### File Structure for Deployment
```
nba-predictions/
├── .streamlit/
│   └── config.toml          # Theme configuration
├── data_files/
│   └── logo.png             # Site logo
├── docs/                    # Roadmap documentation
├── models/                  # Saved model files (pickle/joblib)
├── utils/                   # Shared utility functions
│   ├── data_fetcher.py      # nba_api wrapper functions
│   ├── feature_engine.py    # Feature engineering pipeline
│   └── model_utils.py       # Model loading and prediction
├── pages/                   # Streamlit multi-page app
│   ├── 1_Game_Predictions.py
│   ├── 2_Pick_6.py
│   ├── 3_Team_Stats.py
│   ├── 4_Player_Stats.py
│   └── 5_Model_Performance.py
├── predictions.py           # Main entry point
├── requirements.txt
└── README.md
```
