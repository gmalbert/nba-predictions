> **AI Onboarding Guide** — See also the project docs folder for detailed feature and model documentation.

# NBA Predictions — Site Summary

## What This App Does

Streamlit analytics platform predicting NBA game outcomes (spread, moneyline, totals) and DraftKings Pick 6 player prop opportunities using XGBoost and LightGBM models. Includes team stats dashboards, player analysis, standings, and model performance tracking. A nightly GitHub Actions pipeline refreshes data and retrains models.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Run the app
streamlit run predictions.py
```

Data is fetched nightly by GitHub Actions. For local development, run `python utils/data_fetcher.py` to populate the cache manually.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit ≥1.51 (multi-page) |
| ML | XGBoost 2.0+ + LightGBM 4.0+ (soft voting ensemble) |
| Data sources | nba_api, sportsdataverse hoopR, Basketball Reference |
| Data storage | Parquet (primary), JSON cache |
| Visualization | Plotly 5.18+, Matplotlib, Seaborn |
| Hyperparameter tuning | Optuna |
| Python | 3.11+ |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Dashboard landing page — hero metrics, top picks, accuracy cards |
| `pages/1_Game_Predictions.py` | Spread, moneyline, total predictions with edge calculations |
| `pages/2_Pick_6.py` | DraftKings player prop modeling and entry construction |
| `pages/3_Standings.py` | League standings with per-game stats |
| `pages/6_Model_Performance.py` | Accuracy tracking, calibration plots, backtesting |
| `utils/data_fetcher.py` | nba_api client, game scoreboard, standings, odds retrieval |
| `utils/feature_engine.py` | Team and play-by-play derived stats, rolling metrics |
| `utils/hoopr_fetcher.py` | sportsdataverse hoopR integration — pre-cached team/PBP features |
| `utils/model_utils.py` | XGBoost/LightGBM training, evaluation, feature importance |
| `utils/prediction_engine.py` | Live game predictions using trained models |

## Data Flow

1. **Game data**: `nba_api` → game scoreboard, box scores, standings, player stats → JSON cache
2. **Advanced features**: `sportsdataverse hoopR` → team box/PBP features → Parquet cache (daily)
3. **Feature engineering**: `utils/feature_engine.py` → rolling team stats, PBP metrics, opponent defense ranks
4. **Model training**: XGBoost + LightGBM → predictions (win %, spread, total) + Pick 6 props
5. **Live odds**: The Odds API → DraftKings lines → edge = model probability vs implied
6. **UI**: Streamlit reads Parquet → renders predictions and player prop recommendations

## GitHub Actions Workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| `nightly-pipeline.yml` | Nightly | Fetch games, train models, generate predictions |
| `hoopr-daily.yml` | Daily | Refresh hoopR team/PBP features |
| `odds-snapshot.yml` | Multiple/day | Capture DraftKings odds snapshots |
| `referee-assignments.yml` | Daily | Update official referee rosters |

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `ODDS_API_KEY` | The Odds API — DraftKings lines for calibration | Optional |

## Critical Conventions

- **Never** load data at module level — always use `@st.cache_data` (silent crash on Streamlit Cloud otherwise)
- Use `width='stretch'` for dataframes/charts — `use_container_width` is removed in newer Streamlit
- All data loading goes through `@st.cache_data` decorators in the utils modules
- `hoopr` integration added Dec 2025 — team box/PBP features are now cached daily

## Common Gotchas

- nba_api is unofficial and rate-limited — always add delay between bulk calls
- hoopR data may lag by 1 day for the most recent games
- Referee assignments: `nba_official` is the primary source with ESPN as fallback
- Empty DataFrames: check `.empty` before calling `.sort_values()` to avoid KeyError
