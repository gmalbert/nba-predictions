# NBA Predictions 🏀

A Streamlit-powered data analytics platform for NBA game predictions and DraftKings Pick 6 analysis.

## Features

- **Game Predictions**: Win probabilities, predicted spreads & totals for every NBA game
- **Pick 6 Analysis**: Player prop modeling with over/under probabilities and entry building tools
- **Team Stats**: Interactive team dashboards with rolling averages, rankings, and comparisons
- **Player Stats**: Individual player analysis, game logs, splits, and trend charts
- **Model Performance**: Accuracy tracking, calibration plots, and backtesting results

## Data Sources

- [**nba_api**](https://github.com/swar/nba_api) — Primary data source for game data, box scores, player stats, and league standings
- **Basketball Reference** — Advanced metrics, four factors, and historical data
- **The Odds API** — Market odds for model calibration and edge detection
- **ESPN / Rotowire** — Injury reports and lineup confirmations

## Tech Stack

| Category | Tools |
|----------|-------|
| **App Framework** | Streamlit |
| **Data** | nba_api, BeautifulSoup4, requests |
| **Analysis** | pandas, NumPy, SciPy, statsmodels |
| **ML Models** | scikit-learn, XGBoost, LightGBM |
| **Visualization** | Plotly, Matplotlib, Seaborn |

## Roadmap

See the [docs/](docs/) folder for detailed planning:

- [Data Sources](docs/data_sources.md) — nba_api endpoints, scraping targets, odds APIs
- [Features](docs/features.md) — Team, player, matchup, and Pick 6 feature engineering
- [Layout](docs/layout.md) — Streamlit page structure and UI design
- [Models](docs/models.md) — ML model selection, training, and evaluation
- [Predictions](docs/predictions.md) — Game outcome and Pick 6 prediction methodology

## Latest updates (2026-03-24)

- Referee assignments now use `nba_official` as primary source with ESPN fallback; eliminated duplicates (e.g. J.T. Orr / JT Orr mismatch), and dropped `ESPN_GAME_ID` from schema.
- Added workflow `.github/workflows/referee-assignments.yml` scheduled at `0 16 * * *` (11 AM ET) and manual `workflow_dispatch`.
- Implemented `scripts/preload_cache.py` to prewarm parquet cache + run predictions; integrated into `.github/workflows/nightly-pipeline.yml`.
- Added `get_today_predictions()` and `run_and_cache_predictions()` in `utils/data_fetcher.py`.
- Added disk cache for `get_standings()` with same-day freshness guard in `utils/data_fetcher.py`.
- Fixed injury report team field parsing in `get_injury_report()` (`team_entry.displayName` instead of nested `team.displayName`).
- Moved injury report section to top-level in `pages/1_Game_Predictions.py` so it renders independent of prediction success.
- Verified and corrected column mapping in Streamlit UI:
  - `pages/3_Team_Stats.py`: standings columns `HOME`, `ROAD`, `L10`, `strCurrentStreak` (real API fields), plus humanized labels.
  - `pages/4_Player_Stats.py`: renamed game log columns with confirmed fields (`GAME_DATE`, `MATCHUP`, `WL`, `FG_PCT`, etc.).
  - `pages/1_Game_Predictions.py`: injury report title-case columns and include team correctly.
- Added a robust data verification process to inspect actual parquet schema before renaming columns (avoid assumptions).