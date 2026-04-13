# Understanding the NBA Predictions Repository

## Overview

The NBA Predictions repository is a Streamlit-powered data analytics platform for providing insights into NBA game predictions and DraftKings Pick 6 analysis. The repository leverages various data sources, advanced statistical models, and machine learning techniques to deliver comprehensive analytics.

## Key Features

- **Game Predictions**: Win probabilities, predicted spreads & totals, O/U model, confidence tiers.
- **Expected Value & Kelly Sizing**: EV per $100 and Quarter-Kelly bet sizing for every available line.
- **Multi-Book Odds**: Live odds comparison across FanDuel, DraftKings, BetMGM, Caesars, and more via sbrscrape.
- **Bankroll Manager**: Aggregated +EV bets for the day with Quarter-Kelly bet sizes and total risk summary.
- **Pick 6 Analysis**: Player prop modeling with over/under probabilities, confidence tiers, and entry builder.
- **Team Stats**: Interactive dashboards with rolling averages, four factors, and league rankings.
- **Player Stats**: Individual player analysis, game logs, splits, and trend charts.
- **Model Performance**: Accuracy, calibration curve, ROI by confidence tier, and cumulative P&L chart.

## Data Sources

The repository integrates several data sources to ensure comprehensive analytics:

1. **nba_api** - Primary source for game data, box scores, player stats, and league standings.
2. **sbrscrape** - Live multi-sportsbook odds (FanDuel, DraftKings, BetMGM, Caesars, etc.) with The Odds API as fallback.
3. **The Odds API** - Fallback odds source; also used for historical odds backfilling.
4. **nbastuffer** - Advanced player/team stats, rest days, referee stats (scraped via `scripts/scrape_external.py`).
5. **databallr** - Shot quality and advanced team metrics (scraped via `scripts/scrape_external.py`).
6. **ESPN** - Injury reports via `site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`.

## Tech Stack

- **App Framework**: Streamlit
- **Data Tools**: nba_api, BeautifulSoup4, requests
- **Analysis Tools**: pandas, NumPy, SciPy, statsmodels
- **ML Models**: scikit-learn, XGBoost, LightGBM
- **Visualization Tools**: Plotly, Matplotlib, Seaborn

## Key Files and Responsibilities

- `scripts/scrape_external.py`: Scrapes nbastuffer, databallr, and referee data.
- `scripts/preload_cache.py`: Pre-warms historical parquet cache and runs predictions.
- `scripts/daily_update.py`: Orchestrates full daily data refresh pipeline.
- `scripts/fetch_historical_odds.py`: Backfills historical odds via sbrscrape / The Odds API.
- `scripts/train_models.py`: Trains all models with walk-forward CV and isotonic calibration.
- `utils/data_fetcher.py`: Data layer cache, prediction retrieval, injury report parsing, EV/Kelly functions.
- `utils/prediction_engine.py`: Game outcome + O/U + player prop prediction pipelines.
- `utils/model_utils.py`: Model definitions, Elo system, ensemble, feature columns.
- `utils/feature_engine.py`: Feature engineering for games and player props.
- `pages/1_Game_Predictions.py`: Main prediction UI, EV cards, Kelly sizing, O/U, multi-book odds, Bankroll Manager.
- `pages/2_Pick_6.py`: Player prop analysis and DK Pick 6 calculator.
- `pages/3_Standings.py`: League standings display.
- `pages/4_Team_Stats.py`: Team statistics and rolling average dashboards.
- `pages/5_Player_Stats.py`: Player game log, splits, and trend analysis.
- `pages/6_Model_Performance.py`: Model accuracy, calibration curve, ROI by tier, cumulative P&L.
- `config/seasons.toml`: Season date ranges and current season config.
- `.github/workflows/referee-assignments.yml`: Daily referee refresh schedule.
- `.github/workflows/nightly-pipeline.yml`: Nightly data pipeline.

## Important Instructions

1. **Referee Ingestion and Deduplication**:
   - `nba_official` is authoritative for `NBA_GAME_ID` / team names.
   - ESPN data serves as a fallback only for games missing from the official page; it is not used for same-game overwrites.
   - Referee output schema should include: `NBA_GAME_ID`, `HOME_TEAM`, `AWAY_TEAM`, `HOME_TEAM_FULL`, `AWAY_TEAM_FULL`, `REFEREE`, `ORDER`, `CALLED_FOULS_PER_GAME`, `HOME_WIN_PCT`, `FOUL_PCT_ROAD`, `FOUL_PCT_HOME`, `FOUL_DIFFERENTIAL`, `EXPERIENCE_YEARS`, `FETCH_DATE`, `SOURCE`.

2. **Injury Report**:
   - Source: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`
   - Uses `team_entry.get('displayName', '')` for team parsing, not nested `team_entry.get('team', {}).get('displayName', '')`.
   - Output columns: `team`, `player_name`, `position`, `status`, `description`.

3. **Workflow Drift Guards**:
   - Scheduled times must match Eastern (ET) requirements.
   - Add a `workflow_dispatch` hook with an optional `force` parameter for manual override.

4. **Verification Steps for PR Reviews**:
   - Confirm field names in relevant parquets using `python -c "import pandas as pd; ..."` before UI renames.
   - Verify that `get_injury_report()` returns non-empty `team` fields.
   - Ensure `scripts/preload_cache.py` runs end-to-end with `--no-preds / --preds-only` options.
   - Run `python -m pytest` or `get_errors` to verify no linting/type issues.

5. **Notes for Reviewers**:
   - Emphasize correctness over minimal scope; include explicit fallback behavior.
   - Update the README with major pipeline changes and bugfixes.
   - Maintain consistent column naming in UI across pages and align with user expectations (title case).

## Recent Updates

- Referee assignments now use `nba_official` as primary source with ESPN fallback, eliminating duplicates (e.g., J.T. Orr / JT Orr mismatch), and dropping `ESPN_GAME_ID` from schema.
- Added workflow `.github/workflows/referee-assignments.yml` scheduled at `0 16 * * *` (11 AM ET) and manual `workflow_dispatch`.
- Implemented `scripts/preload_cache.py` to prewarm parquet cache + run predictions; integrated into `.github/workflows/nightly-pipeline.yml`.
- Added `get_today_predictions()` and `run_and_cache_predictions()` in `utils/data_fetcher.py`.
- Added disk cache for `get_standings()` with same-day freshness guard in `utils/data_fetcher.py`.
- Fixed injury report team field parsing in `get_injury_report()` (`team_entry.displayName` instead of nested `team.displayName`).
- Moved injury report section to top-level in `pages/1_Game_Predictions.py` so it renders independently of prediction success.
- Verified and corrected column mapping in Streamlit UI:
  - `pages/3_Standings.py`: standings columns `HOME`, `ROAD`, `L10`, `strCurrentStreak` (real API fields), plus humanized labels.
  - `pages/5_Player_Stats.py`: renamed game log columns with confirmed fields (`GAME_DATE`, `MATCHUP`, `WL`, `FG_PCT`, etc.).
  - `pages/1_Game_Predictions.py`: injury report title-case columns and include team correctly.
- Added a robust data verification process to inspect actual parquet schema before renaming columns (avoid assumptions).

## Conclusion

This document provides an overview of the NBA Predictions repository, detailing its features, data sources, tech stack, and important instructions. It serves as a reference for understanding the repository structure and functionality.