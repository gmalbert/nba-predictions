# Betting Baseline — NBA Predictions — Architecture

## Overview
NBA game predictions and betting analytics platform. Predicts totals, moneylines, and player props. Surfaces value bets vs. DraftKings lines using XGBoost models and the hoopr NBA stats library.

## Data Flow
```
hoopr (NBA stats)    The Odds API    ESPN API
        ↓                 ↓               ↓
utils/hoopr_fetcher.py  multi-book odds  game times
        ↓                 ↓
utils/feature_engine.py → engineer_team_features()
        ↓
XGBoost (totals model) + scikit-learn (moneyline model)
        ↓
utils/prediction_engine.py → predict_total_points()
        ↓
utils/data_fetcher.py → get_today_predictions(), expected_value(), kelly_criterion()
        ↓
predictions.py → home_page() (Streamlit, tabbed UI)
        ↓
scripts/export_best_bets.py → data_files/best_bets_today.json
```

## ML Models
- **Totals model**: XGBoost (`utils/model_utils.py → load_totals_model()`)
- **Moneyline**: Team win probability from pace-adjusted offensive/defensive ratings
- Eastern timezone is authoritative (`ZoneInfo("America/New_York")`)

## Confidence Tiers
| Tier | Threshold |
|------|-----------|
| High | Strong model agreement, high edge |
| Medium | Moderate edge |
| Low | Informational only |

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| hoopr | NBA historical + live stats | None (open library) |
| The Odds API | Multi-book lines, best odds | `ODDS_API_KEY` |
| ESPN API | Game schedule, scores | None (public) |

## Key Components
- `predictions.py` — entry, `st.set_page_config`, calls `home_page()` + `add_betting_oracle_footer()`
- `utils/data_fetcher.py` — `get_today_predictions()`, `get_multi_book_odds()`, `get_best_lines()`, `expected_value()`, `kelly_criterion()`, `CURRENT_SEASON`
- `utils/model_utils.py` — `load_eval_metrics()`, `load_totals_model()`
- `utils/prediction_engine.py` — `predict_total_points()`
- `utils/feature_engine.py` — `engineer_team_features()`
- `utils/hoopr_fetcher.py` — NBA stats via hoopr
- `footer.py` — `add_betting_oracle_footer()`
- `scripts/export_best_bets.py` — unified schema JSON writer

## Bet Types
- `total` — over/under total points
- `moneyline` — outright game winner
- `player_prop` — individual player stat bets

## Storage
- `data_files/logo.png` — app logo
- `data_files/best_bets_today.json` — Sports Picks Grid feed
- Model artifacts in `data_files/` or `models/`
