# Betting Baseline — NBA Predictions — GitHub Copilot Instructions

## Project Overview

**App name:** Betting Baseline
**Purpose:** NBA game predictions and betting analytics platform. Predicts totals, moneylines, and player props. Surfaces value bets vs. DraftKings lines.
**Entry point:** `streamlit run predictions.py`
**Part of:** Betting Oracle suite

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (single-page, tabbed) |
| ML | XGBoost, scikit-learn (totals + moneyline models) |
| Data | pandas, hoopr (NBA stats), ESPN API |
| Odds | The Odds API |
| Config | python-dotenv (`.env` file) |
| Python | 3.9+ |

---

## File Conventions

### Key files
- `predictions.py` — entry point; sets `st.set_page_config` ONCE. Calls `home_page()` and `add_betting_oracle_footer()`.
- `utils/data_fetcher.py` — `get_today_predictions()`, `get_multi_book_odds()`, `get_best_lines()`, `expected_value()`, `kelly_criterion()`, `CURRENT_SEASON`.
- `utils/model_utils.py` — `load_eval_metrics()`, `load_totals_model()`.
- `utils/prediction_engine.py` — `predict_total_points()`.
- `utils/feature_engine.py` — `engineer_team_features()`.
- `utils/hoopr_fetcher.py` — NBA stats via hoopr library.
- `footer.py` — `add_betting_oracle_footer()` called at page bottom.

### Data files
- `data_files/logo.png` — app logo
- `data_files/best_bets_today.json` — unified schema for Sports Picks Grid aggregator

---

## NBA Domain Knowledge

### Confidence tiers
- `High` — strong model agreement, high edge
- `Medium` — moderate edge
- `Low` — low edge, informational only

### Bet types
- `total` — over/under total points
- `moneyline` — outright game winner
- `player_prop` — individual player stat bets (yards, points, assists)

### Key metrics
- `expected_value(model_prob, dk_odds)` — EV per $1 staked
- `kelly_criterion(model_prob, dk_odds)` — fractional Kelly bet size
- Eastern timezone is authoritative for game scheduling (`ZoneInfo("America/New_York")`)

---

## Coding Conventions

### Streamlit patterns
```python
@st.cache_data(ttl=3600)
def load_something() -> pd.DataFrame: ...
```
- `st.set_page_config()` called ONCE in `predictions.py` only — NEVER in sub-pages or utility files
- Use `width='stretch'` for dataframes/charts (not deprecated `use_container_width`)
- Load all heavy data in `home_page()`, not at module level

### Security
- API keys via `python-dotenv`; never hardcode; `.env` is gitignored
- Guard all numeric formatting against None/NaN

### Error handling
- Wrap `get_today_predictions()` in try/except; fall back to empty DataFrame
- Always check DataFrame emptiness before rendering tables

---

## Export for Sports Picks Grid

Maintain `scripts/export_best_bets.py` to write `data_files/best_bets_today.json`. Uses unified schema:
```json
{"meta": {"sport": "NBA", ...}, "bets": [{"bet_type": "moneyline|total|player_prop", ...}]}
```
Run: `python scripts/export_best_bets.py`
