# Outstanding Tasks — Status

> Last audited: April 12, 2026.
> Items verified against actual source files unless noted otherwise.

---

## ✅ Completed — Verified in Codebase

### P0 — Core odds / model integration
- [x] `expected_value()`, `kelly_criterion()`, `american_to_decimal()`, `implied_probability()` in `utils/data_fetcher.py`
- [x] Odds-derived feature columns (`OU_LINE`, `IMPLIED_PROB_HOME`, `spread_consensus`, etc.) in `utils/model_utils.py` (`FEATURE_COLS_GAME_ODDS`)
- [x] EV cards, Kelly sizing, O/U predictions, multi-book odds comparison table in `pages/1_Game_Predictions.py`
- [x] Isotonic calibration + held-out calibration set in `scripts/train_models.py`
- [x] Calibration curve, ROI by confidence tier, cumulative P&L in `pages/6_Model_Performance.py`
- [x] `config/seasons.toml` controls season date ranges (no hardcoded constants)
- [x] `scripts/daily_update.py` orchestrates daily pipeline refresh

### P1 — Multi-sportsbook / training
- [x] `sbrscrape>=0.0.12` in `requirements.txt`
- [x] `get_multi_book_odds()` and `get_best_lines()` in `utils/data_fetcher.py`
- [x] `scripts/fetch_historical_odds.py` backfills odds via sbrscrape + Odds API fallback
- [x] Walk-forward `TimeSeriesSplit` CV in `scripts/train_models.py`
- [x] Odds features included in training via `FEATURE_COLS_GAME_ODDS`

### P2 — UI / analytics features
- [x] Line shopping display (multi-book odds comparison per game with best-line highlighting)
- [x] ROI by confidence tier chart in `pages/6_Model_Performance.py`
- [x] Cumulative P&L chart in `pages/6_Model_Performance.py`
- [x] Kelly sizing shown per-game in EV/Kelly expander
- [x] **Bankroll Manager** section added to `pages/1_Game_Predictions.py` — aggregates all +EV bets, Quarter-Kelly bet sizes, total risk and expected profit (added April 2026)
- [x] Pick 6 page: `predict_player_prop()` returns `over_probability`, `confidence`, `direction`; ranked recommendations in Top Picks tab

### P3 — Maintenance / documentation
- [x] No emoji in page filenames (`1_Game_Predictions.py`, `3_Standings.py`, etc.)
- [x] `st.set_page_config()` called exactly once in `predictions.py`; `st.navigation()` configured with icons
- [x] `docs/understanding.md` updated: correct page filenames, expanded data sources, updated Key Features
- [x] **Line movement tracking** implemented (April 2026):
  - `snapshot_odds()` in `utils/data_fetcher.py` — appends a timestamped row to `data_files/odds_snapshots.parquet`
  - `get_line_movement(home, away, date)` in `utils/data_fetcher.py` — reads snapshots and returns sorted history
  - `scripts/daily_update.py --snapshot` — fast mode (~5 s) for frequent intra-day scheduling
  - Line movement displayed in the 📊 Multi-Book Odds expander per game card (spread, total, home/away ML with arrows and deltas)

---

## 🔲 Open — Not Yet Implemented

*None.* All documented tasks are complete.

To activate line movement tracking, schedule the snapshot command:
```
# Every 30 minutes from 10 AM to 11 PM (Windows Task Scheduler or cron)
*/30 10-23 * * * /path/to/venv/bin/python /path/to/nba-predictions/scripts/daily_update.py --snapshot
```

---

## Notes

- All P0–P3 items are verified present in the codebase.
- The Bankroll Manager (P2) was added in April 2026.
- Line movement tracking (P3) was added in April 2026; activate by scheduling `--snapshot` runs.
- If a task is implemented in the future, move it from Open to Completed and note the date.
