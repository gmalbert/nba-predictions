# Additional Considerations — Architecture, Risk & Strategy

## Source
[kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) — architectural patterns, risk factors, and strategic considerations

---

## 1. Architecture Comparison

| Aspect | kyleskom | Our Site |
|--------|----------|----------|
| **Framework** | Flask web app + CLI | Streamlit |
| **Data storage** | SQLite (3 databases) | Parquet files |
| **ML training** | Standalone scripts | `scripts/train_models.py` |
| **Config** | `config.toml` | Hardcoded constants |
| **Prediction flow** | CLI → stdout → Flask parses | Direct function calls |
| **Deployment** | Local Flask | Streamlit (local or Cloud) |
| **Models** | XGBoost, NN, LR | XGBoost, LightGBM, RF, LR, Elo, Ensemble |
| **Feature scope** | Team stats × 2 + OU + rest | 31 base + 42 extended features |

**Takeaway:** Our architecture is more sophisticated (more models, richer features, better UI). Their strengths are in **odds integration**, **probability calibration**, and **betting math (EV/Kelly)** — which are the pieces to adopt.

---

## 2. Feature Set Comparison

### Their features (Create_Games.py):
All columns from `LeagueDashTeamStats` for home team + away team (concatenated), plus:
- `OU` (sportsbook total line)
- `Days-Rest-Home`, `Days-Rest-Away`

### Our features (model_utils.py):
**31 base features** including engineered metrics like:
- Win %, home/away win %
- Offensive/defensive ratings
- Net rating, pace, PIE
- FG%, 3PT%, FT%, TS%, EFG%
- Rebounds, assists, turnovers, steals, blocks
- Plus/minus
- Elo ratings
- Rest days, injury impact

**42 extended features** (with nbastuffer):
- All base features
- Offensive/defensive efficiency ranks
- Clutch metrics
- Pace ranks
- SOS (strength of schedule)

**Our advantage:** We use **engineered features** (ratios, differences, rankings) rather than raw stat columns. This is generally better for ML because:
1. Differential features (home_rating − away_rating) capture matchup context
2. Ranking features are robust to season-level shifts
3. Fewer features = less overfitting risk

**Their advantage:** Using the **OU line as a feature** is very powerful. The sportsbook total encodes all the information the books have about game pace, scoring ability, injuries, etc. We should add this.

---

## 3. Push Handling in Over/Under

**Their approach:** 3-class model: Under (0), Over (1), Push (2).

**Why this matters:** NBA totals set at integer or half-point lines:
- **Half-point lines (214.5):** Push is impossible → 2-class is fine
- **Integer lines (215):** Push happens ~2-3% of the time

**Recommendation:** Use a 2-class model (Over/Under) for half-point lines and handle pushes separately:
- If the line is X.5, predict Over vs Under (2-class)
- If the line is integer X, still predict Over vs Under (2-class) but note push possibility
- Alternative: Always use 3-class but weight the push class appropriately

The 2-class approach is simpler and more practical since most modern sportsbooks use half-point lines.

---

## 4. Model Retraining Strategy

**Their approach:** Retrain from scratch at the start of each season using all historical data.

**Our approach:** Train once with `scripts/train_models.py`, retrain manually.

**Recommended strategy:**

| Trigger | Action |
|---------|--------|
| Season start | Full retrain with all historical data |
| Monthly (mid-season) | Incremental retrain adding new games |
| Model drift detected (accuracy drops below 58%) | Emergency retrain |
| New features added | Full retrain + evaluate improvement |

Create `scripts/retrain_check.py`:
```python
"""Check if model needs retraining based on recent accuracy."""
def should_retrain(recent_accuracy, threshold=0.58, window=50):
    if recent_accuracy < threshold:
        print(f"⚠️ Accuracy {recent_accuracy:.1%} below {threshold:.0%} over last {window} games")
        return True
    return False
```

---

## 5. Legal & Ethical Considerations

### Odds Scraping
- **`sbrscrape`** scrapes SBR (ScoresAndOdds.com / SBR Odds) — a publicly available odds comparison site
- This is **gray area legally** — SBR displays odds publicly but may have ToS against automated scraping
- **Mitigation:** Rate-limit requests (1 per second), don't hammer during peak times, have fallback to The Odds API
- **Alternative:** The Odds API is explicitly designed for programmatic access (we already use it for DraftKings)

### API Key Management
- The Odds API key is currently in our code — ensure it's in `.env` or `st.secrets`, never committed to git
- `sbrscrape` requires no API key (advantage)

### Responsible Gambling
- Include disclaimers: "For entertainment purposes only"
- Kelly Criterion outputs should show **quarter-Kelly** as default (full Kelly is too aggressive)
- Consider adding a maximum bet size cap regardless of Kelly output
- Never guarantee profits — emphasize that +EV betting requires large sample sizes

---

## 6. Performance & Scalability

### Current Scale
- ~1,200 games per season × 5 seasons = ~6,000 training rows
- 31-42 features
- 5 models (each trains in seconds)

### With Proposed Changes
- ~1,200 games × 8 seasons = ~9,600 training rows
- 45-55 features (with odds-derived features)
- 6-7 models (add O/U model)
- Daily odds snapshots: ~15 games/day × ~180 days = ~2,700 rows/season

### Scalability Concerns
- **None for model training** — even 20K rows with 55 features is tiny for modern hardware
- **Streamlit caching** — ensure `@st.cache_data(ttl=3600)` is used for all data fetches
- **sbrscrape rate limiting** — add `time.sleep(1)` between requests in backfill scripts

---

## 7. Suggested Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. Add `expected_value()` and `kelly_criterion()` to `utils/prediction_engine.py`
2. Display EV on Game Predictions page for existing DraftKings odds
3. Add probability calibration to training pipeline
4. Add calibration curve to Model Performance page

### Phase 2: Multi-Book Odds (2-3 days)
5. Install `sbrscrape`, create `get_multi_book_odds()` in `data_fetcher.py`
6. Add odds comparison table to Game Predictions page
7. Highlight best available line per market
8. Start daily odds collection (going forward)

### Phase 3: Model Enhancements (3-5 days)
9. Add odds-derived features (implied prob, spread, total) to training data
10. Train Over/Under model
11. Display O/U predictions on Game Predictions page
12. Extend training history to 8 seasons
13. Implement walk-forward CV for hyperparameter search

### Phase 4: Advanced Features (5+ days)
14. Bankroll management dashboard
15. Automated daily data pipeline
16. Config-based season management
17. Line movement tracking (requires scheduled scraping)
18. ROI analysis charts (by confidence tier, over time)

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `sbrscrape` breaks (SBR changes HTML) | Medium | High | Fall back to The Odds API; pin sbrscrape version |
| Overfitting with odds features | Medium | Medium | Use walk-forward CV; check out-of-sample accuracy |
| `nba_api` rate limiting | Low | Medium | Already handled with caching and retries |
| Model accuracy regression after changes | Medium | Medium | A/B test: run old and new models in parallel for 2 weeks |
| Legal issues with odds scraping | Low | High | Use The Odds API as primary; sbrscrape as supplement |
| Streamlit performance degradation | Low | Low | Profile with `st.cache_data` TTLs; lazy-load heavy components |

---

## 9. What NOT to Adopt

Some aspects of kyleskom's repo are **not worth adopting**:

| Feature | Why Not |
|---------|---------|
| **Flask web app** | Streamlit is better for our use case (faster development, built-in widgets) |
| **SQLite storage** | Parquet is better for our read-heavy analytics workload |
| **Raw feature columns** | Our engineered features (differentials, ratios) are more informative |
| **Subprocess-based prediction** | Our direct function call architecture is cleaner |
| **Neural network model** | Adds TensorFlow dependency for marginal gain over XGBoost/LightGBM |
| **CLI interface** | We don't need CLI — the Streamlit app is the interface |

---

## Status / - [x] Completed in code

- [x] `sbrscrape` multi-book odds support in `utils/data_fetcher.py`
- [x] `get_multi_book_odds()` and `get_best_lines()` implemented
- [x] `expected_value()` and `kelly_criterion()` in `utils/prediction_engine.py`
- [x] O/U model and `TOTAL_PTS` pipeline in `utils/model_utils.py`/`utils/feature_engine.py`
- [x] `HISTORICAL_SEASONS` expanded to 2017-18..2025-26
- [x] Walk-forward CV + isotonic calibration in `scripts/train_models.py`
- [x] `pages/1_Game_Predictions.py` uses preloaded `team_feat_map` in cached mode
- [x] ROI and cumulative P&L charts added to `pages/5_Model_Performance.py`
- [x] `config/seasons.toml` added for season management
- [x] `scripts/daily_update.py` created for daily pipeline orchestration

## Summary of All Integration Docs

| Document | Focus | Key Additions |
|----------|-------|---------------|
| [integration-odds-multi-sportsbook.md](integration-odds-multi-sportsbook.md) | `sbrscrape` for 7-book odds | Multi-book odds display, historical odds DB, line shopping |
| [integration-model-improvements.md](integration-model-improvements.md) | Training & prediction enhancements | O/U model, calibration, walk-forward CV, EV, Kelly |
| [integration-new-features.md](integration-new-features.md) | UI features & visualizations | EV cards, O/U display, odds table, new charts, bankroll mgmt |
| [integration-data-sources.md](integration-data-sources.md) | Data pipeline & storage | Historical odds, extended seasons, daily pipeline, config mgmt |
| **This document** | Architecture & strategy | Roadmap, risks, legal, what not to adopt |
