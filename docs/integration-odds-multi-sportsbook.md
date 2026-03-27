# Multi-Sportsbook Odds Integration

## Source
[kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) — `SbrOddsProvider`, `Get_Odds_Data`, `sbrscrape` package

---

## Status / - [x] Completed in code

- [x] `sbrscrape` integration implemented via `utils/data_fetcher.py` (`get_multi_book_odds()` + `get_best_lines()`)
- [x] Historical odds backfill script implemented as `scripts/fetch_historical_odds.py`
- [x] Odds features added, and model training includes odds-based feature columns
- [x] Live multi-book odds UI in `pages/1_Game_Predictions.py`
- [x] Odds API still present as fallback

## What They Have

The repo uses **`sbrscrape`** (`pip install sbrscrape`) to pull live and historical odds from **ScoresAndOdds / SBR** for these sportsbooks:

| Book | Key |
|------|-----|
| FanDuel | `fanduel` |
| DraftKings | `draftkings` |
| BetMGM | `betmgm` |
| PointsBet | `pointsbet` |
| Caesars | `caesars` |
| Wynn | `wynn` |
| Bet Rivers NY | `bet_rivers_ny` |

For each game, `sbrscrape.Scoreboard(sport="NBA")` returns a dict per game with:

```python
{
    "home_team": "Boston Celtics",
    "away_team": "Miami Heat",
    "home_spread":    {"fanduel": -7.5, "draftkings": -7, ...},
    "away_spread":    {"fanduel": 7.5,  "draftkings": 7,  ...},
    "home_spread_odds": {"fanduel": -110, ...},
    "away_spread_odds": {"fanduel": -110, ...},
    "home_ml":        {"fanduel": -320, "draftkings": -310, ...},
    "away_ml":        {"fanduel": 260,  "draftkings": 255,  ...},
    "total":          {"fanduel": 214.5, "draftkings": 215, ...},
    "over_odds":      {"fanduel": -110, ...},
    "under_odds":     {"fanduel": -110, ...},
    "home_score": 0,        # live or final
    "away_score": 0,
}
```

### Historical Odds Pipeline
Their `Get_Odds_Data.py` iterates every date in a season, calls `Scoreboard(date=date_pointer)`, and stores rows into SQLite with columns:
- `Date`, `Home`, `Away`, `OU`, `Spread`, `ML_Home`, `ML_Away`, `Points`, `Win_Margin`, `Days_Rest_Home`, `Days_Rest_Away`

---

## What We Currently Have

- **The Odds API** (`get_nba_odds(api_key)`) — fetches DraftKings only, returns moneyline + spreads + totals
- Single sportsbook (DraftKings) shown on Game Predictions page
- No historical odds storage
- No spread comparison across books

---

## Integration Plan

### Phase 1: Add `sbrscrape` for Live Multi-Book Odds

**Install:** `pip install sbrscrape`

**New function in `utils/data_fetcher.py`:**

```python
from sbrscrape import Scoreboard

@st.cache_data(ttl=300)  # 5 min refresh
def get_multi_book_odds() -> list[dict]:
    """Fetch live NBA odds from all major sportsbooks via sbrscrape."""
    sb = Scoreboard(sport="NBA")
    if not hasattr(sb, "games") or not sb.games:
        return []
    return sb.games
```

**Display on Game Predictions page** — new "📊 Odds Comparison" expander per game showing:

| Book | Spread | ML Home | ML Away | Total |
|------|--------|---------|---------|-------|
| FanDuel   | -7.5 (-110) | -320 | +260 | 214.5 |
| DraftKings | -7 (-110)  | -310 | +255 | 215   |
| BetMGM    | -7 (-110)   | -300 | +250 | 214.5 |

Highlight the **best line** per market (greenest moneyline, tightest spread, best total).

### Phase 2: Historical Odds Database

**New script** `scripts/fetch_historical_odds.py`:

```python
# Iterate dates from config, store to data_files/historical/odds_{season}.parquet
# Columns: Date, Home, Away, OU_{book}, ML_Home_{book}, ML_Away_{book},
#           Spread_{book}, Points, Win_Margin, Days_Rest_Home, Days_Rest_Away
```

Store per-book columns (e.g., `ML_Home_fanduel`, `ML_Home_draftkings`) so we can:
1. Compare opening vs closing lines
2. Train models with odds-as-features
3. Backtest against specific books

### Phase 3: Odds-Aware Features for Models

Add to `FEATURE_COLS_GAME_EXTENDED`:
- `IMPLIED_PROB_HOME` — consensus implied probability (average across books)
- `SPREAD_CONSENSUS` — average spread across books
- `TOTAL_CONSENSUS` — average total across books
- `LINE_MOVEMENT` — spread change from open to close (requires storing timestamps)
- `ODDS_DISAGREEMENT` — std dev of moneylines across books (higher = more uncertain)

### Phase 4: Best-Line Shopping Display

New **"Line Shopping"** tab on Game Predictions page:
- For each game, show the best available line per market across all books
- Calculate **EV at each book** using our model's probability
- Recommend which sportsbook to bet at

---

## Advantages Over The Odds API

| | The Odds API (current) | sbrscrape (proposed) |
|---|---|---|
| **Cost** | Paid ($$$/mo after free tier) | Free |
| **Books** | 1 (DraftKings) | 7+ |
| **Data** | Moneyline, spread, total | Same + spread odds, o/u odds |
| **Historical** | No | Yes (backfill by date) |
| **Rate limits** | API quota | Polite scraping (1-3s delay) |
| **Maintenance** | API versioning | Scraping may break |

**Recommendation:** Use `sbrscrape` as the primary odds source. Keep The Odds API as a fallback for when SBR is unreachable.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `sbrscrape` breaks (scraping target changes) | Keep The Odds API fallback; pin sbrscrape version |
| Rate limiting from SBR | Random 1-3 second delay between requests (their pattern) |
| Missing books for some games | Graceful `None` handling per-book; show available books only |
| Historical backfill takes hours | Run incrementally; cache to parquet; skip existing dates |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `requirements.txt` | Add `sbrscrape>=0.0.12` |
| `utils/data_fetcher.py` | Add `get_multi_book_odds()` |
| `scripts/fetch_historical_odds.py` | New — backfill historical odds to parquet |
| `pages/1_Game_Predictions.py` | Add odds comparison table per game |
| `utils/feature_engine.py` | Add odds-derived features |
| `scripts/train_models.py` | Include odds features in training |
