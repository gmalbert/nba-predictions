# Data Sources & Pipeline Integration

## Source
[kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) — SQLite pipeline, historical data, config management

---

## Current Data Sources (Our Site)

| Source | Data | Storage | Seasons |
|--------|------|---------|---------|
| `nba_api` | Team stats, player stats, game logs, standings | Parquet | 5 (2021-25) |
| ESPN | Injury reports | API (live) | Current |
| The Odds API | DraftKings moneyline, spread, totals | API (live) | Current |
| nbastuffer | Advanced player/team/ref stats, rest days | Web scrape | Current |
| databallr | Advanced team metrics | Web scrape | Current |

## What kyleskom Uses

| Source | Data | Storage | Seasons |
|--------|------|---------|---------|
| `nba_api` (`LeagueDashTeamStats`) | Daily team stat snapshots | SQLite | 18 (2007-25) |
| `sbrscrape` | Live odds from 7 sportsbooks | SQLite | Current + backfill |
| Schedule CSVs | Game dates (for rest days calc) | CSV | All |
| RapidAPI | Player headshots, roster data | API (live) | Current |

---

## Integration Recommendations

### 1. Historical Odds Database

**Their approach:** `Get_Odds_Data.py` iterates day-by-day through each season, calling `sbrscrape.Scoreboard(date=date_pointer)` and storing odds into SQLite:

```
Schema: Date, Home, Away, OU, Spread, ML_Home, ML_Away, Points, Win_Margin, Days_Rest_Home, Days_Rest_Away
```

**Why we need this:** To train models with odds features (spread, total, implied prob), we need historical odds paired with historical game outcomes. Without this, we can only use odds features for prediction but not for training.

**Implementation:**

Create `scripts/fetch_historical_odds.py`:
```python
from sbrscrape import Scoreboard
import pandas as pd
from datetime import datetime, timedelta

def fetch_odds_for_date(date: datetime, sportsbook: str = "fanduel"):
    sb = Scoreboard(sport="NBA", date=date)
    rows = []
    for game in sb.games if sb.games else []:
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_ml": game.get("home_ml", {}).get(sportsbook),
            "away_ml": game.get("away_ml", {}).get(sportsbook),
            "spread": game.get("home_spread", {}).get(sportsbook),
            "total": game.get("total", {}).get(sportsbook),
        })
    return rows

def backfill_season(start_date, end_date, sportsbook="fanduel"):
    all_rows = []
    date = start_date
    while date <= end_date:
        all_rows.extend(fetch_odds_for_date(date, sportsbook))
        date += timedelta(days=1)
        time.sleep(1)  # Rate limit
    return pd.DataFrame(all_rows)
```

Store as `data_files/historical_odds.parquet` (consistent with our existing storage).

**Caveat:** `sbrscrape` may not return historical odds — it scrapes live SBR pages. If historical scraping fails, consider:
- Start collecting going forward (daily cron/scheduled task)
- Use `The Odds API` historical endpoint (paid, 500 requests/month on free tier)
- Source archived odds from `covers.com` or `oddsportal.com` (manual or scraping)

---

### 2. Extended Season History (2007–Present)

**Their approach:** `config.toml` defines season date ranges from 2007-08 through 2025-26, and `Get_Data.py` pulls `LeagueDashTeamStats` snapshots for each day of each season.

**What we have:** 5 seasons (2021-25) of team stats.

**Should we extend?**

| Consideration | For | Against |
|--------------|-----|---------|
| More training data | ✅ More samples, better generalization | |
| Rule changes | | ❌ Pre-2019 3PT era is different basketball |
| COVID season | | ❌ 2020 bubble was anomalous |
| Feature availability | | ❌ Some advanced stats not available pre-2015 |
| Storage | | ❌ 18 seasons × ~1200 games = ~21,600 rows |

**Recommendation:** Extend to **8 seasons** (2018-25). This captures the modern 3-point era, avoids ancient data that doesn't reflect current play, and roughly triples our training data.

Update `scripts/fetch_historical.py`:
```python
SEASONS_TO_FETCH = [
    "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25"
]
```

---

### 3. Config-Based Season Management

**Their approach:** `config.toml` stores season date ranges:

```toml
[data.2024-25]
start = "10/22/2024"
end = "04/13/2025"

[odds.2024-25]
start = "10/22/2024"
end = "04/13/2025"
```

**What we have:** Hardcoded `CURRENT_SEASON = "2024-25"` and date ranges scattered across files.

**Recommendation:** Create `config/seasons.toml` (or add to existing config):

```toml
[seasons]
current = "2024-25"

[seasons.2024-25]
start = "2024-10-22"
end = "2025-04-13"

[seasons.2023-24]
start = "2023-10-24"
end = "2024-04-14"
```

Load with `tomllib` (Python 3.11+) or `tomli`:

```python
import tomllib
with open("config/seasons.toml", "rb") as f:
    config = tomllib.load(f)
```

This makes adding new seasons a config change, not a code change.

---

### 4. Daily Data Collection Pipeline

**Their approach:** `main.py` with `--data` flag triggers daily data + odds collection before running predictions.

**What we have:** Data fetching is built into page load via `@st.cache_data`.

**Recommendation:** Create an optional offline data pipeline for production use:

Create `scripts/daily_update.py`:
```python
"""Run daily to refresh data before app start."""
import subprocess, sys

steps = [
    "fetch_historical.py --current-only",  # Today's team stats
    "fetch_historical_odds.py --today",     # Today's odds snapshot
    "scrape_external.py",                   # nbastuffer + databallr
]

for script in steps:
    subprocess.run([sys.executable, f"scripts/{script}"], check=True)

print("Daily data update complete.")
```

Benefits:
- App startup is faster (data pre-cached)
- Can run on schedule (Task Scheduler on Windows, cron on Linux)
- Decouples data collection from user-facing app

---

### 5. SQLite vs Parquet Decision

**Their approach:** SQLite databases (`TeamData.sqlite`, `OddsData.sqlite`, `dataset.sqlite`).

**What we use:** Parquet files in `data_files/`.

**Comparison:**

| Factor | Parquet | SQLite |
|--------|---------|--------|
| Read speed (analytics) | ✅ Faster for full-table scans | Slower |
| Write speed (append) | Slower (rewrite file) | ✅ Fast INSERT |
| Query flexibility | Need pandas | ✅ SQL queries |
| File size | ✅ Smaller (columnar compression) | Larger |
| Concurrent access | ❌ Single reader/writer | ✅ Multi-reader |
| Streamlit compatibility | ✅ Direct to DataFrame | Need cursor → DataFrame |

**Recommendation:** Stay with Parquet. Our use case is batch-read analytics (load all → filter in pandas), not transactional writes. Parquet's columnar compression and fast reads are ideal. Only consider SQLite if we add a daily append pipeline (odds snapshots) where atomic writes matter.

---

### 6. Potential New Data Sources

Beyond what kyleskom uses, consider adding:

| Source | Data | Free? | Notes |
|--------|------|-------|-------|
| `sbrscrape` | Live multi-book odds | ✅ Yes | See `integration-odds-multi-sportsbook.md` |
| Basketball Reference | Historical box scores, advanced stats | ✅ Yes | Rate-limited, use `basketball_reference_web_scraper` |
| NBA Schedule API | Full season schedule | ✅ Yes | For days-rest calculation |
| Covers.com | Historical odds/lines | ⚠️ Scraping | Terms of service concerns |
| Prop odds APIs | Player prop lines | 💰 Paid | For Pick 6 page accuracy |
| Weather/altitude | Denver altitude factor | ✅ Yes | Minor edge for Denver games |

---

## Priority Ranking

| Priority | Data Source | Effort | Impact |
|----------|-----------|--------|--------|
| 🔴 **P0** | `sbrscrape` live odds | Low | High — multi-book odds, line shopping |
| 🔴 **P0** | Historical odds collection (going forward) | Low | High — enables odds-as-features |
| 🟡 **P1** | Extended season history (8 seasons) | Medium | Medium — more training data |
| 🟡 **P1** | Config-based season management | Low | Medium — maintainability |
| 🟢 **P2** | Daily data pipeline script | Medium | Medium — production-readiness |
| ⚪ **P3** | Historical odds backfill | High | Medium — depends on source availability |
| ⚪ **P3** | Player prop odds APIs | High | Medium — Pick 6 enhancement |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/fetch_historical_odds.py` | **New** — daily odds collection from sbrscrape |
| `scripts/daily_update.py` | **New** — orchestrate all data updates |
| `config/seasons.toml` | **New** — centralized season date config |
| `scripts/fetch_historical.py` | Extend to support 8 seasons |
| `utils/data_fetcher.py` | Add `get_multi_book_odds()` (see odds doc) |
| `data_files/historical_odds.parquet` | **New** — historical odds storage |
