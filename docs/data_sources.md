# Data Sources

## Primary: nba_api

The [`nba_api`](https://github.com/swar/nba_api) Python package is our primary data source, providing access to the full suite of NBA.com stats endpoints.

### Key Endpoints

#### Game & Schedule Data
| Endpoint | Description | Use Case |
|----------|-------------|----------|
| `scoreboardv2` | Live/daily scoreboard | Today's games, live scores |
| `leaguegamefinder` | Search games by team/player/season | Historical game logs, filtering |
| `leaguegamelog` | Game logs for entire league | Season-wide game data |
| `leagueschedule` | Full season schedule | Upcoming game schedules |

#### Team Statistics
| Endpoint | Description | Use Case |
|----------|-------------|----------|
| `teamgamelog` | Team game-by-game stats | Rolling averages, streaks |
| `teamdashboardbygeneralsplits` | Team splits (home/away, etc.) | Situational analysis |
| `teamestimatedmetrics` | Advanced estimated metrics | Offensive/defensive ratings |
| `leaguestandingsv3` | Current standings | Conference/division context |
| `teamvsplayer` | Team performance vs specific players | Matchup analysis |
| `teaminfocommon` | Team metadata | Team info lookups |

#### Player Statistics
| Endpoint | Description | Use Case |
|----------|-------------|----------|
| `playergamelog` | Player game-by-game stats | Individual performance trends |
| `playerdashboardbygeneralsplits` | Player splits | Situational player stats |
| `playerestimatedmetrics` | Advanced player metrics | PER, usage, efficiency |
| `commonplayerinfo` | Player metadata | Position, height, age, etc. |
| `playercareerstats` | Career statistics | Long-term baselines |
| `playerindex` | All active players | Player roster lookups |

#### Box Scores & Play-by-Play
| Endpoint | Description | Use Case |
|----------|-------------|----------|
| `boxscoretraditionalv2` | Traditional box scores | Points, rebounds, assists, etc. |
| `boxscoreadvancedv2` | Advanced box scores | ORtg, DRtg, pace, TS%, etc. |
| `boxscoreplayertrackv2` | Player tracking data | Speed, distance, touches |
| `boxscorescoringv2` | Scoring breakdown | Paint/midrange/3pt scoring |
| `playbyplayv2` | Play-by-play data | Clutch analysis, momentum |

#### League-Wide Data
| Endpoint | Description | Use Case |
|----------|-------------|----------|
| `leaguedashteamstats` | League-wide team stats | League averages, rankings |
| `leaguedashplayerstats` | League-wide player stats | Player rankings, comparisons |
| `leaguedashlineups` | Lineup statistics | Lineup impact analysis |

### nba_api Usage Notes
- **Rate Limiting**: NBA.com enforces rate limits. Use delays between requests (`time.sleep(0.6)` recommended).
- **Caching**: Cache responses locally to avoid redundant API calls. Use `st.cache_data` in Streamlit.
- **Seasons**: Use format `"2025-26"` for season parameters.
- **Historical Data**: Available back to 1996-97 season for most endpoints.

---

## Web Scraping Sources

### Basketball Reference (basketball-reference.com)
- **Four Factors**: Team-level eFG%, TOV%, ORB%, FT/FGA
- **Advanced Stats**: BPM, VORP, Win Shares not available via nba_api
- **Historical Odds**: Closing lines for backtesting
- **Scraping Method**: `requests` + `BeautifulSoup4` (HTML parsing)
- **Consideration**: Respect `robots.txt`, use delays between scrapes, cache aggressively

### ESPN (espn.com)
- **Injury Reports**: Real-time injury updates and status
- **BPI (Basketball Power Index)**: ESPN's proprietary team ratings
- **Scraping Method**: JSON API available at `site.api.espn.com`

### NBA Injury Reports (official.nba.com)
- **Official Injury Reports**: Mandatory pre-game injury reports
- **Format**: PDF reports, typically released 1:30 PM ET on game days
- **Use**: Critical for Pick 6 — player availability directly impacts prop predictions

### Rotowire (rotowire.com)
- **Lineup Confirmations**: Starting lineup announcements
- **Injury Analysis**: Expert injury analysis and return timelines
- **News Feed**: Player news that may affect performance

---

## Odds & Lines Data

### The Odds API (the-odds-api.com)
- **Coverage**: Moneyline, spread, totals from 20+ sportsbooks
- **Free Tier**: 500 requests/month (sufficient for daily use)
- **Endpoints**: `/sports/basketball_nba/odds`
- **Use Cases**:
  - Closing line comparison for model calibration
  - Identifying edges between model predictions and market odds
  - Historical line movement analysis
- **Setup**: Requires free API key (see [the-odds-api.com](https://the-odds-api.com))

### DraftKings (for Pick 6 context)
- **Pick 6 Board**: Daily player prop selections available at [pick6.draftkings.com](https://pick6.draftkings.com/?sport=NBA)
- **Prop Categories**: Points, rebounds, assists, 3-pointers made, steals, blocks, pts+reb+ast combos
- **Format**: Pick more/less on 2-6 player stat props
- **Note**: No official API — board data may need manual entry or scraping (check terms of service)

---

## Data Pipeline Architecture

```
nba_api ─────────────┐
                     │
Basketball Reference ─┤
                     ├──► Raw Data ──► Feature Engineering ──► Model Input
ESPN / Injury Data ──┤
                     │
The Odds API ────────┘
```

### Recommended Update Schedule
| Data Type | Frequency | Timing |
|-----------|-----------|--------|
| Game scores & box scores | After each game | Post-game (~midnight ET) |
| Team/player rolling stats | Daily | Morning refresh |
| Injury reports | Twice daily | 10 AM ET + 1:30 PM ET |
| Odds/lines | Pre-game | 2-3 hours before tip-off |
| Pick 6 board | Daily | When board is posted (~10 AM ET) |
| Season standings | Daily | Morning refresh |

### Data Storage Strategy
- **Development**: CSV/Parquet files in `data_files/` directory
- **Production**: SQLite database or Streamlit Cloud compatible storage
- **Caching**: `st.cache_data` with appropriate TTL values per data type
