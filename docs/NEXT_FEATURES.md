# NBA Predictions — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Referee Foul Rate Feature

**Why:** Similar to the EPL referee impact feature, NBA referees have measurably different foul-call tendencies. High-foul referees inflate pace and push games over the total; low-foul referees favor under. Referee assignment is announced 2–3 hours before tip-off.

**How:**
1. Add `utils/nba_refs.py` that fetches referee assignments from the NBA Stats API (`https://stats.nba.com/stats/scoreboardv2`)
2. Compute per-referee historical stats from box score data: fouls/game, home team win%, total points differential
3. Add `ref_fouls_per_game` and `ref_home_win_pct` as features in `utils/feature_engine.py`
4. Display assigned referee + historical stats on `pages/1_Game_Predictions.py` game cards

**Complexity:** Medium

---

## Feature 2: Back-to-Back Schedule Density Feature

**Why:** NBA teams on the second night of a back-to-back have measurably lower win rates, especially on the road. The schedule data is available via `nba_api.live.nba.endpoints` and this is one of the most reliable and underused betting signals.

**How:**
1. In `utils/data_fetcher.py`, for each upcoming game, fetch the team's last game date
2. Compute `home_b2b` and `away_b2b` binary flags (1 if played last night)
3. Add to `utils/feature_engine.py` feature matrix
4. Display "B2B" warning badge on each game card in the UI
5. Historical analysis: model accuracy on B2B games vs well-rested games

**Complexity:** Low

---

## Feature 3: Injury Report Integration

**Why:** A missing starter (especially a star player with high usage rate) can shift the line by 3–5 points. The NBA publishes official injury reports 2 hours before tip-off. This is the single most actionable contextual signal before game time.

**How:**
1. Add `utils/injury_report.py` that fetches the official NBA injury report from `data.nba.net/prod/v1/{date}/players.json`
2. Cross-reference injured players with team starter lineup and compute an `impact_score` based on player usage rate (USG%)
3. Add `home_injury_impact` and `away_injury_impact` as numeric features (0 = full strength, 10 = star player out)
4. Display injury flags prominently on the Today page

**Complexity:** Medium

---

## Feature 4: Best Bets JSON Export for Sports-Picks-Grid

**Why:** `nba-predictions` is listed in the sports-picks-grid REPOS mapping. A consistent nightly export of `data_files/best_bets_today.json` using the unified schema would integrate NBA picks into the aggregator dashboard.

**How:**
1. Add `scripts/export_best_bets.py` that reads today's predictions from the Parquet cache
2. Filter to bets with confidence ≥ 58% and edge ≥ 3%
3. Write `data_files/best_bets_today.json` per the unified schema (`meta` + `bets` array)
4. Add to GitHub Actions nightly pipeline as the last step after predictions generate
5. Validate schema against sports-picks-grid `docs/02-unified-schema.md`

**Complexity:** Low

---

## Feature 5: Pace-Adjusted Totals Model

**Why:** The current totals model likely uses raw points averages. Adjusting for team pace (possessions per 48 minutes) and opponent pace gives a more stable estimate of expected game total. This is especially important when a fast-paced team plays a slow-paced team.

**How:**
1. Add `PACE` stat fetch via `nba_api.statistics.teams.TeamStats` or hoopR
2. Compute `expected_possessions = average(home_pace, away_pace)` per matchup
3. Derive `pace_adjusted_total = expected_possessions × (home_off_rtg + away_off_rtg) / 100`
4. Use `pace_adjusted_total` as a feature in the LightGBM totals model (alongside raw stats)
5. Compare edge on totals bets using pace-adjusted vs raw model across backtesting hold-out

**Complexity:** Medium
