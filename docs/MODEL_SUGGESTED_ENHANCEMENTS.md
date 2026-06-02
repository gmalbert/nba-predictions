# Betting Baseline (NBA) — Model Suggested Enhancements

## Priority 1: Totals Model

### Pace-Adjusted Features
- Current totals model uses raw scoring averages. Add pace-adjusted features: `pace_adj_ortg` and `pace_adj_drtg` from hoopr stats.
- Pace adjustment removes the bias caused by fast/slow teams inflating/deflating totals.

### Opponent Pace Interaction
- `|home_pace - away_pace|` as a variance feature. High pace differential → less predictable totals.

### Referee Foul Tendency
- NBA referee assignment data from `official_nba_com` or third-party sources. High-foul referees boost totals by 3–5 points historically.

## Priority 2: Moneyline Model

### Rest Differential
- `home_rest_days - away_rest_days`. Teams on rest advantage win at a measurably higher rate.

### Back-to-Back Flag
- `away_is_b2b`: Away team playing second night of B2B is a strong short-side signal (roughly −4% win rate).

### Travel Distance
- Cross-country travel (e.g., LAL flying to MIA) on short rest has a measurable negative effect on away teams.

## Priority 3: Player Prop Features

### Recent Hot/Cold Streaks
- Beyond rolling averages, add `pts_above_season_avg_l5` to capture hot streaks.

### Matchup-Based Prop Adjustment
- Adjust player prop predictions based on the opposing team's defensive rank vs. each position.

### Injury Impact Model
- When a key player is ruled out, redistribute their projected stats to teammates using `hoopr` lineup data.

## Priority 4: Calibration

- Apply isotonic regression to totals probability outputs.
- Track Brier score and AUC weekly through the season.
- Add model accuracy tracker tab to the dashboard.
