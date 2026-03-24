# Feature Engineering

This document outlines all features to be engineered for NBA game outcome and Pick 6 prediction models.

---

## Team-Level Features

### Basic Performance Metrics
| Feature | Description | Window |
|---------|-------------|--------|
| `win_pct` | Win percentage | Season, Last 10, Last 5 |
| `home_win_pct` / `away_win_pct` | Win % by location | Season |
| `streak` | Current win/loss streak length | Current |
| `avg_pts_scored` | Average points scored | Season, Last 10 |
| `avg_pts_allowed` | Average points allowed | Season, Last 10 |
| `avg_margin` | Average point differential | Season, Last 10 |

### Pace & Tempo
| Feature | Description | Source |
|---------|-------------|--------|
| `pace` | Possessions per 48 minutes | nba_api (teamestimatedmetrics) |
| `pace_rank` | Pace rank in league | Derived |
| `pace_diff` | Pace difference vs opponent | Derived |
| `possessions_per_game` | Estimated possessions | Derived from pace |

### Offensive Features
| Feature | Description |
|---------|-------------|
| `off_rating` | Points scored per 100 possessions |
| `efg_pct` | Effective field goal percentage |
| `ts_pct` | True shooting percentage |
| `ast_pct` | Assist percentage |
| `tov_pct` | Turnover percentage |
| `ft_rate` | Free throw rate (FTA/FGA) |
| `three_pt_rate` | Three-point attempt rate |
| `three_pt_pct` | Three-point percentage |
| `paint_pts_pct` | Percentage of points in the paint |
| `fastbreak_pts` | Average fast break points |
| `second_chance_pts` | Average second chance points |

### Defensive Features
| Feature | Description |
|---------|-------------|
| `def_rating` | Points allowed per 100 possessions |
| `opp_efg_pct` | Opponent effective FG% |
| `opp_tov_pct` | Forced turnover rate |
| `opp_ft_rate` | Opponent free throw rate |
| `blk_pct` | Block percentage |
| `stl_pct` | Steal percentage |
| `dreb_pct` | Defensive rebound percentage |
| `opp_three_pt_pct` | Opponent 3-point percentage |
| `opp_paint_pts` | Opponent points in the paint |

### Situational Features
| Feature | Description |
|---------|-------------|
| `is_home` | Home (1) or Away (0) |
| `rest_days` | Days since last game |
| `is_back_to_back` | Playing on consecutive days (0/1) |
| `is_3_in_4` | Third game in 4 nights (0/1) |
| `is_4_in_5` | Fourth game in 5 nights (0/1) |
| `travel_distance` | Estimated travel distance from last game |
| `time_zone_change` | Time zone shifts since last game |
| `days_since_road_trip_start` | Fatigue from extended road trips |

### Strength of Schedule
| Feature | Description |
|---------|-------------|
| `sos` | Strength of schedule (opponent win %) |
| `sos_last_10` | Recent strength of schedule |
| `opp_win_pct` | Current opponent's win percentage |
| `opp_rank` | Opponent's conference rank |

---

## Player-Level Features

### Core Statistics (Per Game & Per 36)
| Feature | Description |
|---------|-------------|
| `pts` | Points per game |
| `reb` | Rebounds per game |
| `ast` | Assists per game |
| `stl` | Steals per game |
| `blk` | Blocks per game |
| `tov` | Turnovers per game |
| `min` | Minutes per game |
| `fgm` / `fga` / `fg_pct` | Field goals made/attempted/percentage |
| `three_pm` / `three_pa` / `three_pct` | Three-pointers made/attempted/percentage |
| `ftm` / `fta` / `ft_pct` | Free throws made/attempted/percentage |

### Advanced Metrics
| Feature | Description | Source |
|---------|-------------|--------|
| `per` | Player Efficiency Rating | nba_api / derived |
| `ts_pct` | True Shooting Percentage | Derived |
| `usg_pct` | Usage Rate | nba_api (playerestimatedmetrics) |
| `plus_minus` | Plus/Minus per game | nba_api |
| `net_rating` | Net rating (on-court) | nba_api |
| `off_rating` | Offensive rating (on-court) | nba_api |
| `def_rating` | Defensive rating (on-court) | nba_api |

### Trend & Consistency Features
| Feature | Description |
|---------|-------------|
| `pts_rolling_5` | Points rolling average (last 5 games) |
| `pts_rolling_10` | Points rolling average (last 10 games) |
| `pts_std_dev` | Points standard deviation (consistency) |
| `min_trend` | Minutes trend (increasing/decreasing) |
| `usage_trend` | Usage rate trend |
| `pts_vs_season_avg` | Recent performance vs season average |

### Availability & Health
| Feature | Description |
|---------|-------------|
| `injury_status` | Healthy / Questionable / Doubtful / Out |
| `games_missed_recent` | Games missed in last 2 weeks |
| `games_since_return` | Games played since returning from injury |
| `minutes_restriction` | Whether on a minutes limit |

---

## Matchup Features

### Head-to-Head
| Feature | Description |
|---------|-------------|
| `h2h_record_season` | Head-to-head record this season |
| `h2h_avg_margin` | Average margin in H2H games |
| `h2h_record_3yr` | Head-to-head record last 3 seasons |

### Pace & Style Matchup
| Feature | Description |
|---------|-------------|
| `pace_matchup` | Combined pace prediction for the game |
| `projected_possessions` | Estimated possessions in game |
| `style_clash` | Offensive style vs defensive style compatibility |
| `three_pt_rate_vs_opp_three_def` | Team 3PT rate vs opponent 3PT defense |

### Ratings Differentials
| Feature | Description |
|---------|-------------|
| `net_rating_diff` | Team net rating minus opponent net rating |
| `off_rating_diff` | Offensive rating differential |
| `def_rating_diff` | Defensive rating differential |
| `elo_diff` | Elo rating differential |

---

## Pick 6-Specific Features

These features are designed for predicting player prop outcomes (more/less) in DraftKings Pick 6.

### Player Prop Context
| Feature | Description |
|---------|-------------|
| `prop_line` | The DraftKings prop line for the stat |
| `season_avg_vs_line` | Season average minus prop line |
| `recent_avg_vs_line` | Last 5 game average minus prop line |
| `over_rate_season` | % of games player went over this line (season) |
| `over_rate_last_10` | % of games player went over this line (last 10) |

### Matchup-Specific Props
| Feature | Description |
|---------|-------------|
| `opp_pts_allowed_to_pos` | Opponent points allowed to player's position |
| `opp_reb_allowed_to_pos` | Opponent rebounds allowed to player's position |
| `opp_ast_allowed_to_pos` | Opponent assists allowed to player's position |
| `opp_three_allowed_to_pos` | Opponent 3PM allowed to player's position |
| `opp_def_rating_vs_pos` | Opponent defensive rating vs player's position |
| `opp_pace_factor` | Pace impact on counting stats |

### Game Context Props
| Feature | Description |
|---------|-------------|
| `projected_game_total` | Vegas over/under for the game |
| `projected_spread` | Point spread (playing from behind = more stats) |
| `blowout_risk` | Probability of a large margin game (reduces minutes) |
| `teammate_availability` | Key teammate in/out (affects usage) |

### Correlation Features (for multi-pick strategy)
| Feature | Description |
|---------|-------------|
| `same_game_correlation` | Correlation between picks in same game |
| `stat_correlation` | Correlation between different stat types |
| `game_total_sensitivity` | How sensitive the prop is to game total |

---

## Feature Engineering Pipeline

```
Raw Game Data ──► Per-Game Stats ──► Rolling Averages ──► Feature Vectors
                                          │
Player Data ────► Per-Game Stats ──► Rolling Averages ──┤
                                          │              ├──► Model Input
Matchup Data ──► H2H Stats ──────► Differentials ──────┤
                                          │
Situational ───► Rest/Travel ─────► Flags ─────────────┘
```

### Rolling Window Strategy
- **Short-term**: Last 5 games (recent form)
- **Medium-term**: Last 10 games (trend detection)
- **Long-term**: Full season (baseline)
- **Weighted**: Exponentially weighted moving average (more weight on recent games)

### Feature Normalization
- Standardize features to z-scores (mean=0, std=1) for model compatibility
- Min-max scaling for neural network inputs
- Keep raw values for tree-based models (XGBoost, LightGBM)

### Missing Data Strategy
- Players with < 5 games: Use league average for position as fallback
- Injured/out players: Remove from player-level predictions, adjust team features
- New teams/trades: Shorter rolling windows, higher uncertainty
