# Prediction Methodology

## Status
- [x] Complete

This document details how game outcome predictions and DraftKings Pick 6 predictions are generated, including confidence scoring, real-time updates, and strategy recommendations.

---

## Game Outcome Predictions

### What We Predict

For each NBA game, the model outputs:

| Prediction | Description | Method |
|------------|-------------|--------|
| **Win Probability** | Probability each team wins (e.g., 62% / 38%) | Ensemble model output |
| **Predicted Spread** | Expected point differential (e.g., Team A -5.5) | Regression model or calibrated from win prob |
| **Predicted Total** | Expected combined score (e.g., 218.5) | Separate regression model |
| **Confidence Tier** | High / Medium / Low | Based on model agreement and edge size |

### Prediction Pipeline

```
1. Fetch today's schedule (nba_api: scoreboardv2)
         │
2. For each game, compute features for both teams
         │
3. Run features through ensemble model
         │
4. Output: win probability, spread, total
         │
5. Compare model output to market lines (if available)
         │
6. Assign confidence tier
         │
7. Display on Streamlit dashboard
```

### Confidence Tiers

Confidence is determined by **model agreement** and **edge vs market**.

| Tier | Criteria | Description |
|------|----------|-------------|
| **High** | All ensemble members agree + edge > 5% vs market | Strong conviction — models align and see value |
| **Medium** | Majority agree OR edge 2-5% vs market | Moderate conviction — some signals in our favor |
| **Low** | Models disagree OR edge < 2% | Low conviction — close to a coin flip or no edge |

**Edge calculation:**
$$\text{Edge} = P_{\text{model}} - P_{\text{implied by odds}}$$

Example: Model says 65% win probability, odds imply 58% → Edge = +7%

### Spread Prediction

Method: Convert win probability to predicted spread using historical calibration.

$$\text{Predicted Spread} \approx -13 \times \ln\left(\frac{1 - P_{\text{win}}}{P_{\text{win}}}\right)$$

This logistic approximation is calibrated against historical closing lines.

### Total Prediction

A separate regression model predicts game totals using:
- Combined team pace
- Combined offensive/defensive ratings
- Pace matchup factor
- Historical totals in similar matchups
- Altitude factor (Denver home games trend higher)

---

## DraftKings Pick 6 Predictions

### How Pick 6 Works

[DraftKings Pick 6](https://pick6.draftkings.com/?sport=NBA) is a daily fantasy contest where you:

1. **Select 2-6 player stat props** from the available board
2. **Pick MORE or LESS** for each prop (will the player go over or under the line?)
3. **Payouts increase** with more picks:
   - 2 picks correct: 3x
   - 3 picks correct: 5x
   - 4 picks correct: 10x
   - 5 picks correct: 20x
   - 6 picks correct: 40x

### Prop Categories

| Category | Example | Source for Modeling |
|----------|---------|-------------------|
| Points | LeBron James MORE 25.5 PTS | `playergamelog`, `boxscoretraditionalv2` |
| Rebounds | Anthony Davis LESS 11.5 REB | `playergamelog`, `boxscoretraditionalv2` |
| Assists | Luka Doncic MORE 8.5 AST | `playergamelog`, `boxscoretraditionalv2` |
| 3-Pointers Made | Steph Curry MORE 4.5 3PM | `playergamelog`, `boxscoretraditionalv2` |
| Pts + Reb + Ast | Jokic MORE 45.5 PRA | Derived (sum of individual stats) |
| Steals + Blocks | Player MORE 2.5 STL+BLK | Derived |

### Player Prop Prediction Pipeline

```
1. Get today's Pick 6 board (manual entry or scraping)
         │
2. For each prop, identify player and stat category
         │
3. Fetch player's recent game logs (nba_api)
         │
4. Compute player-level features (rolling averages, splits, matchup)
         │
5. Compute matchup features (opponent defense vs position)
         │
6. Run through prop-specific model (XGBoost/LightGBM)
         │
7. Output: predicted stat value + over/under probability
         │
8. Rank props by confidence
         │
9. Build recommended Pick 6 entries
```

### Prop Modeling Approach

**For each stat category**, train a separate model:

#### Points Model
- **Key features**: Minutes projection, usage rate, opponent defensive rating, recent scoring trend, pace matchup, home/away
- **Output**: Predicted points + P(over line)

#### Rebounds Model
- **Key features**: Position, opponent rebound rate, pace (more possessions = more boards), height matchup, minutes
- **Output**: Predicted rebounds + P(over line)

#### Assists Model
- **Key features**: Usage rate, team assist rate, pace, opponent turnover rate, point guard vs non-PG
- **Output**: Predicted assists + P(over line)

#### 3-Pointers Made Model
- **Key features**: 3PT attempt rate, 3PT%, opponent 3PT defense, pace matchup, recent shooting trend
- **Output**: Predicted 3PM + P(over line)
- **Note**: High variance stat — confidence intervals will be wider

### Over/Under Probability

$$P(\text{OVER}) = P(X > \text{line})$$

Where $X$ is the predicted stat distribution. We model this as:

1. **Point estimate**: Model predicts expected value $\mu$
2. **Uncertainty**: Historical standard deviation $\sigma$ for that player/stat
3. **Probability**: 

$$P(\text{OVER}) = 1 - \Phi\left(\frac{\text{line} - \mu}{\sigma}\right)$$

Where $\Phi$ is the standard normal CDF.

**Alternative**: Use the classifier output directly (XGBoost trained on binary over/under target).

---

## Pick Building Strategy

### Confidence Scoring

Each prop gets a confidence score:

$$\text{Confidence} = |P(\text{OVER}) - 0.5| \times 2$$

This ranges from 0 (complete toss-up) to 1 (extremely confident).

| Confidence Score | Label | Recommendation |
|-----------------|-------|----------------|
| > 0.40 | Strong | Include in picks |
| 0.25 - 0.40 | Moderate | Include with caveats |
| 0.10 - 0.25 | Weak | Avoid unless needed to fill |
| < 0.10 | Skip | Too close to call |

### Correlation Awareness

When building multi-pick entries, account for correlations:

**Positive correlations** (good to stack):
- Player points + game total (high-scoring game = more individual points)
- Teammates in a blowout favorite (both benefit from dominant performance)

**Negative correlations** (avoid combining):
- Opposing players in same game (if one team dominates, the other underperforms)
- Points MORE + Assists MORE for same player (usage trade-off in some cases)

**Same-game considerations:**
- Multiple picks from same game increases variance
- Diversify across games when possible for more independent outcomes

### Entry Construction Algorithm

```python
def build_pick6_entry(props, n_picks=6, risk='balanced'):
    # 1. Sort props by confidence
    ranked = sorted(props, key=lambda p: p['confidence'], reverse=True)
    
    # 2. Select top props, avoiding high-correlation pairs
    selected = []
    for prop in ranked:
        if len(selected) >= n_picks:
            break
        if not has_high_correlation(prop, selected):
            selected.append(prop)
    
    # 3. Calculate combined probability
    combined_prob = product(p['confidence'] for p in selected)
    
    # 4. Calculate expected value
    payout = get_payout_multiplier(n_picks)
    ev = combined_prob * payout - 1
    
    return selected, combined_prob, ev
```

### Risk Profiles

| Profile | Strategy | Pick Count |
|---------|----------|------------|
| **Conservative** | Top 2-3 highest confidence picks only | 2-3 |
| **Balanced** | Mix of high and moderate confidence | 4-5 |
| **Aggressive** | 6 picks targeting max payout (40x) | 6 |

### Expected Value Calculation

$$EV = P(\text{all correct}) \times \text{Payout} - \text{Entry Cost}$$

Example (5 picks, each 60% confidence):
$$EV = 0.60^5 \times 20 - 1 = 0.0778 \times 20 - 1 = 1.555 - 1 = +0.555$$

Positive EV means the entry is profitable in the long run.

---

## Real-Time Updates

### Pre-Game Updates
| Trigger | Action | Timing |
|---------|--------|--------|
| Injury report released | Recalculate affected game predictions | 1:30 PM ET |
| Starting lineups confirmed | Update player minutes projections | ~30 min pre-game |
| Line movement detected | Recalculate edge vs market | Continuous |

### In-Game (Future Enhancement)
- Live win probability updates using play-by-play data
- Halftime model adjustments for second-half predictions

---

## Displaying Predictions

### Game Prediction Card
```
┌─────────────────────────────────────┐
│ LAL Lakers  vs  BOS Celtics         │
│                                     │
│ ████████████░░░░░░░░  42% │ 58%     │
│                                     │
│ Predicted Spread: BOS -4.5          │
│ Predicted Total: 219.5              │
│ Confidence: HIGH ●●●               │
│                                     │
│ Key Factors:                        │
│ • BOS defensive rating #2 in NBA   │
│ • LAL on 2nd night of back-to-back │
│ • BOS 8-2 in last 10 home games    │
└─────────────────────────────────────┘
```

### Pick 6 Recommendation Card
```
┌─────────────────────────────────────┐
│ 🏀 Today's Pick 6 (5 picks)        │
│ Combined Confidence: 14.2%          │
│ Payout: 20x  │  EV: +$1.84         │
│                                     │
│ ▲ LeBron James    PTS MORE 25.5  72%│
│ ▼ Jayson Tatum    REB LESS 8.5   68%│
│ ▲ Nikola Jokic    AST MORE 9.5   65%│
│ ▲ Steph Curry     3PM MORE 4.5   61%│
│ ▼ Luka Doncic     PTS LESS 32.5  59%│
└─────────────────────────────────────┘
```
