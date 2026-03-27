# New Features, UI Enhancements & Visualizations

## Source
[kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) — feature engineering, Flask UI, and analysis patterns

---

## Status / - [x] Completed in code

- [x] Odds-derived feature set added (`IMPLIED_PROB_HOME`, `IMPLIED_PROB_AWAY`, `spread_consensus`, `total_consensus`, `odds_disagreement_ml`, `odds_disagreement_total`)
- [x] Day-rest and team trends already included via nbastuffer features
- [x] EV display added to `pages/1_Game_Predictions.py`
- [x] O/U display added to `pages/1_Game_Predictions.py` via `predict_total_points()`
- [x] Multi-book odds table added `pages/1_Game_Predictions.py`, using `get_multi_book_odds()` + `get_best_lines()`
- [x] Calibration/ROI/P&L charts added to `pages/5_Model_Performance.py`

---

## 1. Odds-Derived Features for Models

**Their approach:** The training dataset (`Create_Games.py`) includes the sportsbook Over/Under line as a direct feature. This gives the model "market wisdom" — the OU line itself encodes information the books have about each matchup.

**What we can add:**

| Feature | Description | Source |
|---------|-------------|--------|
| `OU_LINE` | Sportsbook total (e.g., 214.5) | The Odds API or sbrscrape |
| `SPREAD` | Sportsbook spread (e.g., -7.5) | sbrscrape |
| `IMPLIED_PROB_HOME` | Home ML odds → implied probability | American odds conversion |
| `IMPLIED_PROB_AWAY` | Away ML odds → implied probability | American odds conversion |
| `CONSENSUS_TOTAL` | Average total across available books | sbrscrape multi-book |
| `CONSENSUS_SPREAD` | Average spread across available books | sbrscrape multi-book |
| `LINE_DISAGREEMENT` | Std deviation of total across books | sbrscrape multi-book |
| `SPREAD_DISAGREEMENT` | Std deviation of spread across books | sbrscrape multi-book |

**Why this matters:** Odds lines are the single most predictive feature available — they encode injury news, home court, rest, matchup strength, and public/sharp money. Adding them as features could meaningfully boost accuracy.

**Implementation:**

```python
def implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    return 100 / (american_odds + 100)

# In feature engineering
features["IMPLIED_PROB_HOME"] = implied_probability(home_ml)
features["IMPLIED_PROB_AWAY"] = implied_probability(away_ml)
features["SPREAD"] = home_spread
features["OU_LINE"] = total_line
```

---

## 2. Days-Rest Feature

**Their approach:** `Create_Games.py` computes `Days-Rest-Home` and `Days-Rest-Away` from the schedule — number of days since each team's last game.

**What we have:** We already scrape rest days from nbastuffer, but verify it's included in the model feature set.

**Validate:** Check `FEATURE_COLS_GAME` in `utils/model_utils.py` for rest-day features. If not present, add `DAYS_REST_HOME` and `DAYS_REST_AWAY`.

---

## 3. Expected Value Display (Game Predictions Page)

**New UI element — EV Card per game:**

```
┌──────────────────────────────────────────┐
│  🏀 BOS Celtics vs MIA Heat              │
│                                          │
│  Model: BOS 72.3%  |  Market: BOS 68.1%  │
│  Edge: +4.2%       |  EV: +$8.50/100     │
│                                          │
│  🟢 POSITIVE EV — BOS Moneyline (-210)   │
│  Kelly: Bet 3.1% of bankroll             │
└──────────────────────────────────────────┘
```

- Show for both home and away sides
- Highlight +EV bets in green
- Include EV magnitude ($X per $100 wagered)
- Optional: Kelly Criterion recommendation (behind toggle)

---

## 4. Over/Under Predictions Display

**New section on Game Predictions page:**

```
┌──────────────────────────────────────────┐
│  Total: 214.5                            │
│  Model: OVER (58.2%)                     │
│  EV: +$3.20/100    Kelly: 1.8%           │
│                                          │
│  ▓▓▓▓▓▓▓▓▓▓▓░░░░░  58.2% OVER          │
│  ░░░░░░▓▓▓▓▓▓▓▓▓▓▓  39.1% UNDER        │
│  ░▓░░░░░░░░░░░░░░░░  2.7% PUSH          │
└──────────────────────────────────────────┘
```

---

## 5. Multi-Sportsbook Odds Comparison Table

**New tab or section on Game Predictions page:**

| Sportsbook | Home ML | Away ML | Spread | Total | Best? |
|-----------|---------|---------|--------|-------|-------|
| DraftKings | -210 | +175 | -5.5 | 214.5 | |
| FanDuel | -215 | +180 | -5.5 | 215.0 | ✅ Total |
| BetMGM | -200 | +170 | -5.0 | 214.5 | ✅ Home ML |
| Caesars | -210 | +178 | -5.5 | 214.5 | |

- Highlight best line per market
- Show "line shopping savings" (EV gain vs worst line)
- Auto-update when odds change (with `sbrscrape`)

---

## 6. New Graphs for Model Performance Page

### A. Calibration Curve
**What:** Plot model predicted probability (x-axis) vs actual win rate (y-axis). Perfect calibration = diagonal line.

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
fig = go.Figure()
fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, name="Model"))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Perfect", line=dict(dash="dash")))
```

### B. Expected Value Distribution
**What:** Histogram of EV across all bets — shows how many bets were +EV vs −EV over the season.

### C. ROI by Confidence Tier
**What:** Bar chart showing ROI when betting only on games where model confidence ≥ 55%, 60%, 65%, 70%, 75%.

### D. Profit/Loss Over Time
**What:** Cumulative P&L chart showing how a $1000 bankroll evolves over the season following the model's +EV bets.

### E. Accuracy by Spread Range
**What:** How does accuracy change with game closeness?
- Expected blowouts (|spread| > 8): accuracy %
- Moderate favorites (|spread| 3-8): accuracy %
- Toss-ups (|spread| < 3): accuracy %

---

## 7. Line Movement Tracking (Future)

If we store odds snapshots over time (via `sbrscrape` on a schedule):
- **Opening vs Current line** — shows market movement
- **Sharp vs Public** — big line moves on low public % = sharp action
- **Reverse line movement** — line moves opposite to public betting %

Display as:
```
BOS -5.5 → BOS -6.5 (moved 1 point toward BOS)
OU 214.5 → OU 212.0 (moved 2.5 points under)
```

---

## 8. Bankroll Management Dashboard (New Page/Section)

Inspired by their Kelly Criterion integration:

```
┌──────────────────────────────────────────┐
│  💰 Bankroll Manager                     │
│                                          │
│  Starting Bankroll: $1,000               │
│  Today's +EV Bets: 4                     │
│                                          │
│  BOS ML -210  →  Bet $31 (3.1%)         │
│  LAL ML +140  →  Bet $22 (2.2%)         │
│  GSW/PHX O 226 → Bet $18 (1.8%)        │
│  DAL ML -150  →  Bet $15 (1.5%)         │
│                                          │
│  Total Risk: $86 (8.6% of bankroll)      │
│  Expected Return: +$14.20                │
└──────────────────────────────────────────┘
```

- Input bankroll amount
- Auto-calculate bet sizes using quarter-Kelly
- Track cumulative results over time

---

## Priority Ranking

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 🔴 **P0** | EV display on predictions | Low | High — core betting value |
| 🔴 **P0** | Over/Under predictions section | Medium | High — new market |
| 🟡 **P1** | Odds-derived model features | Medium | High — accuracy boost |
| 🟡 **P1** | Multi-book comparison table | Low | Medium — line shopping |
| 🟡 **P1** | Calibration curve on perf page | Low | Medium — model trust |
| 🟢 **P2** | ROI by confidence tier chart | Low | Medium |
| 🟢 **P2** | Cumulative P&L chart | Low | Medium |
| 🟢 **P2** | Bankroll management dashboard | Medium | Medium |
| ⚪ **P3** | Line movement tracking | High | Medium — needs scheduled scraping |
| ⚪ **P3** | Accuracy by spread range | Low | Low |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `pages/1_Game_Predictions.py` | Add EV cards, O/U section, odds comparison table, Kelly sizing |
| `pages/5_Model_Performance.py` | Add calibration curve, ROI by tier, P&L chart, accuracy by spread |
| `utils/prediction_engine.py` | Add `implied_probability()`, `expected_value()`, `kelly_criterion()` |
| `utils/feature_engineering.py` | Add odds-derived features (`IMPLIED_PROB_*`, `SPREAD`, `OU_LINE`, disagreement metrics) |
| `utils/model_utils.py` | Add `FEATURE_COLS_TOTALS`, update feature sets with odds columns |
