# Model Improvements — Lessons from kyleskom/NBA-ML-Sports-Betting

## Source
[kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) — XGBoost training, probability calibration, Over/Under model, Expected Value, Kelly Criterion

---

## What They Do That We Don't

### 1. Over/Under (Totals) Model

**Their approach:** A separate XGBoost model predicts whether the game total goes OVER or UNDER the sportsbook line. This is a 3-class classification problem (Under=0, Over=1, Push=2).

**Why this matters:** Our site currently shows totals lines from DraftKings but **does not predict** whether the game goes over/under. This is a massive gap — totals betting is one of the most popular NBA markets.

**Integration plan:**

```
Target: OU-Cover (0=Under, 1=Over, 2=Push)
Features: All current game features + the sportsbook OU line itself
Model: XGBoost (3-class softprob) — same architecture as our game winner model
```

Add to `utils/model_utils.py`:
- `FEATURE_COLS_TOTALS = FEATURE_COLS_GAME + ["OU_LINE"]`
- `train_totals_model()` — XGBoost multi:softprob with 3 classes
- `predict_totals()` — returns P(Under), P(Over), P(Push)

Display on Game Predictions page:
- **Over/Under card** next to win probability card
- Total line, model prediction (OVER/UNDER), confidence %
- Expected value for each side

---

### 2. Probability Calibration (Sigmoid / Isotonic)

**Their approach:** After training, they wrap the XGBoost booster in `sklearn.calibration.CalibratedClassifierCV` with sigmoid (Platt scaling) or isotonic regression calibration, using a held-out calibration set.

**Why this matters:** Raw XGBoost probabilities are often overconfident. Calibrated probabilities are critical for:
- Accurate expected value calculations
- Kelly Criterion bet sizing
- Trustworthy confidence tiers

**Integration plan:**

```python
from sklearn.calibration import CalibratedClassifierCV

# After training XGBoost
calibrator = CalibratedClassifierCV(
    BoosterWrapper(best_model, NUM_CLASSES),
    method="sigmoid",  # or "isotonic"
    cv="prefit",
)
calibrator.fit(X_calib, y_calib)
```

Modify `scripts/train_models.py`:
- Split training data: 80% train, 10% calibration, 10% test
- Train model on train set, calibrate on calibration set, evaluate on test set
- Save calibrator alongside model: `xgboost_game_calibrated.pkl`

Add calibration reliability plot to Model Performance page:
- Show raw vs calibrated probabilities
- Plot calibration curve (predicted prob vs actual win rate)

---

### 3. Walk-Forward Cross-Validation (TimeSeriesSplit)

**Their approach:** Uses `sklearn.model_selection.TimeSeriesSplit(n_splits=5)` for hyperparameter search — ensures temporal ordering is preserved.

**What we do:** We already use time-based train/test split (train 2021-24, test 2024-25), but our hyperparameter optimization within that window may not be strictly temporal.

**Integration plan:**

```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_cv_loss(X, y, params, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    for train_idx, val_idx in tscv.split(X):
        model = train_xgb(X[train_idx], y[train_idx], X[val_idx], y[val_idx], params)
        probs = model.predict(X[val_idx])
        losses.append(log_loss(y[val_idx], probs))
    return np.mean(losses)
```

Replace Optuna's default CV in `train_models.py` with walk-forward splits.

---

### 4. Expected Value Calculator

**Their formula:**

```python
def expected_value(Pwin, odds):
    """EV per $100 bet"""
    Ploss = 1 - Pwin
    Mwin = payout(odds)   # American odds → payout amount
    return round((Pwin * Mwin) - (Ploss * 100), 2)

def payout(odds):
    if odds > 0:
        return odds          # +150 → $150 profit on $100
    else:
        return (100 / (-1 * odds)) * 100  # -200 → $50 profit on $100
```

**What we have:** We show edge (model prob − market prob) but don't compute actual dollar EV.

**Integration plan:**

Add to `utils/prediction_engine.py`:

```python
def american_to_decimal(american_odds: int) -> float:
    if american_odds >= 100:
        return american_odds / 100
    return 100 / abs(american_odds)

def expected_value(model_prob: float, american_odds: int) -> float:
    """EV per $100 wagered. Positive = profitable bet."""
    decimal = american_to_decimal(american_odds)
    return round(model_prob * decimal * 100 - (1 - model_prob) * 100, 2)
```

Display on Game Predictions page:
- **EV per $100** for home and away moneyline bets
- Color-code: green for +EV, red for −EV
- Only show bets where EV > $0

---

### 5. Kelly Criterion Bet Sizing

**Their formula:**

```python
def calculate_kelly_criterion(american_odds, model_prob):
    decimal_odds = american_to_decimal(american_odds)
    fraction = (decimal_odds * model_prob - (1 - model_prob)) / decimal_odds
    return max(fraction, 0)  # never bet negative
```

**What it does:** Given model probability and sportsbook odds, it calculates the optimal fraction of bankroll to wager.

**Integration plan:**

Add to `utils/prediction_engine.py`:

```python
def kelly_criterion(model_prob: float, american_odds: int) -> float:
    """Fraction of bankroll (0-1) to wager. 0 = no bet."""
    decimal = american_to_decimal(american_odds)
    f = (decimal * model_prob - (1 - model_prob)) / decimal
    return round(max(f, 0), 4)
```

Display on Game Predictions page (behind a toggle):
- **Bankroll %** per bet (e.g., "Bet 3.2% of bankroll on BOS ML")
- Use **quarter Kelly** (÷4) as the conservative default — full Kelly is too aggressive

---

### 6. Random Search Hyperparameter Optimization (100 trials)

**Their approach:** Random search over 100 parameter combinations for XGBoost:
- `max_depth`: 2–12
- `eta` (learning rate): 0.003–0.3 (log-uniform)
- `subsample`: 0.5–1.0
- `colsample_bytree/bylevel/bynode`: 0.5–1.0
- `min_child_weight`: 1–20
- `gamma`: 0–10
- `max_delta_step`: 0–10
- `max_bin`: 128–1024
- `lambda` (L2 reg): 0.1–10 (log-uniform)
- `alpha` (L1 reg): 0.01–5 (log-uniform)
- `num_boost_round`: 300–2500

**What we do:** Optuna with 50 trials (already good).

**Consideration:** Their search space is wider and includes `colsample_bylevel`, `colsample_bynode`, `max_delta_step`, and `max_bin` — features we may not be tuning. Adding these to our Optuna space could improve results.

---

### 7. Logistic Regression Baseline

**Their approach:** In addition to XGBoost and NN, they train a calibrated `LogisticRegression` with `StandardScaler`, class weighting, and L1/L2 penalty search.

**What we have:** We already have a logistic regression model. Ensure it uses `StandardScaler` (important for LR) and `class_weight="balanced"`.

---

## Priority Ranking

| Priority | Improvement | Effort | Impact |
|----------|------------|--------|--------|
| 🔴 **P0** | Over/Under totals model | Medium | High — new market coverage |
| 🔴 **P0** | Expected Value calculator | Low | High — core betting metric |
| 🟡 **P1** | Probability calibration | Low | High — better probability estimates |
| 🟡 **P1** | Kelly Criterion sizing | Low | Medium — bankroll management |
| 🟢 **P2** | Walk-forward CV | Low | Medium — better hyperparameter selection |
| 🟢 **P2** | Wider hyperparameter search | Low | Low-Medium — diminishing returns |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `utils/prediction_engine.py` | Add `expected_value()`, `kelly_criterion()`, `american_to_decimal()`, `predict_totals()` |
| `utils/model_utils.py` | Add `FEATURE_COLS_TOTALS`, `train_totals_model()`, calibration wrapper |
| `scripts/train_models.py` | Add O/U model training, calibration step, walk-forward CV |
| `models/` | New: `xgboost_totals_latest.pkl`, `xgboost_game_calibrated.pkl` |
| `pages/1_Game_Predictions.py` | Add O/U prediction, EV display, Kelly sizing toggle |
| `pages/5_Model_Performance.py` | Add calibration reliability plot, O/U model metrics |
