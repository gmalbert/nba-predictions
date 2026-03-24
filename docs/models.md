# Models

This document outlines the machine learning models, training strategies, and evaluation methods for NBA game outcome and Pick 6 predictions.

---

## Model Overview

| Model | Use Case | Complexity | Interpretability |
|-------|----------|------------|-----------------|
| Elo Rating System | Game outcomes (baseline) | Low | High |
| Logistic Regression | Game outcomes (baseline) | Low | High |
| XGBoost | Game outcomes & props (primary) | Medium | Medium |
| LightGBM | Game outcomes & props (primary) | Medium | Medium |
| Random Forest | Game outcomes (ensemble member) | Medium | Medium |
| Neural Network | Player props (advanced) | High | Low |
| Ensemble | Final predictions | High | Medium |

---

## Baseline Models

### Elo Rating System

A custom Elo rating system tailored for NBA predictions.

**How it works:**
- Each team starts with a rating of 1500 at the beginning of each season
- Ratings update after each game based on outcome and margin of victory
- Home court advantage is built into the system as a fixed bonus

**Parameters to tune:**
- `K-factor`: How much ratings change per game (start with K=20)
- `Home advantage`: Elo points added for home team (start with +100)
- `Margin of victory multiplier`: Scale K by margin to reward/penalize blowouts
- `Season carryover`: Regress ratings toward mean between seasons (e.g., 75% carryover)

**Expected outcome prediction:**
$$P(\text{Team A wins}) = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

Where $R_A$ and $R_B$ are the Elo ratings of teams A and B (with home advantage applied).

**Advantages:**
- Simple, fast, no feature engineering needed
- Great baseline to beat with more complex models
- Naturally adapts to team strength changes over the season

---

### Logistic Regression

A simple logistic regression on the core feature set.

**Target**: Binary (1 = home team wins, 0 = away team wins)

**Input features** (start with):
- Net rating differential
- Home/away indicator
- Rest days differential
- Win percentage differential
- Pace differential

**Regularization**: L2 (Ridge) with cross-validated `C` parameter

**Purpose**: Interpretable baseline — coefficients directly show feature importance.

---

## Primary Models

### XGBoost (Gradient Boosted Trees)

The primary model for both game outcomes and player prop predictions.

**Game Outcome Model:**
- **Target**: Binary classification (home team win/loss)
- **Features**: Full team-level and matchup feature set (see [features.md](features.md))
- **Output**: Win probability for each team

**Player Prop Model (for Pick 6):**
- **Target**: Regression (predicted stat value) or Classification (over/under the line)
- **Features**: Player-level + matchup + game context features
- **Output**: Predicted stat value and over/under probability

**Hyperparameter Search Space:**
```python
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}
```

**Tuning strategy**: Bayesian optimization with `Optuna` or `TimeSeriesSplit` cross-validation.

---

### LightGBM

Faster training alternative to XGBoost with comparable performance.

**Advantages over XGBoost:**
- Faster training on large datasets
- Native categorical feature support
- Lower memory usage

**Hyperparameter Search Space:**
```python
param_grid = {
    'num_leaves': [15, 31, 63],
    'max_depth': [-1, 5, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 1, 5]
}
```

**Use case**: Primary model for player prop predictions (many individual models needed — speed matters).

---

### Random Forest

Ensemble of decision trees — used as a diversity member in the final ensemble.

**Configuration:**
- `n_estimators`: 500
- `max_depth`: 8-12
- `min_samples_leaf`: 10
- `max_features`: 'sqrt'

**Purpose**: Provides uncorrelated predictions to XGBoost/LightGBM for ensembling.

---

## Advanced Models

### Neural Network (Optional)

A feedforward neural network for player prop predictions where complex feature interactions may exist.

**Architecture:**
```
Input Layer (N features)
    → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    → Dense(64, ReLU) → BatchNorm → Dropout(0.2)
    → Dense(32, ReLU)
    → Output Layer (1, Sigmoid for classification / Linear for regression)
```

**Training:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary cross-entropy (classification) or MSE (regression)
- Early stopping with patience=10
- Learning rate scheduler (reduce on plateau)

**Note**: Only use if tree-based models plateau. Neural networks require more data and tuning to outperform gradient boosting on tabular data.

---

## Ensemble Strategy

### Stacking Ensemble

Combine predictions from multiple models for the final output.

**Level 1 Models** (base learners):
- Elo Rating System
- Logistic Regression
- XGBoost
- LightGBM
- Random Forest

**Level 2 Model** (meta-learner):
- Logistic Regression on Level 1 outputs
- Uses out-of-fold predictions to avoid data leakage

**Weighting** (alternative to stacking):
- Weighted average based on recent validation performance
- Dynamically adjust weights as model performance shifts

```python
final_prob = (w1 * elo_prob + w2 * lr_prob + w3 * xgb_prob + 
              w4 * lgb_prob + w5 * rf_prob)
# Where weights sum to 1 and are optimized on validation set
```

---

## Training Strategy

### Data Splitting (Time Series)

**Critical**: Never use future data to predict past games. Always split chronologically.

```
Season 2020-21 ──► Training
Season 2021-22 ──► Training
Season 2022-23 ──► Training
Season 2023-24 ──► Validation (hyperparameter tuning)
Season 2024-25 ──► Test (final evaluation)
Season 2025-26 ──► Live predictions
```

### Walk-Forward Validation

For in-season evaluation:
1. Train on all data up to game day N
2. Predict game day N+1
3. Add game day N+1 results to training data
4. Repeat

This mimics real-world deployment and gives realistic accuracy estimates.

### Cross-Validation for Hyperparameters

Use `TimeSeriesSplit` from scikit-learn:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### Retraining Schedule
- **Full retrain**: Weekly (incorporate latest week of data)
- **Model update**: Daily (add new game results, update rolling features)
- **Hyperparameter retune**: Monthly or when performance degrades

---

## Evaluation Metrics

### Game Outcome Predictions

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | % of games correctly predicted | > 65% |
| **Log Loss** | Penalizes confident wrong predictions | < 0.60 |
| **Brier Score** | Mean squared error of probabilities | < 0.22 |
| **AUC-ROC** | Discrimination ability | > 0.70 |
| **Calibration** | Predicted prob ≈ actual win rate | Calibration curve close to diagonal |

### Player Prop Predictions (Pick 6)

| Metric | Description | Target |
|--------|-------------|--------|
| **Over/Under Accuracy** | % of correct more/less calls | > 55% |
| **MAE** | Mean absolute error on stat prediction | Varies by stat |
| **Hit Rate by Confidence** | Accuracy in top-confidence picks | > 60% |
| **Pick 6 ROI** | Return on investment for recommended entries | Positive |

### Benchmarks to Beat

| Benchmark | Expected Accuracy |
|-----------|------------------|
| Coin flip | 50.0% |
| Always pick home team | ~57% |
| Vegas closing line | ~68-70% |
| Elo-only model | ~63-65% |
| **Our target** | **> 65%** |

---

## Model Persistence

### Saving Models
```python
import joblib

# Save trained model
joblib.dump(model, 'models/xgb_game_outcome_v1.pkl')

# Save with metadata
model_meta = {
    'model': model,
    'features': feature_list,
    'trained_date': '2025-26',
    'accuracy': 0.67,
    'version': 'v1'
}
joblib.dump(model_meta, 'models/xgb_game_outcome_v1.pkl')
```

### Model Versioning
- Use date-stamped filenames: `xgb_game_v20260323.pkl`
- Keep a model registry (CSV or JSON) tracking versions, accuracy, and deployed status
- Store in `models/` directory (gitignored if large, or use Git LFS)

### Model Loading in Streamlit
```python
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)
```
