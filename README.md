# NBA Predictions 🏀

A Streamlit-powered data analytics platform for NBA game predictions and DraftKings Pick 6 analysis.

## Features

- **Game Predictions**: Win probabilities, predicted spreads & totals for every NBA game
- **Pick 6 Analysis**: Player prop modeling with over/under probabilities and entry building tools
- **Team Stats**: Interactive team dashboards with rolling averages, rankings, and comparisons
- **Player Stats**: Individual player analysis, game logs, splits, and trend charts
- **Model Performance**: Accuracy tracking, calibration plots, and backtesting results

## Data Sources

- [**nba_api**](https://github.com/swar/nba_api) — Primary data source for game data, box scores, player stats, and league standings
- **Basketball Reference** — Advanced metrics, four factors, and historical data
- **The Odds API** — Market odds for model calibration and edge detection
- **ESPN / Rotowire** — Injury reports and lineup confirmations

## Tech Stack

| Category | Tools |
|----------|-------|
| **App Framework** | Streamlit |
| **Data** | nba_api, BeautifulSoup4, requests |
| **Analysis** | pandas, NumPy, SciPy, statsmodels |
| **ML Models** | scikit-learn, XGBoost, LightGBM |
| **Visualization** | Plotly, Matplotlib, Seaborn |

## Setup

### Prerequisites
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/gmalbert/nba-predictions.git
cd nba-predictions

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run predictions.py
```

## Deployment

This app is deployed on **Streamlit Cloud**. To deploy your own instance:

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `predictions.py` as the entry point
4. Add any API keys in the Streamlit Cloud Secrets settings

## Project Structure

```
nba-predictions/
├── .streamlit/              # Streamlit theme configuration
├── data_files/
│   └── logo.png             # Site logo
├── docs/                    # Project roadmap & documentation
│   ├── data_sources.md      # Data sourcing strategy
│   ├── features.md          # Feature engineering plan
│   ├── layout.md            # Site layout & UX design
│   ├── models.md            # ML models & training strategy
│   └── predictions.md       # Prediction methodology
├── pages/                   # Streamlit multi-page app (coming soon)
├── predictions.py           # Main Streamlit entry point
├── requirements.txt         # Python dependencies
└── README.md
```

## Roadmap

See the [docs/](docs/) folder for detailed planning:

- [Data Sources](docs/data_sources.md) — nba_api endpoints, scraping targets, odds APIs
- [Features](docs/features.md) — Team, player, matchup, and Pick 6 feature engineering
- [Layout](docs/layout.md) — Streamlit page structure and UI design
- [Models](docs/models.md) — ML model selection, training, and evaluation
- [Predictions](docs/predictions.md) — Game outcome and Pick 6 prediction methodology

## License

This project is for informational and entertainment purposes only. Please gamble responsibly.
