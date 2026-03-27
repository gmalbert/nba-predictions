"""
scripts/daily_update.py

Orchestrates the complete daily data refresh pipeline:

  1. Fetch today's odds (sbrscrape + The Odds API fallback)
  2. Fetch any missing historical game logs for the current season
  3. Optionally scrape external data (nbastuffer, databallr)
  4. Pre-warm the Streamlit prediction cache

Run this once per day before the first scheduled tip-off, e.g.:
    python scripts/daily_update.py

Environment variables (optional):
    ODDS_API_KEY   — The Odds API key for fallback odds (free tier available at
                     https://the-odds-api.com)

Add to cron / Task Scheduler:
    # example cron — runs daily at 10:30 AM Eastern
    30 10 * * * /path/to/venv/bin/python /path/to/nba-predictions/scripts/daily_update.py
"""

import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def step_fetch_odds():
    """Fetch today's multi-book odds via sbrscrape and persist to parquet."""
    log.info("── Step 1: Fetch today's odds ───────────────────────────────")
    try:
        # Import lazily to avoid crashing the whole pipeline if sbrscrape is absent
        from scripts.fetch_historical_odds import fetch_date_range
        today_str = datetime.today().strftime("%Y-%m-%d")
        rows = fetch_date_range(datetime.today(), datetime.today())
        if rows:
            import pandas as pd
            from utils.data_fetcher import CURRENT_SEASON, HIST_DIR
            df = pd.DataFrame(rows)
            path = HIST_DIR / f"odds_{CURRENT_SEASON.replace('-', '_')}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df]).drop_duplicates(
                    subset=["date", "home_team", "away_team"], keep="last"
                ).reset_index(drop=True)
            df.to_parquet(path, index=False)
            log.info("  ✓ Saved %d odds row(s) to %s", len(df), path)
        else:
            log.warning("  No odds data returned for today.")
    except Exception as exc:
        log.warning("  Odds fetch failed (non-fatal): %s", exc)


def step_fetch_current_season_gamelog():
    """Ensure the current-season game log cache is up to date."""
    log.info("── Step 2: Refresh current-season game log ──────────────────")
    try:
        from utils.data_fetcher import get_league_game_log, CURRENT_SEASON, HIST_DIR
        # Remove stale disk cache so _read_or_fetch re-fetches from nba_api
        for season_type in ("Regular_Season", "Playoffs"):
            stale = HIST_DIR / f"league_gamelog_{CURRENT_SEASON.replace('-', '_')}_{season_type}.parquet"
            if stale.exists():
                stale.unlink()
                log.info("  Removed stale cache: %s", stale.name)

        df = get_league_game_log(CURRENT_SEASON, "Regular Season")
        log.info("  ✓ Loaded %d game rows for %s", len(df), CURRENT_SEASON)
    except Exception as exc:
        log.error("  Game log refresh failed: %s", exc)


def step_scrape_external():
    """Refresh nbastuffer / databallr external HTML data."""
    log.info("── Step 3: Scrape external data (nbastuffer / databallr) ────")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "scrape_external.py")],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            log.info("  ✓ scrape_external.py succeeded.")
        else:
            log.warning("  scrape_external.py returned non-zero: %s", result.stderr[:300])
    except Exception as exc:
        log.warning("  External scrape failed (non-fatal): %s", exc)


def step_prewarm_cache():
    """Pre-warm the prediction cache so the first Streamlit user doesn't wait."""
    log.info("── Step 4: Pre-warm prediction cache ───────────────────────")
    try:
        from utils.data_fetcher import get_today_predictions
        preds = get_today_predictions()
        log.info("  ✓ %d game prediction(s) cached.", len(preds))
    except Exception as exc:
        log.warning("  Cache pre-warm failed (non-fatal): %s", exc)


def main():
    log.info("=" * 60)
    log.info("NBA Predictions — Daily Update Pipeline")
    log.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    t0 = time.time()

    step_fetch_odds()
    step_fetch_current_season_gamelog()
    step_scrape_external()
    step_prewarm_cache()

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Daily update complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
