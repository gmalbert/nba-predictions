"""
Pre-warm all disk-based parquet caches for Streamlit pages.

Run this after the nightly data pipeline so Streamlit Cloud reads every
dataset directly from committed parquet files rather than making live
NBA API calls.

Usage:
    python scripts/preload_cache.py            # full preload for today
    python scripts/preload_cache.py --no-preds # skip prediction pipeline
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TODAY = datetime.today().strftime("%Y-%m-%d")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(label: str, fn, *args, **kwargs):
    log.info(f"  [{label}] fetching ...")
    try:
        df = fn(*args, **kwargs)
        rows = len(df) if isinstance(df, pd.DataFrame) else "n/a"
        log.info(f"  [{label}] {rows} rows ✓")
        return df
    except Exception as exc:
        log.warning(f"  [{label}] FAILED: {exc}")
        return pd.DataFrame()


# ── Step 1: Refresh current-season NBA API parquets ───────────────────────────

def refresh_current_season(season: str, hist_dir: Path):
    """Delete stale current-season parquets then re-fetch fresh from NBA API."""
    slug = season.replace("-", "_")
    stale_files = [
        hist_dir / f"league_gamelog_{slug}_Regular_Season.parquet",
        hist_dir / f"league_teamstats_{slug}_Regular_Season.parquet",
        hist_dir / f"league_playerstats_{slug}_Regular_Season.parquet",
        hist_dir / f"team_est_metrics_{slug}.parquet",
    ]
    for p in stale_files:
        if p.exists():
            p.unlink()
            log.info(f"  [stale] removed {p.name}")

    from utils.data_fetcher import (
        get_league_game_log,
        get_league_team_stats,
        get_league_player_stats,
        get_team_estimated_metrics,
    )

    log.info("── Current-season NBA API data ──")
    _run("game log",          get_league_game_log,          season, "Regular Season")
    time.sleep(0.6)
    _run("team stats",        get_league_team_stats,        season)
    time.sleep(0.6)
    _run("player stats",      get_league_player_stats,      season)
    time.sleep(0.6)
    _run("estimated metrics", get_team_estimated_metrics,   season)
    time.sleep(0.6)


# ── Step 2: Standings ─────────────────────────────────────────────────────────

def refresh_standings(season: str, hist_dir: Path):
    """Fetch current standings and persist to disk with a FETCH_DATE stamp."""
    path = hist_dir / f"standings_{season.replace('-', '_')}.parquet"

    # Skip if already fresh today
    if path.exists():
        try:
            cached = pd.read_parquet(path)
            if (
                "FETCH_DATE" in cached.columns
                and str(cached["FETCH_DATE"].iloc[0])[:10] == TODAY
            ):
                log.info("  [standings] already fresh for today — skipping")
                return
        except Exception:
            pass

    log.info("── Standings ──")
    from nba_api.stats.endpoints import leaguestandingsv3

    time.sleep(0.6)
    try:
        raw = leaguestandingsv3.LeagueStandingsV3(season=season)
        df = raw.standings.get_data_frame()
        if not df.empty:
            df["FETCH_DATE"] = TODAY
            df.to_parquet(path, index=False)
            log.info(f"  [standings] {len(df)} rows saved ✓")
    except Exception as exc:
        log.warning(f"  [standings] FAILED: {exc}")


# ── Step 3: Game predictions ──────────────────────────────────────────────────

def refresh_predictions(hist_dir: Path):
    """Pre-run today's prediction pipeline and save to disk."""
    from utils.data_fetcher import run_and_cache_predictions

    path = hist_dir / f"predictions_{TODAY}.parquet"
    if path.exists():
        log.info("  [predictions] already cached for today — skipping")
        return

    log.info("── Game predictions ──")
    df = _run("predictions", run_and_cache_predictions, TODAY)
    if not df.empty:
        log.info(f"  [predictions] {len(df)} games pre-cached ✓")
    else:
        log.info("  [predictions] no games today or pipeline returned empty")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-warm Streamlit page caches")
    parser.add_argument(
        "--no-preds",
        action="store_true",
        help="Skip the prediction pipeline (faster, use when models are not yet trained)",
    )
    parser.add_argument(
        "--preds-only",
        action="store_true",
        help="Only refresh predictions, skip nba_api + standings fetch",
    )
    args = parser.parse_args()

    from utils.data_fetcher import CURRENT_SEASON, HIST_DIR

    log.info("=" * 60)
    log.info(f"Cache preload  —  {TODAY}")
    log.info("=" * 60)

    if not args.preds_only:
        refresh_current_season(CURRENT_SEASON, HIST_DIR)
        refresh_standings(CURRENT_SEASON, HIST_DIR)

    if not args.no_preds:
        refresh_predictions(HIST_DIR)

    log.info("=" * 60)
    log.info("Preload complete.")


if __name__ == "__main__":
    main()
