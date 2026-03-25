"""
Historical NBA data download script.

Pulls 5 seasons (2021-22 through 2025-26) of data from nba_api and saves
to data_files/historical/ as parquet files.

Usage:
    python scripts/fetch_historical.py                   # Game logs + stats only
    python scripts/fetch_historical.py --boxscores       # Also fetch all box scores (slow)
    python scripts/fetch_historical.py --season 2024-25  # Box scores for one season only

NOTE: The base fetch (no --boxscores) makes ~20 API calls and takes ~30 seconds.
      With --boxscores it makes ~6000+ calls and may take 60-90 minutes.
      Run once; all data is cached to disk and subsequent runs are instant.
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure repo root is on the path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_fetcher import (
    get_league_game_log,
    get_league_team_stats,
    get_league_player_stats,
    get_team_estimated_metrics,
    get_box_score_traditional,
    get_box_score_advanced,
    HISTORICAL_SEASONS,
    HIST_DIR,
)


def log(msg: str):
    print(f"[fetch_historical] {msg}", flush=True)


def fetch_season_summaries():
    """
    Download league-level summaries for all 5 historical seasons.
    These are the four files needed for model training:
      - league_gamelog  : per-game team stats for every game played
      - league_teamstats: season-averaged stats for all 30 teams
      - league_playerstats: season-averaged stats for all players
      - team_est_metrics : ORtg / DRtg / pace / net rating
    """
    log(f"Fetching summaries for seasons: {HISTORICAL_SEASONS}")
    log(f"Cache directory: {HIST_DIR.resolve()}\n")

    for season in HISTORICAL_SEASONS:
        log(f"{'-'*50}")
        log(f"Season: {season}")

        # 1. League game log (team perspective, regular season)
        log("  [1/4] League game log (regular season)...")
        try:
            df = get_league_game_log(season, "Regular Season")
            log(f"        ✓ {len(df):,} rows")
        except Exception as e:
            log(f"        ✗ FAILED: {e}")

        # 2. League game log (playoffs)
        log("  [2/4] League game log (playoffs)...")
        try:
            df = get_league_game_log(season, "Playoffs")
            log(f"        ✓ {len(df):,} rows")
        except Exception as e:
            log(f"        ✗ FAILED: {e}")

        # 3. League-wide per-game team stats
        log("  [3/4] League team stats (per game)...")
        try:
            df = get_league_team_stats(season)
            log(f"        ✓ {len(df):,} rows")
        except Exception as e:
            log(f"        ✗ FAILED: {e}")

        # 4. League-wide per-game player stats
        log("  [4/4] League player stats (per game)...")
        try:
            df = get_league_player_stats(season)
            log(f"        ✓ {len(df):,} rows")
        except Exception as e:
            log(f"        ✗ FAILED: {e}")

        # 5. Team estimated metrics (ORtg / DRtg / pace)
        log("  [5/5] Team estimated metrics...")
        try:
            df = get_team_estimated_metrics(season)
            log(f"        ✓ {len(df):,} rows")
        except Exception as e:
            log(f"        ✗ FAILED: {e}")

    log("\n✓ Summary fetch complete.")


def fetch_box_scores(seasons: list[str]):
    """
    Download individual box scores for every game in the specified seasons.
    This is optional — only needed for play-level or lineup-level analysis.

    WARNING: ~1,200 games/season × 2 calls (trad + adv) × 0.7s = ~28 min/season.
    """
    try:
        from tqdm import tqdm
        USE_TQDM = True
    except ImportError:
        USE_TQDM = False

    for season in seasons:
        log(f"\nFetching box scores for {season}...")
        game_log = get_league_game_log(season)
        if game_log.empty:
            log(f"  No game log found for {season}. Skipping.")
            continue

        game_ids = game_log["GAME_ID"].unique().tolist()
        log(f"  {len(game_ids):,} games to fetch.")

        iterator = tqdm(game_ids, desc=season) if USE_TQDM else game_ids
        fetched = 0
        skipped = 0

        for game_id in iterator:
            game_id = str(game_id)
            trad_exists = (HIST_DIR / f"bst_player_{game_id}.parquet").exists()
            adv_exists = (HIST_DIR / f"bsa_player_{game_id}.parquet").exists()

            if not trad_exists:
                try:
                    get_box_score_traditional(game_id)
                    fetched += 1
                except Exception as e:
                    if not USE_TQDM:
                        log(f"    ✗ Trad box score {game_id}: {e}")

            if not adv_exists:
                try:
                    get_box_score_advanced(game_id)
                    fetched += 1
                except Exception as e:
                    if not USE_TQDM:
                        log(f"    ✗ Adv box score {game_id}: {e}")

            if trad_exists and adv_exists:
                skipped += 1

        log(f"  ✓ {fetched} fetched, {skipped} already cached.")

    log("\n✓ Box score fetch complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download 5 years of historical NBA data from nba_api"
    )
    parser.add_argument(
        "--boxscores",
        action="store_true",
        help="Also fetch individual box scores for every game (slow — ~60-90 minutes total)",
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Limit --boxscores to a single season (e.g. 2024-25)",
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        default=False,
        help="Only fetch season summaries (default behavior, explicit flag)",
    )
    args = parser.parse_args()

    start = time.time()
    fetch_season_summaries()

    if args.boxscores:
        seasons = [args.season] if args.season else HISTORICAL_SEASONS
        fetch_box_scores(seasons)

    elapsed = time.time() - start
    log(f"\nTotal time: {elapsed / 60:.1f} minutes")
