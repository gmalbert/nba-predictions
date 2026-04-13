"""
Download sportsdataverse/hoopR-data parquets and pre-aggregate heavy datasets.

Run nightly during the NBA season to keep the local cache fresh.
All data is committed to the repo under data_files/hoopr/ so Streamlit Cloud
never needs to make network calls at page-load time.

Usage:
    python scripts/fetch_hoopr_data.py                 # current season only
    python scripts/fetch_hoopr_data.py --all-seasons   # all configured seasons
    python scripts/fetch_hoopr_data.py --force         # re-download even if cached
    python scripts/fetch_hoopr_data.py --skip-pbp      # skip large PBP files
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.hoopr_fetcher import (
    get_pbp_features_path,
    load_hoopr_parquet,
    season_str_to_int,
)
from utils.data_fetcher import CURRENT_SEASON

# Seasons to download team_box and player_box for
_TEAM_BOX_SEASONS_STR = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
# PBP is large (~200 MB/season) — only aggregate recent seasons
_PBP_SEASONS_STR = ["2022-23", "2023-24", "2024-25", "2025-26"]


def log(msg: str) -> None:
    print(f"[fetch_hoopr] {msg}", flush=True)


# ── Download helpers ───────────────────────────────────────────────────────────

def fetch_team_box(seasons: list[int], force: bool = False) -> None:
    log("── Team Box ──")
    for season in seasons:
        try:
            df = load_hoopr_parquet("team_box", season, force=force)
            if df.empty:
                log(f"  team_box {season}: empty / not available yet")
            else:
                log(f"  team_box {season}: {len(df):,} rows, {len(df.columns)} cols ✓")
        except Exception as exc:
            log(f"  team_box {season}: FAILED — {exc}")
        time.sleep(1.0)


def fetch_player_box(seasons: list[int], force: bool = False) -> None:
    log("── Player Box ──")
    for season in seasons:
        try:
            df = load_hoopr_parquet("player_box", season, force=force)
            if df.empty:
                log(f"  player_box {season}: empty / not available yet")
            else:
                log(f"  player_box {season}: {len(df):,} rows ✓")
        except Exception as exc:
            log(f"  player_box {season}: FAILED — {exc}")
        time.sleep(1.0)


# ── PBP aggregation ────────────────────────────────────────────────────────────

def _clutch_pts(gdf: pd.DataFrame) -> float:
    """Points scored in final 5 min of regulation/OT when score is within 5."""
    if "start_game_seconds_remaining" not in gdf.columns:
        return 0.0
    mask = (gdf["start_game_seconds_remaining"].fillna(999) <= 300) & (
        pd.to_numeric(gdf.get("period", pd.Series([0])), errors="coerce").fillna(0) >= 4
    )
    if "home_score" in gdf.columns and "away_score" in gdf.columns:
        diff = abs(
            pd.to_numeric(gdf["home_score"], errors="coerce") -
            pd.to_numeric(gdf["away_score"], errors="coerce")
        )
        mask = mask & (diff.fillna(99) <= 5)
    clutch = gdf[mask & gdf["scoring_play"]]
    return float(clutch["score_value"].sum())


def _transition_pct(scoring: pd.DataFrame) -> float:
    """Fraction of scoring plays within 7 seconds of clock start in the period."""
    if scoring.empty:
        return 0.0
    s_col = "start_game_seconds_remaining"
    e_col = "end_game_seconds_remaining"
    if s_col not in scoring.columns or e_col not in scoring.columns:
        return float("nan")
    elapsed = (
        pd.to_numeric(scoring[s_col], errors="coerce") -
        pd.to_numeric(scoring[e_col], errors="coerce")
    )
    return float((elapsed.fillna(99) <= 7).sum() / max(len(scoring), 1))


def _run_stats(game_all_plays: pd.DataFrame, team_id: str) -> dict:
    """Return count of 6+ point unanswered runs and the largest run for team_id."""
    scoring = game_all_plays[game_all_plays["scoring_play"]].copy()
    if scoring.empty:
        return {"run_count": 0, "max_run": 0}
    if "start_game_seconds_remaining" in scoring.columns:
        scoring = scoring.sort_values("start_game_seconds_remaining", ascending=False)

    run_count = max_run = 0
    current_team: str | None = None
    current_run = 0.0

    for _, play in scoring.iterrows():
        pt  = str(play.get("team_id", ""))
        pts = float(play.get("score_value", 0) or 0)
        if pt == current_team:
            current_run += pts
        else:
            if current_team == team_id and current_run >= 6:
                run_count += 1
                max_run = max(max_run, int(current_run))
            current_run  = pts
            current_team = pt

    if current_team == team_id and current_run >= 6:
        run_count += 1
        max_run = max(max_run, int(current_run))

    return {"run_count": run_count, "max_run": max_run}


def _shot_distance_stats(shots: pd.DataFrame) -> dict:
    """
    Estimate shot location stats from ESPN coordinate_x / coordinate_y.

    ESPN half-court coords: basket roughly at (0, 25) in feet.
    """
    empty = {"avg_shot_dist": float("nan"), "pct_paint_shots": float("nan"), "pct_three_range": float("nan")}
    if shots.empty or "coordinate_x" not in shots.columns or "coordinate_y" not in shots.columns:
        return empty
    x = pd.to_numeric(shots["coordinate_x"], errors="coerce")
    y = pd.to_numeric(shots["coordinate_y"], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return empty
    x, y = x[valid], y[valid]
    dist = np.sqrt(x ** 2 + (y - 25) ** 2)
    return {
        "avg_shot_dist":    float(dist.mean()),
        "pct_paint_shots":  float((dist <= 8).mean()),
        "pct_three_range":  float((dist >= 22).mean()),
    }


def aggregate_pbp_to_team_features(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ESPN play-by-play into per-team-per-game features.

    Output columns (per game_id / team_id):
      game_date, run_count_6plus, max_run_scored, clutch_pts,
      transition_pts_pct, avg_shot_dist, pct_paint_shots, pct_three_range
    """
    if pbp_df.empty:
        return pd.DataFrame()

    pbp = pbp_df.copy()
    pbp.columns = pbp.columns.str.lower()

    required = {"game_id", "team_id", "scoring_play"}
    if not required.issubset(set(pbp.columns)):
        log(f"  PBP missing required columns: {required - set(pbp.columns)}")
        return pd.DataFrame()

    pbp["scoring_play"] = pbp["scoring_play"].astype(bool)
    if "shooting_play" in pbp.columns:
        pbp["shooting_play"] = pbp["shooting_play"].astype(bool)
    pbp["score_value"] = pd.to_numeric(pbp.get("score_value", 0), errors="coerce").fillna(0)

    if "game_date" in pbp.columns:
        pbp["game_date"] = pd.to_datetime(pbp["game_date"]).dt.normalize()

    results = []
    game_ids = pbp["game_id"].unique()

    for gid in game_ids:
        game_df = pbp[pbp["game_id"] == gid]
        teams = game_df["team_id"].dropna().unique()

        for tid in teams:
            gdf = game_df[game_df["team_id"] == tid]
            row: dict = {"game_id": str(gid), "team_id": str(tid)}

            if "game_date" in game_df.columns and game_df["game_date"].notna().any():
                row["game_date"] = game_df["game_date"].dropna().iloc[0]

            row["clutch_pts"]        = _clutch_pts(gdf)
            scoring = gdf[gdf["scoring_play"]]
            row["transition_pts_pct"] = _transition_pct(scoring)
            runs = _run_stats(game_df, str(tid))
            row["run_count_6plus"]   = runs["run_count"]
            row["max_run_scored"]    = runs["max_run"]

            if "shooting_play" in gdf.columns:
                shots = gdf[gdf["shooting_play"]]
                row.update(_shot_distance_stats(shots))
            else:
                row.update({"avg_shot_dist": float("nan"), "pct_paint_shots": float("nan"), "pct_three_range": float("nan")})

            results.append(row)

    return pd.DataFrame(results)


def fetch_and_aggregate_pbp(seasons: list[int], force: bool = False) -> None:
    """Download raw PBP and save pre-aggregated team-game features only."""
    log("── PBP (download + aggregate) ──")
    for season in seasons:
        out_path = get_pbp_features_path(season)
        if out_path.exists() and not force:
            log(f"  pbp_features {season}: already cached — skipping")
            continue
        try:
            log(f"  pbp {season}: downloading (large file, may take a minute)...")
            pbp = load_hoopr_parquet("pbp", season, force=True)
            if pbp.empty:
                log(f"  pbp {season}: empty / not available")
                continue
            log(f"  pbp {season}: {len(pbp):,} plays — aggregating to team features...")
            feats = aggregate_pbp_to_team_features(pbp)
            if not feats.empty:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                feats.to_parquet(out_path, index=False)
                log(f"  pbp_features {season}: {len(feats):,} team-game rows saved ✓")
            else:
                log(f"  pbp_features {season}: aggregation returned empty")
        except Exception as exc:
            log(f"  pbp {season}: FAILED — {exc}")
        time.sleep(2.0)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download hoopR/sportsdataverse parquets")
    parser.add_argument("--all-seasons", action="store_true",
                        help="Download all configured seasons (default: current season only)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached")
    parser.add_argument("--skip-pbp", action="store_true",
                        help="Skip PBP download/aggregation (faster)")
    args = parser.parse_args()

    current_int = season_str_to_int(CURRENT_SEASON)

    if args.all_seasons:
        team_box_ints = [season_str_to_int(s) for s in _TEAM_BOX_SEASONS_STR]
        pbp_ints      = [season_str_to_int(s) for s in _PBP_SEASONS_STR]
    else:
        team_box_ints = [current_int]
        pbp_ints      = [current_int]

    fetch_team_box(team_box_ints, force=args.force)
    fetch_player_box(team_box_ints, force=args.force)

    if not args.skip_pbp:
        fetch_and_aggregate_pbp(pbp_ints, force=args.force)

    log("✓ Done.")


if __name__ == "__main__":
    main()
