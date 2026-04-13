"""
Loader for sportsdataverse/hoopR-data pre-built parquet files.

NBA team box, player box, PBP, and schedule data sourced from ESPN (2002–present).
All files are cached locally under data_files/hoopr/ — no live download happens
at Streamlit page-load time.

Season integer convention: 2025 = 2024-25 season (ending year).
"""

import io
from pathlib import Path

import pandas as pd
import requests

DATA_DIR    = Path("data_files")
HOOPR_CACHE = DATA_DIR / "hoopr"
HOOPR_CACHE.mkdir(parents=True, exist_ok=True)

# Try GitHub releases first (primary sportsdataverse distribution method),
# then fall back to raw main-branch files.
_URL_PATTERNS: list[str] = [
    "https://github.com/sportsdataverse/hoopR-data/releases/download/nba_{tag}/nba_{tag}_{season}.parquet",
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-data/main/nba/{folder}/nba_{tag}_{season}.parquet",
]

_DATASETS: dict[str, dict] = {
    "team_box":   {"tag": "team_box",   "folder": "team_box"},
    "player_box": {"tag": "player_box", "folder": "player_box"},
    "pbp":        {"tag": "pbp",        "folder": "pbp"},
    "schedule":   {"tag": "schedule",   "folder": "schedules"},
}

# Abbreviations that differ between nba_api (training data) and ESPN/hoopR
_NBA_TO_ESPN: dict[str, str] = {
    "GSW": "GS",
    "NOP": "NO",
    "NYK": "NY",
    "SAS": "SA",
}
_ESPN_TO_NBA: dict[str, str] = {v: k for k, v in _NBA_TO_ESPN.items()}


def normalize_abbr_to_nba(abbr: str) -> str:
    """Convert ESPN abbreviation to nba_api style (e.g. 'GS' → 'GSW')."""
    if not isinstance(abbr, str):
        return str(abbr)
    return _ESPN_TO_NBA.get(abbr.upper(), abbr.upper())


def normalize_abbr_to_espn(abbr: str) -> str:
    """Convert nba_api abbreviation to ESPN style (e.g. 'GSW' → 'GS')."""
    if not isinstance(abbr, str):
        return str(abbr)
    return _NBA_TO_ESPN.get(abbr.upper(), abbr.upper())


def season_str_to_int(season: str) -> int:
    """Convert '2024-25' → 2025, '2025-26' → 2026."""
    return int(season.split("-")[0]) + 1


def _download_parquet(url: str, timeout: int = 120) -> pd.DataFrame:
    """Download a parquet file via HTTP, following GitHub release redirects."""
    resp = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def load_hoopr_parquet(dataset: str, season: int, force: bool = False) -> pd.DataFrame:
    """
    Load a sportsdataverse hoopR parquet from disk cache or GitHub.

    Parameters
    ----------
    dataset : 'team_box', 'player_box', 'pbp', or 'schedule'
    season  : ending year (e.g. 2025 for 2024-25)
    force   : re-download even if already cached

    Returns
    -------
    pd.DataFrame — empty DataFrame on all-URL failure (never raises).
    """
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(_DATASETS)}")

    info  = _DATASETS[dataset]
    fname = f"nba_{info['tag']}_{season}.parquet"
    local = HOOPR_CACHE / fname

    if local.exists() and not force:
        try:
            return pd.read_parquet(local)
        except Exception:
            local.unlink(missing_ok=True)  # corrupt cache — re-download

    errors: list[str] = []
    for pattern in _URL_PATTERNS:
        url = pattern.format(tag=info["tag"], folder=info["folder"], season=season)
        try:
            df = _download_parquet(url)
            HOOPR_CACHE.mkdir(parents=True, exist_ok=True)
            df.to_parquet(local, index=False)
            return df
        except Exception as exc:
            errors.append(f"  {url}: {exc}")

    print(f"[hoopr] WARNING: could not load {dataset} {season}:\n" + "\n".join(errors))
    return pd.DataFrame()


def load_hoopr_seasons(
    dataset: str,
    seasons: list[int],
    force: bool = False,
) -> pd.DataFrame:
    """Load and concatenate multiple seasons. Silently skips failed downloads."""
    frames = [load_hoopr_parquet(dataset, s, force=force) for s in seasons]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_hoopr_cache_path(dataset: str, season: int) -> Path:
    """Return the local cache path for a given dataset/season."""
    info = _DATASETS.get(dataset, {"tag": dataset})
    return HOOPR_CACHE / f"nba_{info['tag']}_{season}.parquet"


def get_pbp_features_path(season: int) -> Path:
    """Pre-aggregated per-team-per-game PBP features (much smaller than raw PBP)."""
    return HOOPR_CACHE / f"nba_pbp_team_features_{season}.parquet"


def load_hoopr_team_box_all(season_ints: list[int]) -> pd.DataFrame:
    """
    Load team box parquets for multiple seasons from disk cache only.
    Never makes a network call — safe to use at Streamlit page-load time.
    """
    frames = []
    for s in season_ints:
        path = get_hoopr_cache_path("team_box", s)
        if path.exists():
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_pbp_features_all(season_ints: list[int]) -> pd.DataFrame:
    """
    Load pre-aggregated PBP team features from disk cache only.
    Never makes a network call.
    """
    frames = []
    for s in season_ints:
        path = get_pbp_features_path(s)
        if path.exists():
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
