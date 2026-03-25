"""
scripts/scrape_external.py

Scrapes supplemental NBA data from external sources across all historical
seasons (2021-22 through 2025-26).

Sources
-------
  1. nbastuffer.com  – player stats, team stats, referee stats, rest-days stats
  2. databallr.com   – shot quality, team metrics  (requires: playwright)
  3. ESPN API        – today's referee assignments (daily, pure JSON — no playwright)

Strategy
--------
  * Raw HTML is cached to data_files/raw_html/  (one file per page+season).
  * Parsed DataFrames are saved as parquet to data_files/historical/.
  * Re-running the script is idempotent; use --refresh to force re-download.

Usage
-----
  python scripts/scrape_external.py                       # scrape everything
  python scripts/scrape_external.py --source nbastuffer   # only nbastuffer
  python scripts/scrape_external.py --source databallr    # only databallr
  python scripts/scrape_external.py --source refs         # today's ref assignments
  python scripts/scrape_external.py --season 2025-26      # single season
  python scripts/scrape_external.py --refresh             # ignore HTML cache

Column fallbacks
----------------
  If <th> elements in the HTML are empty, predefined fallback column names
  are applied per page type.  Override by inspecting the saved HTML in
  data_files/raw_html/ and adjusting the FALLBACK_COLS dicts below.
"""

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Directories ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW_HTML_DIR = ROOT / "data_files" / "raw_html"
HIST_DIR = ROOT / "data_files" / "historical"
RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
HIST_DIR.mkdir(parents=True, exist_ok=True)

# ── Season list ────────────────────────────────────────────────────────────────

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# ── HTTP headers ───────────────────────────────────────────────────────────────

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    ),
    "Referer": "https://www.nbastuffer.com/",
}

_SCRAPE_DELAY = 2.5  # polite delay between HTTP requests (seconds)

# ── Fallback column names (used when <th> elements are empty) ─────────────────

FALLBACK_COLS = {
    "player": [
        "RANK", "PLAYER", "TEAM", "STATUS", "POS", "AGE", "GP", "MPG",
        "USG_PCT", "TO_PCT", "FTA", "FT_PCT", "FGA_2", "FG_PCT_2",
        "FGA_3", "FG_PCT_3", "EFG_PCT", "TS_PCT", "PTS", "REB", "AST",
        "STL", "BLK", "TOV", "DD2", "TD3", "PRA", "ORTG", "DRTG",
    ],
    "team": [
        "RANK", "TEAM", "CONF", "DIV", "GP", "PPG", "oPPG", "pDIFF",
        "PACE", "oEFF", "dEFF", "eDIFF", "SoS", "rSoS", "SAR", "CONS",
        "a4F", "W", "L", "W_PCT", "eWIN_PCT", "pWIN_PCT", "ACH", "STRK",
    ],
    "referee": [
        "RANK", "REFEREE", "LEVEL", "GENDER", "EXP", "GP", "W_PCT",
        "AED", "PTS", "FOULS_PER_GAME", "HOME_W_PCT", "AWAY_W_PCT",
        "HOME_ADV",
    ],
    "restdays": [
        "RANK", "TEAM", "OPP_TODAY",
        "TODAY_GP", "TODAY_W_PCT", "TODAY_AED",
        "3IN4_B2B_GP", "3IN4_B2B_W_PCT", "3IN4_B2B_AED",
        "B2B_GP", "B2B_W_PCT", "B2B_AED",
        "3IN4_GP", "3IN4_W_PCT", "3IN4_AED",
        "1DAY_GP", "1DAY_W_PCT", "1DAY_AED",
        "2DAY_GP", "2DAY_W_PCT", "2DAY_AED",
        "3PLUS_GP", "3PLUS_W_PCT", "3PLUS_AED",
    ],
}

# nbastuffer URL slug for each page type
_NBASTUFFER_TYPE_SLUG = {
    "player":   "nba-player-stats",
    "team":     "nba-team-stats",
    "referee":  "nba-referee-stats",
    "restdays": "nba-rest-days-stats",
}


# ── Helper utilities ───────────────────────────────────────────────────────────

def _season_to_url_prefix(season: str) -> str:
    """'2021-22' → '2021-2022' for use in nbastuffer URLs."""
    start, end = season.split("-")
    end_4 = start[:2] + end  # '20' + '22' = '2022'
    return f"{start}-{end_4}"


def _season_slug(season: str) -> str:
    """'2021-22' → '2021_22' for file naming."""
    return season.replace("-", "_")


def _download_page(url: str, cache_path: Path, force_refresh: bool = False) -> str:
    """
    Return cached HTML if available, otherwise download and cache.
    Returns the HTML string; returns '' on failure.
    """
    if cache_path.exists() and not force_refresh:
        log.info(f"  [cache] {cache_path.name}")
        return cache_path.read_text(encoding="utf-8", errors="replace")

    log.info(f"  [GET]   {url}")
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text, encoding="utf-8")
        time.sleep(_SCRAPE_DELAY)
        return resp.text
    except requests.RequestException as exc:
        log.error(f"  [FAIL]  {url}: {exc}")
        return ""


# ── HTML table parsing ─────────────────────────────────────────────────────────

def _table_to_df(table, fallback_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Convert a BeautifulSoup <table> element to a DataFrame.

    Header extraction order:
      1. <thead> last <tr> — text of each <th>/<td>
      2. fallback_cols (if all <th> are empty)
      3. Positional integers
    """
    # -- headers --
    headers: list[str] = []
    thead = table.find("thead")
    if thead:
        th_rows = thead.find_all("tr")
        if th_rows:
            last_hrow = th_rows[-1]
            headers = [
                th.get_text(strip=True)
                for th in last_hrow.find_all(["th", "td"])
            ]

    # -- body rows --
    tbody = table.find("tbody") or table
    rows_data: list[list[str]] = []
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows_data.append(cells)

    if not rows_data:
        return pd.DataFrame()

    n_cols = len(rows_data[0])

    # Decide column names
    def _not_empty(lst):
        return any(str(v).strip() for v in lst)

    if headers:
        headers = [_clean_column_name(h) for h in headers]

    if headers and len(headers) == n_cols and _not_empty(headers):
        col_names = headers
    elif fallback_cols and len(fallback_cols) == n_cols:
        col_names = fallback_cols
    elif fallback_cols and len(fallback_cols) > n_cols:
        col_names = fallback_cols[:n_cols]
    else:
        col_names = [str(i) for i in range(n_cols)]
        if headers:
            log.warning(
                f"  Header count {len(headers)} ≠ col count {n_cols}; "
                "using positional names"
            )

    # Normalise row lengths (pad / truncate to match header)
    n = len(col_names)
    rows_data = [r[:n] + [""] * max(0, n - len(r)) for r in rows_data]

    df = pd.DataFrame(rows_data, columns=col_names)
    # Deduplicate column names (e.g. Team repeated on both ends of the row)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.replace("", np.nan).dropna(how="all").reset_index(drop=True)
    return df


def _clean_column_name(name: str) -> str:
    """Strip nbastuffer css3_tooltip shortcodes, keeping just the abbreviated label.

    '[css3_tooltip header=...]PTS/GM[/css3_tooltip]' → 'PTS/GM'
    """
    m = re.search(r'\]([^\[]+)\[/css3_tooltip\]', name)
    if m:
        return m.group(1).strip()
    return name.strip()


def _parse_first_table(html: str, fallback_cols: list[str] | None = None) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        log.warning("  No <table> found in HTML")
        return pd.DataFrame()
    return _table_to_df(table, fallback_cols)


def _parse_all_tables(html: str) -> list[pd.DataFrame]:
    """Return a DataFrame for every <table> in the page."""
    soup = BeautifulSoup(html, "html.parser")
    return [_table_to_df(t) for t in soup.find_all("table")]


# ── nbastuffer scrapers ────────────────────────────────────────────────────────

def scrape_nbastuffer_page(
    season: str,
    page_type: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download and parse one nbastuffer page for a given season.

    page_type: 'player' | 'team' | 'referee' | 'restdays'

    For 'team', all four table splits (Regular Season, Last 5, Road, Home)
    are saved as separate parquet files; the Regular Season table is returned.

    Parquet saved to:
      data_files/historical/nbastuffer_{page_type}_{season_slug}.parquet
      data_files/historical/nbastuffer_teamstats_{season_slug}_{split}.parquet
    """
    if page_type not in _NBASTUFFER_TYPE_SLUG:
        raise ValueError(f"Unknown page_type '{page_type}'")

    url_prefix = _season_to_url_prefix(season)
    url = (
        f"https://www.nbastuffer.com/"
        f"{url_prefix}-{_NBASTUFFER_TYPE_SLUG[page_type]}/"
    )
    ss = _season_slug(season)
    cache_path = RAW_HTML_DIR / f"nbastuffer_{page_type}_{ss}.html"
    parquet_path = HIST_DIR / f"nbastuffer_{page_type}_{ss}.parquet"

    # Fast-path: parquet already exists
    if parquet_path.exists() and not force_refresh:
        log.info(f"  [skip]  {parquet_path.name} already exists")
        return pd.read_parquet(parquet_path)

    html = _download_page(url, cache_path, force_refresh)
    if not html:
        return pd.DataFrame()

    fallback = FALLBACK_COLS.get(page_type)

    if page_type == "team":
        # Team stats page has 4 tables: Regular Season, Last 5, Road, Home
        # Some season pages have an extra partial/conference table first (16 rows);
        # skip any table with fewer than 25 rows to find the real 30-team tables.
        all_tables = _parse_all_tables(html)
        team_tables = [df for df in all_tables if len(df) >= 25]
        split_names = ["regular", "last5", "road", "home"]
        main_df = pd.DataFrame()
        for idx, split in enumerate(split_names):
            if idx < len(team_tables) and not team_tables[idx].empty:
                sdf = team_tables[idx].copy()
                if not sdf.empty and (fallback is None or len(sdf.columns) != len(fallback)):
                    # try fallback if columns mismatch
                    pass
                elif fallback and list(sdf.columns) == [str(i) for i in range(len(sdf.columns))]:
                    sdf.columns = fallback[:len(sdf.columns)]
                sdf["SEASON"] = season
                sdf["SPLIT"] = split
                sdf["SOURCE"] = "nbastuffer"
                split_path = HIST_DIR / f"nbastuffer_teamstats_{ss}_{split}.parquet"
                sdf.to_parquet(split_path, index=False)
                log.info(f"  [save]  {split_path.name}  ({len(sdf)} rows)")
                if split == "regular":
                    main_df = sdf

        # Also save combined as the main parquet
        if not main_df.empty:
            main_df.to_parquet(parquet_path, index=False)
        return main_df

    # All other page types: grab first table
    df = _parse_first_table(html, fallback)
    if df.empty:
        log.warning(f"  [warn]  No data for {page_type} {season}")
        return df

    df["SEASON"] = season
    df["SOURCE"] = "nbastuffer"

    df.to_parquet(parquet_path, index=False)
    log.info(f"  [save]  {parquet_path.name}  ({len(df)} rows, {len(df.columns)} cols)")
    return df


def scrape_nbastuffer_all(
    seasons: list[str],
    force_refresh: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Scrape all nbastuffer page types for all seasons."""
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for season in seasons:
        results[season] = {}
        for page_type in _NBASTUFFER_TYPE_SLUG:
            log.info(f"nbastuffer | {season} | {page_type}")
            try:
                df = scrape_nbastuffer_page(season, page_type, force_refresh)
                results[season][page_type] = df
            except Exception as exc:
                log.error(f"  [ERROR] {page_type} {season}: {exc}")
                results[season][page_type] = pd.DataFrame()
    return results


# ── databallr scrapers (playwright) ───────────────────────────────────────────

_DATABALLR_URLS = {
    "teams": "https://databallr.com/teams",
}


def scrape_databallr_page(
    page_type: str,
    season: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download and parse a databallr.com page using playwright.

    Requires:
        pip install playwright
        playwright install chromium

    page_type : 'shotquality' | 'teams'
    season    : if provided, attempt to select that season in the UI.
                Defaults to current/latest available.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error(
            "playwright is not installed.  Run:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )
        return pd.DataFrame()

    url = _DATABALLR_URLS.get(page_type)
    if url is None:
        log.error(f"Unknown databallr page_type: {page_type}")
        return pd.DataFrame()

    ss = _season_slug(season) if season else "current"
    cache_path = RAW_HTML_DIR / f"databallr_{page_type}_{ss}.html"
    parquet_path = HIST_DIR / f"databallr_{page_type}_{ss}.parquet"

    if parquet_path.exists() and not force_refresh:
        log.info(f"  [skip]  {parquet_path.name} already exists")
        return pd.read_parquet(parquet_path)

    html = ""
    if cache_path.exists() and not force_refresh:
        log.info(f"  [cache] {cache_path.name}")
        html = cache_path.read_text(encoding="utf-8", errors="replace")
    else:
        log.info(f"  [playwright] {url}")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({
                "User-Agent": _HTTP_HEADERS["User-Agent"],
                "Accept-Language": "en-US,en;q=0.9",
            })
            try:
                page.goto(url, wait_until="networkidle", timeout=45_000)

                # If a season is specified, attempt to click/select it.
                # databallr uses season buttons / dropdowns — try common patterns.
                if season:
                    season_label = _season_to_url_prefix(season)  # e.g. "2025-2026"
                    selectors = [
                        f"button:has-text('{season_label}')",
                        f"option:has-text('{season_label}')",
                        f"[data-season='{season_label}']",
                        f"a:has-text('{season_label}')",
                    ]
                    for sel in selectors:
                        try:
                            elem = page.locator(sel).first
                            if elem.count() > 0:
                                elem.click()
                                page.wait_for_load_state("networkidle", timeout=10_000)
                                log.info(f"  Selected season {season_label}")
                                break
                        except Exception:
                            pass

                # Wait for at least one table to appear
                try:
                    page.wait_for_selector("table", timeout=15_000)
                except Exception:
                    log.warning("  No <table> appeared within timeout; saving raw page")

                html = page.content()
                cache_path.write_text(html, encoding="utf-8")
                time.sleep(_SCRAPE_DELAY)
            except Exception as exc:
                log.error(f"  [playwright error] {exc}")
                try:
                    html = page.content()
                except Exception:
                    html = ""
            finally:
                browser.close()

    if not html:
        return pd.DataFrame()

    # Try first table; if empty, pick the largest
    df = _parse_first_table(html)
    if df.empty:
        all_dfs = [d for d in _parse_all_tables(html) if not d.empty]
        if all_dfs:
            df = max(all_dfs, key=len)
    if df.empty:
        log.warning(f"  [warn]  No table data for databallr/{page_type} {season}")
        return df

    if season:
        df["SEASON"] = season
    df["SOURCE"] = "databallr"
    df["PAGE_TYPE"] = page_type

    # For teams page: strip '#N' rank suffixes from numeric-looking values
    # e.g. '118.3#7' -> '118.3', '+11.1#1' -> '+11.1'
    if page_type == "teams":
        for col in df.columns:
            if col not in ("SEASON", "SOURCE", "PAGE_TYPE", "Team", "Rk"):
                df[col] = df[col].astype(str).str.replace(r'#\d+$', '', regex=True)

    df.to_parquet(parquet_path, index=False)
    log.info(f"  [save]  {parquet_path.name}  ({len(df)} rows)")
    return df


def scrape_databallr_all(
    seasons: list[str],
    force_refresh: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Scrape all databallr pages for all seasons."""
    results: dict[str, dict[str, pd.DataFrame]] = {}
    for season in seasons:
        results[season] = {}
        for page_type in _DATABALLR_URLS:
            log.info(f"databallr  | {season} | {page_type}")
            try:
                df = scrape_databallr_page(page_type, season, force_refresh)
                results[season][page_type] = df
            except Exception as exc:
                log.error(f"  [ERROR] {page_type} {season}: {exc}")
                results[season][page_type] = pd.DataFrame()
    return results


# ── ESPN API referee assignments (daily, no playwright required) ──────────────

_ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
_ESPN_SUMMARY    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
_REF_PARQUET     = HIST_DIR / "nba_ref_assignments_today.parquet"


def scrape_today_referee_assignments(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch today's referee assignments from official.nba.com (primary source).

    No playwright required — pure HTTP/JSON.
    official.nba.com is used as the single source of truth for referee names
    and game assignments. NBA game IDs are resolved via nba_api ScoreboardV3.
    Each referee is cross-referenced against the cached nbastuffer season-
    aggregate file for foul-tendency stats.

    Falls back to ESPN game summary API only for any games not yet posted on
    the official page.

    Returns a DataFrame with one row per referee per game:
      NBA_GAME_ID, HOME_TEAM, AWAY_TEAM, REFEREE, ORDER,
      CALLED_FOULS_PER_GAME, HOME_WIN_PCT, FOUL_PCT_ROAD, FOUL_PCT_HOME,
      FOUL_DIFFERENTIAL, EXPERIENCE_YEARS, SOURCE

    Saved to data_files/historical/nba_ref_assignments_today.parquet.
    """
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Return cached version if it exists from today and force_refresh is False
    if _REF_PARQUET.exists() and not force_refresh:
        cached = pd.read_parquet(_REF_PARQUET)
        if (
            "FETCH_DATE" in cached.columns
            and "NBA_GAME_ID" in cached.columns  # reject old schema
            and not cached.empty
            and str(cached["FETCH_DATE"].iloc[0])[:10] == today_str
        ):
            log.info(f"  [cache] {_REF_PARQUET.name} (today, {len(cached)} rows)")
            return cached

    sess = requests.Session()
    sess.headers.update(_HTTP_HEADERS)

    # ── 1. Resolve today's NBA game IDs via nba_api ScoreboardV3 ─────────────
    # Build a list of today's games with tricode + display name for matching.
    nba_games: list[dict] = []
    try:
        from nba_api.stats.endpoints import scoreboardv3 as _sb3
        import time as _time
        _time.sleep(0.6)
        _sb = _sb3.ScoreboardV3(
            game_date=pd.Timestamp.now().strftime("%m/%d/%Y"),
            league_id="00",
        )
        _games_df = _sb.data_sets[1].get_data_frame()
        _teams_df = _sb.data_sets[2].get_data_frame()
        _tricode_to_display = {
            r["teamTricode"]: f"{r['teamCity']} {r['teamName']}".lower()
            for _, r in _teams_df.iterrows()
        }
        for _, _g in _games_df.iterrows():
            _code = _g.get("gameCode", "")
            _parts = _code.split("/")
            if len(_parts) == 2 and len(_parts[1]) == 6:
                _away_tri = _parts[1][:3]
                _home_tri = _parts[1][3:]
                nba_games.append({
                    "nba_game_id":   _g["gameId"],
                    "home_tri":      _home_tri,
                    "away_tri":      _away_tri,
                    "home_display":  _tricode_to_display.get(_home_tri, "").lower(),
                    "away_display":  _tricode_to_display.get(_away_tri, "").lower(),
                })
        log.info(f"  [nba_api] {len(nba_games)} games today")
    except Exception as exc:
        log.warning(f"  [nba_api] could not resolve NBA game IDs: {exc}")

    def _find_nba_game(home_text: str, away_text: str) -> dict | None:
        """Match a home/away display name pair to an NBA game dict."""
        h = home_text.lower()
        a = away_text.lower()
        for ng in nba_games:
            home_match = any(w in ng["home_display"] for w in h.split() if len(w) > 3)
            away_match = any(w in ng["away_display"] for w in a.split() if len(w) > 3)
            if home_match and away_match:
                return ng
        return None

    # ── 2. Load nbastuffer season-aggregate referee stats ─────────────────────
    current_season = _season_slug(SEASONS[-1])
    nbastuffer_ref_path = HIST_DIR / f"nbastuffer_referee_{current_season}.parquet"
    ref_stats: dict[str, dict] = {}
    if nbastuffer_ref_path.exists():
        try:
            rdf = pd.read_parquet(nbastuffer_ref_path)
            for _, row in rdf.iterrows():
                name = str(row.get("REFEREE", "")).strip()
                if name:
                    ref_stats[name.lower()] = row.to_dict()
            log.info(f"  [nbastuffer] loaded {len(ref_stats)} ref stats for cross-reference")
        except Exception as exc:
            log.warning(f"  [nbastuffer] could not load ref stats: {exc}")

    def _ref_stats_row(ref_name: str, nba_game: dict | None, home_full: str, away_full: str,
                       order: int, source: str) -> dict:
        stats = ref_stats.get(ref_name.lower(), {})
        home_tri = nba_game["home_tri"] if nba_game else ""
        away_tri = nba_game["away_tri"] if nba_game else ""
        return {
            "NBA_GAME_ID":            nba_game["nba_game_id"] if nba_game else "",
            "HOME_TEAM":              home_tri,
            "AWAY_TEAM":              away_tri,
            "HOME_TEAM_FULL":         home_full,
            "AWAY_TEAM_FULL":         away_full,
            "REFEREE":                ref_name,
            "ORDER":                  order,
            "CALLED_FOULS_PER_GAME":  stats.get("CALLED FOULSPER GAME", None),
            "HOME_WIN_PCT":           stats.get("HOME TEAMWIN%", None),
            "FOUL_PCT_ROAD":          stats.get("FOUL%AGAINST ROAD TEAMS", None),
            "FOUL_PCT_HOME":          stats.get("FOUL%AGAINST HOME TEAMS", None),
            "FOUL_DIFFERENTIAL":      stats.get("FOUL DIFFERENTIAL(Ag.Rd Tm) - (Ag. Hm Tm)", None),
            "EXPERIENCE_YEARS":       stats.get("EXPERIENCE(YEARS)", None),
            "FETCH_DATE":             today_str,
            "SOURCE":                 source,
        }

    # ── 3. Primary: official.nba.com ──────────────────────────────────────────
    rows: list[dict] = []
    covered_nba_ids: set[str] = set()

    official_df = _parse_nba_official_referee_assignments()
    if not official_df.empty:
        log.info(f"  [nba_official] {len(official_df)} game rows from official.nba.com")
        for _, orow in official_df.iterrows():
            game_text = orow.get("Game", "")
            # game_text is like "Denver @ Phoenix"
            parts = [p.strip() for p in re.split(r"\s*@\s*", game_text)]
            if len(parts) != 2:
                continue
            away_text, home_text = parts
            nba_game = _find_nba_game(home_text, away_text)

            home_full = home_text.title()
            away_full = away_text.title()
            if nba_game:
                # Use canonical display names from nba_api
                home_full = nba_game["home_display"].title()
                away_full = nba_game["away_display"].title()
                covered_nba_ids.add(nba_game["nba_game_id"])

            ref_slots = [
                (1, orow.get("Crew Chief", "")),
                (2, orow.get("Referee", "")),
                (3, orow.get("Umpire", "")),
            ]
            for order, full_name in ref_slots:
                full_name = re.sub(r"\s*\(#\d+\)$", "", str(full_name).strip())
                if not full_name:
                    continue
                rows.append(_ref_stats_row(full_name, nba_game, home_full, away_full,
                                           order, "nba_official"))
    else:
        log.warning("  [nba_official] no data returned; falling back entirely to ESPN")

    # ── 4. Fallback: ESPN summary for any games not on official page ──────────
    uncovered = [ng for ng in nba_games if ng["nba_game_id"] not in covered_nba_ids]
    if uncovered:
        log.info(f"  [ESPN] fetching officials for {len(uncovered)} uncovered game(s)")
        # We need ESPN event IDs for the uncovered NBA games — fetch the scoreboard
        try:
            sb_resp = sess.get(_ESPN_SCOREBOARD, timeout=15)
            sb_resp.raise_for_status()
            events = sb_resp.json().get("events", [])
        except Exception as exc:
            log.warning(f"  [ESPN scoreboard] failed: {exc}")
            events = []

        for ev in events:
            comps = ev.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), {})
            home_full = home_c.get("team", {}).get("displayName", "")
            away_full = away_c.get("team", {}).get("displayName", "")
            nba_game  = _find_nba_game(home_full, away_full)
            if not nba_game or nba_game["nba_game_id"] in covered_nba_ids:
                continue
            gid = ev.get("id", "")
            try:
                sm_resp = sess.get(_ESPN_SUMMARY.format(gid), timeout=15)
                sm_resp.raise_for_status()
                officials = sm_resp.json().get("gameInfo", {}).get("officials", [])
            except Exception as exc:
                log.warning(f"  [ESPN summary] game {gid} failed: {exc}")
                officials = []

            if not officials:
                log.info(f"  [ESPN] {ev.get('name', gid)}: no officials yet")
                continue

            covered_nba_ids.add(nba_game["nba_game_id"])
            hf = nba_game["home_display"].title()
            af = nba_game["away_display"].title()
            for off in officials:
                ref_name  = off.get("fullName", "").strip()
                ref_order = off.get("order", 0)
                rows.append(_ref_stats_row(ref_name, nba_game, hf, af, ref_order, "espn_api"))

    df = pd.DataFrame(rows)

    if df.empty:
        log.warning("  [nba_official/ESPN] no officials retrieved for any game today")
        return df

    df.to_parquet(_REF_PARQUET, index=False)
    mapped = df["NBA_GAME_ID"].astype(bool).sum()
    log.info(
        f"  [save]  {_REF_PARQUET.name}  "
        f"({len(df)} rows, {df['NBA_GAME_ID'].nunique()} games, {mapped} rows with NBA_GAME_ID)"
    )
    return df


def _parse_nba_official_referee_assignments() -> pd.DataFrame:
    """Parse official.nba.com referee assignments table and return one row per game.

    Expected columns on official page:
      Game, Crew Chief, Referee, Umpire, Alternate
    """
    url = "https://official.nba.com/referee-assignments/"
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table is None:
            return pd.DataFrame()

        headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")] if table.find("thead") else []
        needed = ["Game", "Crew Chief", "Referee", "Umpire", "Alternate"]
        if not all(x in headers for x in needed):
            # Still attempt to parse by position if headers differ slightly.
            pass

        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 4:
                continue
            while len(cells) < 5:
                cells.append("")
            rows.append({
                "Game": cells[0],
                "Crew Chief": cells[1],
                "Referee": cells[2],
                "Umpire": cells[3],
                "Alternate": cells[4],
            })

        return pd.DataFrame(rows)
    except Exception as exc:
        log.warning(f"  [nba_official] referee page scrape failed: {exc}")
        return pd.DataFrame()


# ── Combined helper function (used by data_fetcher.py) ────────────────────────

def load_nbastuffer(
    page_type: str,
    season: str,
    split: str = "regular",
) -> pd.DataFrame:
    """
    Load a cached nbastuffer parquet.  Does NOT trigger a scrape.

    page_type : 'player' | 'team' | 'referee' | 'restdays'
    season    : e.g. '2025-26'
    split     : only used when page_type=='team';
                one of 'regular' | 'last5' | 'road' | 'home'
    """
    ss = _season_slug(season)
    if page_type == "team" and split != "regular":
        path = HIST_DIR / f"nbastuffer_teamstats_{ss}_{split}.parquet"
    else:
        path = HIST_DIR / f"nbastuffer_{page_type}_{ss}.parquet"

    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_databallr(page_type: str, season: str | None = None) -> pd.DataFrame:
    """Load a cached databallr parquet.  Does NOT trigger a scrape."""
    ss = _season_slug(season) if season else "current"
    path = HIST_DIR / f"databallr_{page_type}_{ss}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_today_referee_assignments() -> pd.DataFrame:
    """Load today's referee assignments (pre-scraped)."""
    if _REF_PARQUET.exists():
        return pd.read_parquet(_REF_PARQUET)
    return pd.DataFrame()


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape supplemental NBA data from external sources",
    )
    parser.add_argument(
        "--source",
        choices=["nbastuffer", "databallr", "refs", "all"],
        default="nbastuffer",
        help="Which source to scrape (default: nbastuffer)",
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Limit to a single season, e.g. '2025-26'",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore HTML cache and re-download all pages",
    )
    args = parser.parse_args()

    seasons = [args.season] if args.season else SEASONS
    refresh = args.refresh

    if args.source in ("nbastuffer", "all"):
        log.info("=" * 60)
        log.info("Scraping nbastuffer.com ...")
        log.info("=" * 60)
        scrape_nbastuffer_all(seasons, force_refresh=refresh)

    if args.source in ("databallr", "all"):
        log.info("=" * 60)
        log.info("Scraping databallr.com (requires playwright) ...")
        log.info("=" * 60)
        scrape_databallr_all(seasons, force_refresh=refresh)

    if args.source in ("refs", "all"):
        log.info("=" * 60)
        log.info("Scraping today's referee assignments ...")
        log.info("=" * 60)
        scrape_today_referee_assignments(force_refresh=refresh)

    log.info("Done.")


if __name__ == "__main__":
    main()
