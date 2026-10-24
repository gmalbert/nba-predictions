"""
scripts/export_best_bets.py — NBA (nba-predictions)
Reads the pre-computed predictions parquet (written by scripts/preload_cache.py)
and writes data_files/best_bets_today.json in the unified Sports Picks Grid schema.
"""
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

SPORT = "NBA"
MODEL_VERSION = "1.0.0"
_ET = ZoneInfo("America/New_York")
OUT_PATH = Path("data_files/best_bets_today.json")


def _write(bets: list, notes: str = "", *, status: str | None = None) -> None:
    today = datetime.now(tz=_ET).date()
    start_year = today.year if today.month >= 10 else today.year - 1
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": f"{start_year}-{str(start_year + 1)[-2:]}",
            "status": status or ("ok" if bets else "no_picks"),
            "tier_definition": "edge-v1",
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(OUT_PATH)
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _safe_float(val) -> float | None:
    try:
        number = float(val)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _tier(edge: float | None) -> str:
    for tier, threshold in (("Elite", .06), ("Strong", .03), ("Good", .01)):
        if edge is not None and edge >= threshold:
            return tier
    return "Standard"


def main() -> int:
    today = datetime.now(tz=_ET).date()

    # NBA season: mid-October through June (regular season + playoffs)
    month = today.month
    if not ((month == 10 and today.day >= 15) or month in [11, 12, 1, 2, 3, 4, 5, 6]):
        _write([], "NBA off-season", status="off_season")
        return 0

    try:
        import pandas as pd
    except ImportError:
        _write([], "pandas not available", status="pipeline_failed")
        return 1

    # Try today's parquet first, then yesterday's
    df = None
    read_errors = []
    for delta in (0, 1):
        check_date = today - timedelta(days=delta)
        candidates = [
            Path(f"data_files/historical/predictions_{check_date}.parquet"),
            Path(f"data_files/predictions_{check_date}.parquet"),
            Path("data_files/predictions_today.parquet"),
        ]
        for p in candidates:
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    if "game_date" not in df.columns and (delta != 0 or p.name == "predictions_today.parquet"):
                        df = None
                        continue
                    break
                except Exception as exc:
                    read_errors.append(f"{p}: {exc}")
                    continue
        if df is not None:
            break

    if df is None:
        # Publish source health instead of silently retaining months-old picks.
        _write([], "; ".join(read_errors) or f"No predictions artifact for {today}; upstream pipeline has not produced current data", status="pipeline_failed" if read_errors else "pipeline_pending")
        return 1

    # Filter to today
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        df = df[df["game_date"] == today]

    if df.empty:
        # An explicit empty current-day artifact is a successful no-picks run.
        _write([], f"No NBA games in prediction artifact for {today}")
        return 0

    if not {"home_team", "away_team"}.issubset(df.columns) or not any(field in df.columns for field in ("home_win_prob", "win_probability", "confidence")):
        _write([], "Prediction artifact is missing required team or probability columns", status="pipeline_failed")
        return 1

    bets = []
    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        game = f"{away} @ {home}"

        conf = _safe_float(row.get("home_win_prob", row.get("win_probability", row.get("confidence"))))
        edge = _safe_float(row.get("edge"))
        select_home = edge >= 0 if edge is not None else (conf or 0) >= .5
        # The engine exports home-side probability and a signed home edge.
        # An away recommendation needs the complementary probability and edge.
        if "home_win_prob" in row and conf is not None:
            conf = conf if select_home else 1 - conf
            edge = abs(edge) if edge is not None else None
        if conf is None or not 0 <= conf <= 1:
            continue

        # Skip low-edge or low-confidence rows
        if edge is not None and edge < 0.01:
            continue
        if conf is not None and conf < 0.52:
            continue

        edge = round(edge, 4) if edge is not None else None
        tier = _tier(edge)

        bet_type_raw = str(row.get("bet_type", "Moneyline"))
        bt_map = {"moneyline": "Moneyline", "spread": "Spread", "total": "Over/Under",
                  "Moneyline": "Moneyline", "Spread": "Spread", "Over/Under": "Over/Under"}
        bet_type = bt_map.get(bet_type_raw, bet_type_raw)

        pick = str(row.get("pick", row.get("predicted_winner", home if select_home else away)))

        bet: dict = {
            "game_date": str(today),
            "game_time": str(row["game_time"]) if pd.notna(row.get("game_time")) else None,
            "game": game,
            "home_team": home,
            "away_team": away,
            "bet_type": bet_type,
            "pick": pick,
            "confidence": conf,
            "edge": edge,
            "tier": tier,
            "odds": int(row["odds"]) if "odds" in row and _safe_float(row.get("odds")) is not None else None,
            "line": _safe_float(row.get("line")),
            "notes": str(row.get("notes", "")) or None,
        }
        bets.append(bet)

    _write(bets, "" if bets else f"No qualifying NBA picks for {today}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
