"""
scripts/export_best_bets.py — NBA (nba-predictions)
Reads the pre-computed predictions parquet (written by scripts/preload_cache.py)
and writes data_files/best_bets_today.json in the unified Sports Picks Grid schema.
"""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SPORT = "NBA"
MODEL_VERSION = "1.0.0"
SEASON = str(date.today().year)
OUT_PATH = Path("data_files/best_bets_today.json")

TIER_MAP = {
    "High": "Elite",
    "Medium": "Strong",
    "Low": "Good",
    "Elite": "Elite",
    "Strong": "Strong",
    "Good": "Good",
}


def _write(bets: list, notes: str = "") -> None:
    payload: dict = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": SEASON,
        },
        "bets": bets,
    }
    if notes:
        payload["meta"]["notes"] = notes
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[{SPORT}] Wrote {len(bets)} bets -> {OUT_PATH}")


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> None:
    today = date.today()

    # NBA season: mid-October through late April
    month = today.month
    if not ((month == 10 and today.day >= 15) or month in [11, 12, 1, 2, 3] or (month == 4 and today.day <= 30)):
        _write([], "NBA off-season")
        return

    try:
        import pandas as pd
    except ImportError:
        _write([], "pandas not available")
        return

    # Try today's parquet first, then yesterday's
    for delta in (0, 1):
        check_date = today - timedelta(days=delta)
        candidates = [
            Path(f"data_files/historical/predictions_{check_date}.parquet"),
            Path(f"data_files/predictions_{check_date}.parquet"),
            Path("data_files/predictions_today.parquet"),
        ]
        df = None
        for p in candidates:
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    break
                except Exception:
                    continue
        if df is not None:
            break

    if df is None or df.empty:
        _write([], f"No predictions parquet found for {today}")
        return

    # Filter to today
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        df = df[df["game_date"] == today]

    if df.empty:
        _write([], f"No NBA games for {today}")
        return

    bets = []
    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        game = f"{away} @ {home}"

        conf = _safe_float(row.get("home_win_prob", row.get("win_probability", row.get("confidence"))))
        edge = _safe_float(row.get("edge"))

        # Skip low-edge or low-confidence rows
        if edge is not None and edge < 0.01:
            continue
        if conf is not None and conf < 0.52:
            continue

        tier_raw = str(row.get("confidence", row.get("tier", "Low")))
        tier = TIER_MAP.get(tier_raw, "Good")

        bet_type_raw = str(row.get("bet_type", "Moneyline"))
        bt_map = {"moneyline": "Moneyline", "spread": "Spread", "total": "Over/Under",
                  "Moneyline": "Moneyline", "Spread": "Spread", "Over/Under": "Over/Under"}
        bet_type = bt_map.get(bet_type_raw, bet_type_raw)

        pick = str(row.get("pick", row.get("predicted_winner", home if (conf or 0) >= 0.5 else away)))

        bet: dict = {
            "game_date": str(today),
            "game_time": str(row.get("game_time", "")) or None,
            "game": game,
            "home_team": home,
            "away_team": away,
            "bet_type": bet_type,
            "pick": pick,
            "confidence": conf,
            "edge": edge,
            "tier": tier,
            "odds": int(row["odds"]) if "odds" in row and _safe_float(row.get("odds")) is not None else None,
            "line": _safe_float(row.get("predicted_spread", row.get("line"))),
            "notes": str(row.get("notes", "")) or None,
        }
        bets.append(bet)

    _write(bets, "" if bets else f"No qualifying NBA picks for {today}")


if __name__ == "__main__":
    main()
