from __future__ import annotations
import json
import os
import logging
from datetime import datetime, timezone
from config import LOG_DIR

logger = logging.getLogger(__name__)


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _daily_log_path(date: str) -> str:
    _ensure_log_dir()
    return os.path.join(LOG_DIR, f"{date}.json")


def save_daily_run(
    date: str,
    account_before: dict,
    account_after: dict,
    ai_decisions: dict,
    trades_executed: list[dict],
    stop_losses_triggered: list[dict],
):
    record = {
        "date": date,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account_before": account_before,
        "account_after": account_after,
        "market_summary": ai_decisions.get("market_summary", ""),
        "position_decisions": ai_decisions.get("position_decisions", []),
        "buy_candidates": ai_decisions.get("buy_candidates", []),
        "trades_executed": trades_executed,
        "stop_losses_triggered": stop_losses_triggered,
        "daily_pnl": account_after["portfolio_value"] - account_before["portfolio_value"],
    }

    path = _daily_log_path(date)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Daily run saved to {path}")
    return record


def print_summary(record: dict):
    pnl = record["daily_pnl"]
    pnl_sign = "+" if pnl >= 0 else ""
    print("\n" + "=" * 50)
    print(f"  TRADING BOT DAILY SUMMARY — {record['date']}")
    print("=" * 50)
    print(f"  Market: {record['market_summary']}")
    print(f"  Portfolio value: ${record['account_after']['portfolio_value']:.2f}")
    print(f"  Cash:            ${record['account_after']['cash']:.2f}")
    print(f"  Daily P&L:       {pnl_sign}${pnl:.2f}")
    print()

    if record["stop_losses_triggered"]:
        print("  STOP LOSSES TRIGGERED:")
        for sl in record["stop_losses_triggered"]:
            print(f"    {sl['symbol']}  {sl['pl_pct']:.1f}%  (stop loss)")

    if record["trades_executed"]:
        print("  TRADES EXECUTED:")
        for t in record["trades_executed"]:
            print(f"    {t.get('action', '?')} {t['symbol']}  {t.get('detail', '')}")
    else:
        print("  No trades executed today.")

    print("=" * 50 + "\n")


_NON_RUN_FILES = {"positions_meta.json", "signal_stats.json"}


def load_history() -> list[dict]:
    """Load all daily run records, ignoring metadata and stats files."""
    _ensure_log_dir()
    records = []
    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith(".json") or fname in _NON_RUN_FILES:
            continue
        with open(os.path.join(LOG_DIR, fname)) as f:
            try:
                data = json.load(f)
                if "date" in data and "account_after" in data:
                    records.append(data)
            except json.JSONDecodeError:
                pass
    return records


def get_day_summary(today: str) -> dict | None:
    """
    Merge all of today's run records (open, midday, close) into a single
    end-of-day record for the daily summary email.
    account_before = start of open run; account_after = end of close run.
    All trades across all three runs are combined.
    """
    all_records = load_history()
    today_records = sorted(
        [r for r in all_records if r.get("date", "").startswith(today)],
        key=lambda r: r.get("timestamp", r["date"]),
    )
    if not today_records:
        return None

    all_trades = [t for r in today_records for t in r.get("trades_executed", [])]
    all_stops  = [s for r in today_records for s in r.get("stop_losses_triggered", [])]
    buy_candidates    = [b for r in today_records for b in r.get("buy_candidates", [])]
    position_decisions = [d for r in today_records for d in r.get("position_decisions", [])]

    # Use the open run's market summary — it has the full AI analysis
    market_summary = next(
        (r["market_summary"] for r in today_records if r["date"] == today),
        today_records[-1].get("market_summary", ""),
    )

    account_before = today_records[0]["account_before"]
    account_after  = today_records[-1]["account_after"]

    return {
        "date": today,
        "account_before": account_before,
        "account_after":  account_after,
        "market_summary": market_summary,
        "buy_candidates": buy_candidates,
        "position_decisions": position_decisions,
        "trades_executed": all_trades,
        "stop_losses_triggered": all_stops,
        "daily_pnl": account_after["portfolio_value"] - account_before["portfolio_value"],
        "mode": "paper",
    }


def get_track_record(n_days: int = 10) -> list[dict]:
    """Return a compact summary of the last n_days for the AI to learn from."""
    records = load_history()
    recent = records[-n_days:]
    result = []
    for r in recent:
        trades = []
        for t in r.get("trades_executed", []):
            trades.append({
                "symbol": t.get("symbol"),
                "action": t.get("action"),
                "detail": t.get("detail", ""),
            })
        for s in r.get("stop_losses_triggered", []):
            trades.append({
                "symbol": s.get("symbol"),
                "action": "STOP_LOSS",
                "pl_pct": s.get("pl_pct"),
            })
        result.append({
            "date": r["date"],
            "daily_pnl_usd": round(r.get("daily_pnl", 0), 2),
            "market": r.get("market_summary", ""),
            "trades": trades,
        })
    return result
