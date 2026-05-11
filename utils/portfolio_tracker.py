import json
import logging
import os
from datetime import UTC, datetime
from datetime import date as _date

from config import LOG_DIR

_BASELINE_PATH = os.path.join(LOG_DIR, "daily_baseline.json")
_EXPERIMENT_BASELINE_PATH = os.path.join(LOG_DIR, "experiment_baseline.json")

logger = logging.getLogger(__name__)


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _weekly_log_dir(date_str: str) -> str:
    """Return (and create) the ISO-week subdirectory for a given date string.

    Files are organised as logs/{iso_year}/Week {iso_week}/.
    Week numbers follow ISO 8601 — Monday is the first day, and Monday 4 May 2026
    is the start of Week 19.
    """
    try:
        d = _date.fromisoformat(date_str[:10])
    except ValueError:
        d = datetime.now(UTC).date()
    iso_year, iso_week, _ = d.isocalendar()
    week_dir = os.path.join(LOG_DIR, str(iso_year), f"Week {iso_week}")
    os.makedirs(week_dir, exist_ok=True)
    return week_dir


def _daily_log_path(date: str) -> str:
    return os.path.join(_weekly_log_dir(date), f"{date}.json")


def save_daily_run(
    date: str,
    account_before: dict,
    account_after: dict,
    ai_decisions: dict,
    trades_executed: list[dict],
    stop_losses_triggered: list[dict],
    run_id: str | None = None,
):
    # Unified decisions list — single audit trail across buy and sell decisions.
    # buy_candidates and position_decisions kept for backward compatibility.
    _decisions: list[dict] = [
        {
            "symbol": c.get("symbol", ""),
            "decision_type": "buy",
            "confidence": c.get("confidence"),
            "key_signal": c.get("key_signal"),
            "reasoning": c.get("reasoning", ""),
        }
        for c in ai_decisions.get("buy_candidates", [])
    ] + [
        {
            "symbol": d.get("symbol", ""),
            "decision_type": d.get("action", "SELL").lower(),
            "confidence": d.get("confidence"),
            "key_signal": None,
            "reasoning": d.get("reasoning", ""),
        }
        for d in ai_decisions.get("position_decisions", [])
    ]
    record = {
        "date": date,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "account_before": account_before,
        "account_after": account_after,
        "market_summary": ai_decisions.get("market_summary", ""),
        "decisions": _decisions,
        "position_decisions": ai_decisions.get("position_decisions", []),
        "buy_candidates": ai_decisions.get("buy_candidates", []),
        "trades_executed": trades_executed,
        "stop_losses_triggered": stop_losses_triggered,
        "daily_pnl": account_after["portfolio_value"] - account_before["portfolio_value"],
    }

    path = _daily_log_path(date)

    # Guard against spurious re-runs overwriting a record that already has trades.
    # If the file exists and the prior record has trades, merge rather than overwrite:
    # - Preserve the open-of-day account_before so P&L is always measured from market open.
    # - Union trades by order_id so no executed trade is ever lost.
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
            prior_trades = existing.get("trades_executed", [])
            if prior_trades:
                existing_ids = {t.get("order_id") for t in prior_trades if t.get("order_id")}
                merged = list(prior_trades)
                for t in trades_executed:
                    if t.get("order_id") not in existing_ids:
                        merged.append(t)
                record["trades_executed"] = merged
                record["account_before"] = existing["account_before"]
                record["daily_pnl"] = (
                    account_after["portfolio_value"] - existing["account_before"]["portfolio_value"]
                )
                added = len(merged) - len(prior_trades)
                if added:
                    logger.info(f"Merged {added} new trade(s) into existing {date} record")
                else:
                    logger.warning(
                        f"Spurious re-run detected for {date}: existing record has "
                        f"{len(prior_trades)} trade(s) — preserving; current run had {len(trades_executed)}"
                    )
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt existing file — overwrite silently

    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    # Mirror to SQLite runs table
    try:
        from utils.db import get_db

        mode = "open"
        if "-midday" in date:
            mode = "midday"
        elif "-close" in date:
            mode = "close"
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(date, mode, run_id, timestamp, account_before, account_after, "
                "market_summary, position_decisions, buy_candidates, trades_executed, "
                "stop_losses, daily_pnl) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record["date"],
                    mode,
                    record.get("run_id"),
                    record["timestamp"],
                    json.dumps(record["account_before"]),
                    json.dumps(record["account_after"]),
                    record.get("market_summary", ""),
                    json.dumps(record.get("position_decisions", [])),
                    json.dumps(record.get("buy_candidates", [])),
                    json.dumps(record.get("trades_executed", [])),
                    json.dumps(record.get("stop_losses_triggered", [])),
                    record.get("daily_pnl", 0.0),
                ),
            )
    except Exception as e:
        logger.warning(f"SQLite run write failed: {e}")

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


_NON_RUN_FILES = {
    "positions_meta.json",
    "signal_stats.json",
    "daily_baseline.json",
    "backtest_results.json",
    "runtime_config.json",
    "fmp_fundamentals_cache.json",
    "fmp_analyst_cache.json",
}


def save_daily_baseline(portfolio_value: float) -> None:
    """Persist the open-of-day portfolio value so daily loss checks compare against a real baseline."""
    _ensure_log_dir()
    today = datetime.now(UTC).date().isoformat()
    with open(_BASELINE_PATH, "w") as f:
        json.dump({"date": today, "portfolio_value": portfolio_value}, f)
    try:
        from utils.db import get_db

        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO daily_baselines VALUES (?,?)",
                (today, portfolio_value),
            )
    except Exception as e:
        logger.warning(f"SQLite baseline write failed: {e}")


def load_daily_baseline() -> float | None:
    """Return today's persisted open-of-day equity, or None if not recorded yet."""
    try:
        with open(_BASELINE_PATH) as f:
            data = json.load(f)
        today = datetime.now(UTC).date().isoformat()
        if data.get("date") == today:
            return float(data["portfolio_value"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        pass
    return None


def save_experiment_baseline(portfolio_value: float) -> None:
    """Persist the experiment-start equity — written once and never overwritten.

    Used to enforce MAX_EXPERIMENT_DRAWDOWN_USD across the lifetime of the live
    experiment. Unlike daily_baseline, this survives across days.
    """
    if os.path.exists(_EXPERIMENT_BASELINE_PATH):
        return  # never overwrite — baseline is set on first call only
    _ensure_log_dir()
    with open(_EXPERIMENT_BASELINE_PATH, "w") as f:
        json.dump(
            {
                "set_at": datetime.now(UTC).isoformat(),
                "portfolio_value": portfolio_value,
            },
            f,
        )
    logger.info(f"Experiment baseline set: ${portfolio_value:.2f}")


def load_experiment_baseline() -> float | None:
    """Return the experiment-start equity, or None if not yet set."""
    try:
        with open(_EXPERIMENT_BASELINE_PATH) as f:
            data = json.load(f)
        return float(data["portfolio_value"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None


def load_history() -> list[dict]:
    """Load all daily run records from weekly subdirectories, sorted chronologically."""
    _ensure_log_dir()
    # Collect (filename, full_path) across all subdirs; sort by filename for chronological order
    found: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(LOG_DIR):
        dirs.sort()
        for fname in files:
            if fname.endswith(".json") and fname not in _NON_RUN_FILES:
                found.append((fname, os.path.join(root, fname)))
    records = []
    for _fname, fpath in sorted(found):
        try:
            with open(fpath) as f:
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
    all_stops = [s for r in today_records for s in r.get("stop_losses_triggered", [])]
    buy_candidates = [b for r in today_records for b in r.get("buy_candidates", [])]
    position_decisions = [d for r in today_records for d in r.get("position_decisions", [])]
    decisions = [d for r in today_records for d in r.get("decisions", [])]

    # Use the open run's market summary — it has the full AI analysis
    market_summary = next(
        (r["market_summary"] for r in today_records if r["date"] == today),
        today_records[-1].get("market_summary", ""),
    )

    account_before = today_records[0]["account_before"]
    account_after = today_records[-1]["account_after"]

    return {
        "date": today,
        "account_before": account_before,
        "account_after": account_after,
        "market_summary": market_summary,
        "decisions": decisions,
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
            trades.append(
                {
                    "symbol": t.get("symbol"),
                    "action": t.get("action"),
                    "detail": t.get("detail", ""),
                }
            )
        for s in r.get("stop_losses_triggered", []):
            trades.append(
                {
                    "symbol": s.get("symbol"),
                    "action": "STOP_LOSS",
                    "pl_pct": s.get("pl_pct"),
                }
            )
        result.append(
            {
                "date": r["date"],
                "daily_pnl_usd": round(r.get("daily_pnl", 0), 2),
                "market": r.get("market_summary", ""),
                "trades": trades,
            }
        )
    return result
