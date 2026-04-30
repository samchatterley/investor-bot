"""
Structured audit trail — written to both logs/audit.jsonl and the SQLite
audit_events table (via utils.db).

Each entry is self-contained with a UTC timestamp. The JSONL file is
append-only and never rotated to satisfy MiFID II Article 25 record-keeping
requirements. SQLite enables fast querying and cross-run correlation by run_id.

To inspect: `cat logs/audit.jsonl | python3 -m json.tool | less`
To filter orders: `grep ORDER_PLACED logs/audit.jsonl`
"""

import json
import os
import logging
from datetime import datetime, timezone
from config import LOG_DIR

logger = logging.getLogger(__name__)

_AUDIT_PATH = os.path.join(LOG_DIR, "audit.jsonl")
_run_id: str | None = None


def set_run_id(run_id: str) -> None:
    """Called once per run in main.py so all events are correlated."""
    global _run_id
    _run_id = run_id


def _write(event_type: str, payload: dict):
    ts = datetime.now(timezone.utc).isoformat()
    entry = {"ts": ts, "event": event_type, **payload}
    if _run_id:
        entry["run_id"] = _run_id

    # JSONL — append-only for MiFID II compliance
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_AUDIT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Audit JSONL write failed: {e}")

    # SQLite — queryable, correlated by run_id
    try:
        from utils.db import get_db
        with get_db() as conn:
            conn.execute(
                "INSERT INTO audit_events (run_id, ts, event, payload) VALUES (?,?,?,?)",
                (_run_id, ts, event_type, json.dumps(payload)),
            )
    except Exception as e:
        logger.error(f"Audit SQLite write failed: {e}")


# ── Run lifecycle ─────────────────────────────────────────────────────────────

def log_run_start(mode: str, portfolio_value: float, cash: float, is_paper: bool):
    _write("RUN_START", {"mode": mode, "portfolio_value": portfolio_value,
                         "cash": cash, "paper": is_paper})


def log_run_end(mode: str, pnl: float, trades_executed: int, portfolio_value: float):
    _write("RUN_END", {"mode": mode, "pnl": round(pnl, 4),
                       "trades_executed": trades_executed, "portfolio_value": portfolio_value})


# ── Orders ────────────────────────────────────────────────────────────────────

def log_order_placed(symbol: str, side: str, notional: float, order_id: str):
    _write("ORDER_PLACED", {"symbol": symbol, "side": side,
                             "notional": round(notional, 4), "order_id": order_id})


def log_order_filled(symbol: str, order_id: str, fill_qty: float):
    _write("ORDER_FILLED", {"symbol": symbol, "order_id": order_id,
                             "fill_qty": round(fill_qty, 6)})


def log_position_closed(symbol: str, reason: str, pl_pct: float):
    _write("POSITION_CLOSED", {"symbol": symbol, "reason": reason,
                                "pl_pct": round(pl_pct, 4)})


# ── AI decisions ──────────────────────────────────────────────────────────────

def log_ai_decision(market_summary: str, buy_count: int, sell_count: int):
    _write("AI_DECISION", {"market_summary": market_summary,
                            "buy_count": buy_count, "sell_count": sell_count})


def log_validation_failure(errors: list[str]):
    _write("VALIDATION_FAILURE", {"errors": errors})


# ── Risk events ───────────────────────────────────────────────────────────────

def log_circuit_breaker(drawdown_pct: float):
    _write("CIRCUIT_BREAKER", {"drawdown_pct": round(drawdown_pct, 2)})


def log_daily_loss_limit(loss_pct: float):
    _write("DAILY_LOSS_LIMIT", {"loss_pct": round(loss_pct, 2)})


def log_earnings_exit(symbol: str, earnings_date: str):
    _write("EARNINGS_EXIT", {"symbol": symbol, "earnings_date": earnings_date})


def log_macro_skip(event: str):
    _write("MACRO_SKIP", {"macro_event": event})


# ── Kill switch / halt ────────────────────────────────────────────────────────

def log_kill_switch(positions_closed: int):
    _write("KILL_SWITCH", {"positions_closed": positions_closed})


def log_halt_cleared():
    _write("HALT_CLEARED", {})
