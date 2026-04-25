"""
Structured audit trail written to logs/audit.jsonl (newline-delimited JSON).

Each line is a self-contained JSON event with a UTC timestamp. The file is
append-only during normal operation — never overwritten or rotated — to
satisfy MiFID II Article 25 record-keeping requirements (orders and decisions
must be retained for 5 years and be reproducible on request).

To inspect: `cat logs/audit.jsonl | python3 -m json.tool | less`
To filter orders: `grep ORDER_PLACED logs/audit.jsonl`
"""

from __future__ import annotations
import json
import os
import logging
from datetime import datetime, timezone
from config import LOG_DIR

logger = logging.getLogger(__name__)

_AUDIT_PATH = os.path.join(LOG_DIR, "audit.jsonl")


def _write(event_type: str, payload: dict):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **payload,
    }
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_AUDIT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Audit log write failed: {e}")


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
    _write("MACRO_SKIP", {"event": event})


# ── Kill switch / halt ────────────────────────────────────────────────────────

def log_kill_switch(positions_closed: int):
    _write("KILL_SWITCH", {"positions_closed": positions_closed})


def log_halt_cleared():
    _write("HALT_CLEARED", {})
