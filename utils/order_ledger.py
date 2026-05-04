"""
Durable order-intent / order-state ledger.

Every buy attempt is recorded as an order_intent before the broker call and
updated as execution progresses. This makes every run restart-safe:

  - On the same day, if an intent is already SUBMITTED or FILLED for a symbol,
    the buy loop skips re-ordering.
  - An UNRESOLVED intent (submitted but no fill/reject confirmed) surfaces in
    the startup health check so the operator can verify manually.

Intent lifecycle:
  pending   → created before broker submission
  submitted → broker call succeeded (order_id assigned)
  filled    → wait_for_fill confirmed fill
  partial   → partial fill only
  rejected  → broker rejected or Alpaca 409 (duplicate client_order_id)
  timeout   → fill poll timed out — position may or may not exist at broker
  cancelled → operator or system cancelled the intent

order_events is an append-only audit trail of all state transitions and
intermediate signals (fill confirmations, stop placement, etc.).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from utils.db import get_db

logger = logging.getLogger(__name__)

# Statuses that mean "we already committed capital or are in the process of it".
# Used to prevent duplicate submissions for the same symbol/date.
_ACTIVE_STATUSES = frozenset({"pending", "submitted", "filled", "partial", "timeout"})


def create_intent(
    symbol: str,
    side: str,
    trade_date: str,
    intended_notional: float,
    client_order_id: str,
) -> int | None:
    """Insert a new order intent. Returns the new row id, or None on failure."""
    now = datetime.now(UTC).isoformat()
    try:
        with get_db() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO order_intents "
                "(symbol, side, trade_date, intended_notional, client_order_id, "
                "status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (symbol, side, trade_date, intended_notional, client_order_id, "pending", now, now),
            )
            return cur.lastrowid
    except Exception as e:
        logger.warning(f"create_intent({symbol}): {e}")
        return None


def update_intent(
    client_order_id: str,
    status: str,
    broker_order_id: str | None = None,
) -> None:
    """Update the status (and optionally broker_order_id) of an existing intent."""
    now = datetime.now(UTC).isoformat()
    try:
        with get_db() as conn:
            if broker_order_id is not None:
                conn.execute(
                    "UPDATE order_intents SET status=?, broker_order_id=?, updated_at=? "
                    "WHERE client_order_id=?",
                    (status, broker_order_id, now, client_order_id),
                )
            else:
                conn.execute(
                    "UPDATE order_intents SET status=?, updated_at=? WHERE client_order_id=?",
                    (status, now, client_order_id),
                )
    except Exception as e:
        logger.warning(f"update_intent({client_order_id} → {status}): {e}")


def log_order_event(
    client_order_id: str,
    event_type: str,
    payload: dict,
    broker_order_id: str | None = None,
) -> None:
    """Append an event to the order_events audit trail."""
    now = datetime.now(UTC).isoformat()
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO order_events "
                "(client_order_id, broker_order_id, event_type, payload, ts) "
                "VALUES (?,?,?,?,?)",
                (client_order_id, broker_order_id, event_type, json.dumps(payload), now),
            )
    except Exception as e:
        logger.warning(f"log_order_event({client_order_id}, {event_type}): {e}")


def has_active_intent(symbol: str, side: str, trade_date: str) -> bool:
    """Return True if there is already an active (non-terminal) intent for this symbol/side/date.

    Used as a pre-buy guard alongside has_pending_buy(). The ledger survives
    process restarts; the broker query does not.
    """
    try:
        with get_db() as conn:
            placeholders = ",".join("?" * len(_ACTIVE_STATUSES))
            row = conn.execute(
                f"SELECT 1 FROM order_intents WHERE symbol=? AND side=? AND trade_date=? "
                f"AND status IN ({placeholders}) LIMIT 1",
                (symbol, side, trade_date, *_ACTIVE_STATUSES),
            ).fetchone()
        return row is not None
    except Exception as e:
        logger.warning(f"has_active_intent({symbol}): {e}")
        return False


def get_unresolved_intents(trade_date: str | None = None) -> list[dict]:
    """Return intents in ambiguous states (submitted/timeout) that need reconciliation.

    'submitted' without a subsequent fill/reject means the process crashed between
    submission and fill confirmation.
    'timeout' means wait_for_fill gave up — the position may or may not exist.
    """
    unresolved_statuses = ("submitted", "timeout")
    try:
        with get_db() as conn:
            if trade_date:
                rows = conn.execute(
                    "SELECT * FROM order_intents WHERE trade_date=? AND status IN (?,?)",
                    (trade_date, *unresolved_statuses),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM order_intents WHERE status IN (?,?)",
                    unresolved_statuses,
                ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"get_unresolved_intents: {e}")
        return []
