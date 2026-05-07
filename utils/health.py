"""
Startup safety health check.

run_startup_health_check() queries broker state, local DB, and config to
produce a structured GREEN / YELLOW / RED verdict before trading begins.

GREEN  — all invariants satisfied; normal operation proceeds.
YELLOW — non-fatal anomalies detected; new buys are blocked until resolved.
RED    — fatal condition; new buys are blocked and an alert is sent.

Usage (called from main._run_inner after reconcile_positions):

    from utils.health import run_startup_health_check, HealthStatus
    report = run_startup_health_check(client)
    if report.status == HealthStatus.RED:
        sys.exit(1)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from enum import StrEnum

import config

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class HealthReport:
    status: HealthStatus
    issues: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def log(self) -> None:
        colour = {"GREEN": "\033[32m", "YELLOW": "\033[33m", "RED": "\033[31m"}
        reset = "\033[0m"
        c = colour.get(self.status, "")
        logger.info(f"BROKER_HEALTH={c}{self.status}{reset}")
        for issue in self.issues:
            logger.warning(f"  • {issue}")
        if not self.issues:
            logger.info("  All startup checks passed")


def run_startup_health_check(client) -> HealthReport:
    """Run all startup safety checks and return a HealthReport.

    Checks (in order):
      1. Halt file absent
      2. Broker positions match local DB
      3. No whole-share position is uncovered by a stop order
      4. No stale open orders (orders older than one trading day)
      5. Current exposure within MAX_DEPLOYED_USD cap
      6. Daily loss within MAX_DAILY_LOSS_USD cap
      7. No unresolved order intents in the ledger (P5, set to pass if table absent)

    A single check failure does not abort the rest — all issues are collected.
    """

    issues: list[str] = []
    metrics: dict = {}

    # ── 1. Halt file ──────────────────────────────────────────────────────────
    if os.path.exists(config.HALT_FILE):
        issues.append(f"halt file present: {config.HALT_FILE}")

    # ── 2. Broker ↔ DB position reconciliation ────────────────────────────────
    broker_symbols: set[str] = set()
    db_symbols: set[str] = set()
    try:
        broker_positions = client.get_all_positions()
        broker_symbols = {p.symbol for p in broker_positions}

        from utils.db import get_db

        with get_db() as conn:
            db_symbols = {row["symbol"] for row in conn.execute("SELECT symbol FROM positions")}

        unexpected_broker = broker_symbols - db_symbols
        missing_from_broker = db_symbols - broker_symbols

        if unexpected_broker:
            for sym in sorted(unexpected_broker):
                issues.append(
                    f"unexpected broker position: {sym} — no local order intent or DB record"
                )
        if missing_from_broker:
            for sym in sorted(missing_from_broker):
                issues.append(f"DB position {sym} not at broker — stale metadata")

        metrics["broker_positions"] = len(broker_symbols)
        metrics["db_positions"] = len(db_symbols)
    except Exception as e:
        issues.append(f"position reconciliation failed: {e}")

    # ── 3. Stop coverage ──────────────────────────────────────────────────────
    try:
        open_orders = client.get_orders()
        from alpaca.trading.enums import OrderSide, OrderType

        stop_qty: dict[str, float] = {}
        for o in open_orders:
            if (
                o.order_type in {OrderType.TRAILING_STOP, OrderType.STOP, OrderType.STOP_LIMIT}
                and o.side == OrderSide.SELL
            ):
                stop_qty[o.symbol] = stop_qty.get(o.symbol, 0.0) + float(o.qty or 0)

        uncovered: list[str] = []
        for p in broker_positions:
            pos_qty = float(p.qty)
            covered = stop_qty.get(p.symbol, 0.0)
            uncovered_qty = pos_qty - covered
            if uncovered_qty > 0.000001:
                whole = int(math.floor(uncovered_qty))
                if whole >= 1:
                    uncovered.append(f"{p.symbol} ({whole} whole share(s) uncovered)")

        if uncovered:
            for u in uncovered:
                issues.append(f"no stop protection: {u}")
        metrics["uncovered_positions"] = len(uncovered)
    except Exception as e:
        issues.append(f"stop coverage check failed: {e}")

    # ── 4. Stale open orders ──────────────────────────────────────────────────
    try:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        stale_orders: list[str] = []
        for o in open_orders:
            if str(o.status) in ("new", "accepted", "pending_new"):
                created_at = getattr(o, "created_at", None)
                if created_at:
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=UTC)
                    age_hours = (now - created_at).total_seconds() / 3600
                    if age_hours > 8:  # older than one trading session
                        stale_orders.append(f"{o.symbol} order {o.id} ({age_hours:.1f}h old)")
        if stale_orders:
            for s in stale_orders:
                issues.append(f"stale open order: {s}")
        metrics["stale_orders"] = len(stale_orders)
    except Exception as e:
        issues.append(f"stale order check failed: {e}")

    # ── 5. Exposure cap ───────────────────────────────────────────────────────
    if config.MAX_DEPLOYED_USD > 0:
        try:
            from execution.trader import get_total_open_exposure
            from models import BrokerStateUnavailable

            exposure = get_total_open_exposure(client)
            metrics["open_exposure_usd"] = round(exposure, 2)
            if exposure > config.MAX_DEPLOYED_USD:
                issues.append(
                    f"exposure ${exposure:.2f} exceeds MAX_DEPLOYED_USD ${config.MAX_DEPLOYED_USD:.2f}"
                )
        except BrokerStateUnavailable as e:
            issues.append(f"cannot verify exposure: {e}")

    # ── 6. Daily loss cap ─────────────────────────────────────────────────────
    if config.MAX_DAILY_LOSS_USD > 0:
        try:
            from utils.portfolio_tracker import load_daily_baseline

            baseline = load_daily_baseline()
            if baseline:
                account = client.get_account()
                portfolio_value = float(account.portfolio_value)
                daily_loss = baseline - portfolio_value
                metrics["daily_loss_usd"] = round(daily_loss, 2)
                if daily_loss >= config.MAX_DAILY_LOSS_USD:
                    issues.append(
                        f"daily loss ${daily_loss:.2f} >= MAX_DAILY_LOSS_USD ${config.MAX_DAILY_LOSS_USD:.2f}"
                    )
        except Exception as e:
            issues.append(f"daily loss cap check failed: {e}")

    # ── 7. Unresolved order intents ───────────────────────────────────────────
    try:
        from utils.order_ledger import (
            auto_cancel_timeout_intents,
            get_unresolved_intents,
        )

        today = config.today_et().isoformat()
        # Cancel timeout intents where the broker has NO position (order never filled).
        # Do NOT call reconcile_filled_intents here — that consumes timeout intents
        # before the main.py reconciliation block can record them in all_trades.
        auto_cancel_timeout_intents(broker_symbols, today)
        unresolved = get_unresolved_intents(trade_date=today)
        if unresolved:
            for intent in unresolved:
                issues.append(
                    f"unresolved order intent: {intent['symbol']} {intent['side']} "
                    f"id={intent['client_order_id']} status={intent['status']}"
                )
        metrics["unresolved_intents"] = len(unresolved)
    except ImportError:
        pass  # order_ledger not yet migrated — skip check
    except Exception as e:
        issues.append(f"order intent check failed: {e}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    fatal_keywords = (
        "unexpected broker position",
        "no stop protection",
        "cannot verify exposure",
        "daily loss",
        "exposure",
        "halt file",
    )
    has_fatal = any(any(kw in issue for kw in fatal_keywords) for issue in issues)

    if not issues:
        status = HealthStatus.GREEN
    elif has_fatal:
        status = HealthStatus.RED
    else:
        status = HealthStatus.YELLOW

    return HealthReport(status=status, issues=issues, metrics=metrics)
