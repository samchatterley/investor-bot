"""
Daily trading run.

Usage:
    python main.py                     # Full cycle at market open
    python main.py --mode midday       # Manage positions, partial exits, no new buys
    python main.py --mode close        # Final review before close
    python main.py --dry-run           # Analyse only, no orders placed
    python main.py --kill-switch       # Emergency: cancel all orders, close all positions, halt bot
    python main.py --clear-halt        # Resume after a kill-switch halt
    python main.py --backtest          # Run historical backtest and exit
"""

import argparse
import contextlib
import json
import logging
import math
import os
import sys
import time
import uuid
from datetime import UTC, datetime

import config
from analysis import ai_analyst, performance
from analysis.weekly_review import get_latest_review
from data import market_data, news_fetcher, options_scanner, sector_data
from data import sentiment as sentiment_module
from execution import stock_scanner, trader
from execution.quote_gate import check_quote_gate
from execution.universe import build_scan_universe
from models import BrokerStateUnavailable, OrderLedgerUnavailable, OrderResult, OrderStatus
from notifications import alerts, emailer
from risk import earnings_calendar, macro_calendar, position_sizer, risk_manager
from utils import audit_log, decision_log, portfolio_tracker
from utils.db import init_db
from utils.health import HealthStatus, run_startup_health_check
from utils.portfolio_tracker import (
    get_day_summary,
    load_experiment_baseline,
    save_experiment_baseline,
)
from utils.validators import check_pre_trade, sanitize_headlines, validate_ai_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_LOCK_MAX_AGE_SECONDS = 1800  # auto-clear locks older than 30 min (handles crash recovery)


def _lock_file() -> str:
    # Computed at call time so a run spanning midnight uses the correct date.
    return os.path.join(config.LOG_DIR, f".lock_{config.today_et().isoformat()}")


# ── Lock file management ──────────────────────────────────────────────────────


def _acquire_lock() -> int | None:
    """Atomically create the lock file. Returns an open fd on success, None if already locked."""
    lock_file = _lock_file()
    os.makedirs(config.LOG_DIR, exist_ok=True)
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        age = time.time() - os.path.getmtime(lock_file)
        if age > _LOCK_MAX_AGE_SECONDS:
            logger.warning(f"Stale lock file found ({age / 3600:.1f}h old) — auto-clearing")
            with contextlib.suppress(OSError):
                os.remove(lock_file)
            try:
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                logger.warning(
                    "Could not acquire lock after stale removal — another run may be in progress."
                )
                return None
        else:
            logger.warning(
                "Lock file exists — another run may be in progress. Remove .lock file to override."
            )
            return None
    payload = json.dumps({"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()})
    os.write(fd, payload.encode())
    return fd


def _release_lock(fd: int | None = None):
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)
    with contextlib.suppress(FileNotFoundError):
        os.remove(_lock_file())


# ── Kill switch ───────────────────────────────────────────────────────────────


def _run_kill_switch():
    """
    Emergency halt: cancel all open orders, liquidate all positions, write HALT file.
    MiFID II Article 17(1) requires firms to be able to halt trading immediately.
    """
    logger.critical("KILL SWITCH ACTIVATED — cancelling all orders and liquidating positions")
    client = trader.get_client()

    # 1. Cancel all open orders
    try:
        client.cancel_orders()
        logger.info("All open orders cancelled")
    except Exception as e:
        logger.error(f"Failed to cancel orders: {e}")

    # 2. Attempt to close all open positions; track per-symbol order result
    positions = trader.get_open_positions(client)
    close_results: dict[str, OrderResult] = {}
    for pos in positions:
        result = trader.close_position(client, pos["symbol"])
        close_results[pos["symbol"]] = result
        pl = pos["unrealized_pl"]
        logger.critical(
            f"  {pos['symbol']}: order={result.status.name}  P&L: {'+' if pl >= 0 else ''}{pl:.2f}"
        )
        if result.is_success:
            audit_log.log_position_closed(pos["symbol"], "kill_switch", pos["unrealized_plpc"])

    # 3. Verify — re-query broker to confirm what actually closed
    verification_error: str | None = None
    remaining: set[str] = set()
    try:
        remaining = {p["symbol"] for p in trader.get_open_positions(client)}
    except Exception as e:
        verification_error = str(e)
        logger.error(
            f"Post-liquidation broker re-query failed: {e} — "
            "liquidation status is UNKNOWN; check broker immediately"
        )

    if verification_error:
        liquidation_complete: bool | str = "UNKNOWN"
        confirmed_closed: list[str] = []
        still_open: list[str] = []
    else:
        confirmed_closed = [s for s in close_results if s not in remaining]
        still_open = [s for s in close_results if s in remaining]
        liquidation_complete = len(still_open) == 0
        if still_open:
            logger.critical(
                f"LIQUIDATION INCOMPLETE — {len(still_open)} position(s) still open "
                f"at broker: {still_open}"
            )
        else:
            logger.critical("Broker confirms: all positions liquidated")

    # 4. Write HALT file as machine-readable JSON
    os.makedirs(config.LOG_DIR, exist_ok=True)
    symbols_detail = {
        symbol: {
            "order_status": result.status.name,
            "broker_status": (
                "unknown" if verification_error else ("open" if symbol in remaining else "closed")
            ),
        }
        for symbol, result in close_results.items()
    }
    halt_data: dict = {
        "halted": True,
        "timestamp": datetime.now(UTC).isoformat(),
        "liquidation_complete": liquidation_complete,
        "symbols": symbols_detail,
    }
    if verification_error:
        halt_data["verification_error"] = verification_error
    else:
        halt_data["positions_remaining"] = still_open

    with open(config.HALT_FILE, "w") as f:
        json.dump(halt_data, f, indent=2)
        f.write("\n")

    closed = len(confirmed_closed)
    total = len(close_results)
    audit_log.log_kill_switch(closed)
    if verification_error:
        alert_msg = (
            f"Emergency halt activated. {total} close attempt(s) made. "
            f"WARNING: Could not verify broker state — {verification_error}. "
            "Check broker immediately."
        )
    else:
        alert_msg = f"Emergency halt activated. {closed}/{total} positions broker-confirmed closed."
        if still_open:
            alert_msg += f" WARNING: {still_open} may still be open — verify at broker."
    alerts.alert_error("KILL SWITCH", alert_msg)
    logger.critical(f"Kill switch complete. {closed}/{total} positions confirmed closed.")
    logger.critical("To resume: python main.py --clear-halt")


def _run_clear_halt():
    if os.path.exists(config.HALT_FILE):
        os.remove(config.HALT_FILE)
        audit_log.log_halt_cleared()
        logger.info("Halt cleared. Trading will resume on next scheduled run.")
    else:
        logger.info("No halt file found — trading is already active.")


# ── Safety-check mode ─────────────────────────────────────────────────────────


def _run_safety_check():
    """Verify broker state, stop coverage, caps, and halt status without trading.

    Prints a structured GREEN / YELLOW / RED report. No AI analysis, no orders.
    Safe to run at any time; recommended before first live session and after
    any incident or restart.

    Exit codes: 0 = GREEN, 1 = YELLOW, 2 = RED.
    """
    print("\n=== InvestorBot Safety Check ===")
    print(f"Mode: {'PAPER' if config.IS_PAPER else '*** LIVE ***'}")
    print(f"Time: {datetime.now(UTC).isoformat()}\n")

    client = trader.get_client()

    # Run startup health check
    report = run_startup_health_check(client)
    report.log()

    # Additional: confirm stops are current
    try:
        stops_ok = trader.ensure_stops_attached(client)
        if not stops_ok:
            report.issues.append(
                "ensure_stops_attached: one or more positions could not be protected"
            )
        else:
            logger.info("  Stop coverage: OK")
    except Exception as e:
        report.issues.append(f"ensure_stops_attached failed: {e}")

    # Print a human-readable summary
    print(f"\n--- Result: BROKER_HEALTH={report.status} ---")
    if report.issues:
        print(f"Issues ({len(report.issues)}):")
        for issue in report.issues:
            print(f"  • {issue}")
    else:
        print("All checks passed — safe to trade.")

    if report.metrics:
        print("\nMetrics:")
        for k, v in sorted(report.metrics.items()):
            print(f"  {k}: {v}")

    audit_log.log_event(
        "SAFETY_CHECK",
        {"status": report.status, "issues": report.issues, "metrics": report.metrics},
    )

    exit_codes = {HealthStatus.GREEN: 0, HealthStatus.YELLOW: 1, HealthStatus.RED: 2}
    sys.exit(exit_codes.get(report.status, 2))


def _run_live_shadow(mode: str):
    """Run the full pipeline against live broker state without placing any orders.

    Purpose: validate that AI decisions, sizing, and all risk gates produce correct
    output before the first real-money session. Emits WOULD_BUY log entries instead
    of submitting orders. Does NOT require LIVE_CONFIRM since nothing is executed.
    """
    print("\n=== InvestorBot Live Shadow Run ===")
    print(f"Mode: {mode.upper()}  |  {'PAPER' if config.IS_PAPER else '*** LIVE ACCOUNT ***'}")
    print("No orders will be placed. All gates run for real.\n")
    audit_log.log_event("LIVE_SHADOW_START", {"mode": mode, "is_paper": config.IS_PAPER})
    run(dry_run=True, mode=mode, _live_shadow=True)


# ── Broker account safety assertions ─────────────────────────────────────────


def _assert_account_safety(client) -> None:
    """Verify broker account type matches safety config. Raises RuntimeError if violated.

    Only enforced in live mode (IS_PAPER=False). Paper accounts may not expose
    the same account flags, so assertions are skipped there.
    """
    if config.IS_PAPER:
        return
    try:
        account = client.get_account()
        if not config.ALLOW_MARGIN:
            if getattr(account, "pattern_day_trader", False):
                raise RuntimeError(
                    "Broker account has PDT/margin flag. "
                    "Set ALLOW_MARGIN=true to override or switch to a cash account."
                )
            # Buying power > 2× equity implies margin. Cash accounts have 1× buying power.
            equity = float(account.equity or 0)
            buying_power = float(account.buying_power or 0)
            if equity > 0 and buying_power > equity * 2.1:
                raise RuntimeError(
                    f"Broker buying_power (${buying_power:.0f}) > 2× equity (${equity:.0f}) "
                    "suggests margin enabled. Set ALLOW_MARGIN=true to override."
                )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"_assert_account_safety: cannot verify account constraints — {e}. "
            "Resolve broker connectivity or set ALLOW_MARGIN=true if intentional."
        ) from e


# ── Stop failure → flatten helper ────────────────────────────────────────────


def _handle_stop_failure(client, symbol: str, dry_run: bool) -> None:
    """When stop placement fails, alert immediately and flatten in live mode.

    In paper/dry-run mode: alert only (no real money at risk).
    In live mode: flatten the position immediately. If flatten also fails,
    write a halt file — a live unprotected position is a fatal condition.
    """
    if dry_run:
        logger.warning(f"  [DRY RUN] Stop failed for {symbol} — would flatten in live mode")
        return
    logger.error(f"Stop placement FAILED for {symbol} — position unprotected!")
    alerts.alert_error(
        "STOP FAILED",
        f"{symbol}: trailing stop placement failed after buy — position has no downside protection.",
    )
    if config.IS_PAPER:
        return
    logger.critical(
        f"Stop placement FAILED for {symbol} — attempting emergency flatten to remove exposure"
    )
    alerts.alert_error(
        "STOP FAILED — FLATTENING",
        f"{symbol}: trailing stop placement failed after buy. Attempting to flatten position.",
    )
    flatten_result = trader.close_position(client, symbol)
    if flatten_result.is_success:
        trader.record_sell(symbol)
        logger.critical(f"  Emergency flatten of {symbol} succeeded — position closed")
        alerts.alert_error(
            "POSITION FLATTENED",
            f"{symbol}: emergency flatten succeeded after stop failure.",
        )
    else:
        logger.critical(
            f"  Emergency flatten of {symbol} FAILED ({flatten_result.rejection_reason}) — "
            "writing halt file. Manual broker action required immediately."
        )
        os.makedirs(config.LOG_DIR, exist_ok=True)
        halt_data = {
            "halted": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": "stop_failure_and_flatten_failure",
            "symbol": symbol,
            "flatten_rejection": flatten_result.rejection_reason,
        }
        with open(config.HALT_FILE, "w") as f:
            json.dump(halt_data, f, indent=2)
            f.write("\n")
        alerts.alert_error(
            "HALT — UNPROTECTED POSITION",
            f"{symbol}: stop failed AND flatten failed. Bot halted. "
            "Check broker immediately — position may still be open.",
        )


# ── Partial exits ─────────────────────────────────────────────────────────────


def _handle_partial_exits(client, positions: list, dry_run: bool) -> list:
    """Sell half of any position up more than PARTIAL_PROFIT_PCT (once per position)."""
    executed = []
    for pos in positions:
        if pos["unrealized_plpc"] >= config.PARTIAL_PROFIT_PCT:
            symbol = pos["symbol"]
            meta = trader.get_position_meta(symbol)
            if meta.get("partial_exit_taken_at"):
                logger.info(f"Partial exit already taken for {symbol} — skipping")
                continue
            half_qty = pos["qty"] / 2
            logger.info(
                f"Partial exit: {symbol} +{pos['unrealized_plpc']:.1f}% — selling {half_qty:.6f} shares"
            )
            if not dry_run:
                trader.cancel_open_orders(client, symbol)
                result = trader.place_partial_sell(client, symbol, half_qty)
                if result and result.is_success:
                    audit_log.log_order_placed(
                        symbol, "SELL_PARTIAL", pos["market_value"] / 2, result.broker_order_id
                    )
                    audit_log.log_order_filled(symbol, result.broker_order_id, result.filled_qty)
                    trader.record_partial_exit(symbol)
                    trader.place_trailing_stop(
                        client, symbol, pos["qty"] - half_qty, current_price=pos["current_price"]
                    )
                    executed.append(
                        {
                            "symbol": symbol,
                            "action": "PARTIAL SELL",
                            "detail": f"50% at +{pos['unrealized_plpc']:.1f}%",
                        }
                    )
            else:
                logger.info(f"  [DRY RUN] Would partial-sell {symbol}")
                executed.append({"symbol": symbol, "action": "PARTIAL SELL", "detail": "dry run"})
    return executed


# ── Main run ──────────────────────────────────────────────────────────────────


def run(dry_run: bool = False, mode: str = "open", _live_shadow: bool = False):
    today = config.today_et().isoformat()
    shadow_label = " [LIVE SHADOW — no orders]" if _live_shadow else ""
    mode_label = "PAPER" if config.IS_PAPER else "*** LIVE ***"
    logger.info(
        f"=== Trading bot | {today} | mode={mode} | {mode_label} {'[DRY RUN]' if dry_run else ''}{shadow_label} ==="
    )
    if (
        not config.IS_PAPER
        and not dry_run
        and not _live_shadow
        and config.LIVE_CONFIRM != "I-ACCEPT-REAL-MONEY-RISK"
    ):
        logger.error(
            "Live trading requires LIVE_CONFIRM=I-ACCEPT-REAL-MONEY-RISK in the environment. "
            "Set it in .env or export it before running."
        )
        sys.exit(1)

    if config.SMALL_ACCOUNT_MODE:
        logger.info(
            f"SMALL_ACCOUNT_MODE active: max_order=${config.MAX_SINGLE_ORDER_USD:.0f} "
            f"max_daily=${config.MAX_DAILY_NOTIONAL_USD:.0f} "
            f"max_deployed=${config.MAX_DEPLOYED_USD:.0f} "
            f"max_positions={config.MAX_POSITIONS}"
        )

    try:
        config.validate()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if not config.ALPACA_API_KEY or config.ALPACA_API_KEY == "your_alpaca_api_key_here":
        logger.error("ALPACA_API_KEY not set.")
        sys.exit(1)
    if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        logger.error("ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    # Halt file check — bot refuses to run while HALT file exists
    if os.path.exists(config.HALT_FILE):
        logger.critical("Trading is HALTED. Delete halt file or run: python main.py --clear-halt")
        sys.exit(1)

    init_db()

    lock_fd = _acquire_lock()
    if lock_fd is None:
        return

    try:
        _run_inner(dry_run=dry_run, mode=mode, today=today, _live_shadow=_live_shadow)
    except Exception as e:
        logger.error(f"Unhandled error in trading run: {e}", exc_info=True)
        alerts.alert_error("main.run", str(e))
    finally:
        _release_lock(lock_fd)


def _run_inner(dry_run: bool, mode: str, today: str, _live_shadow: bool = False):
    run_id = str(uuid.uuid4())
    audit_log.set_run_id(run_id)
    decision_log.set_run_id(run_id)
    logger.info(f"run_id={run_id}")

    # True when this run should exercise live safety gates even though no orders are placed.
    # Covers both real live runs and --live-shadow (full-pipeline validation mode).
    should_run_live_gates = (not dry_run) or _live_shadow

    client = trader.get_client()

    # ── Broker account safety assertions (live only) ──────────────────────────
    if not config.IS_PAPER and should_run_live_gates:
        try:
            _assert_account_safety(client)
        except RuntimeError as e:
            logger.critical(f"Account safety check failed: {e}")
            sys.exit(1)

    if not trader.is_market_open(client):
        logger.info("Market is closed. Nothing to do.")
        return

    # ── Account snapshot ──────────────────────────────────────────────────────
    account_before = trader.get_account_info(client)
    logger.info(
        f"Portfolio: ${account_before['portfolio_value']:.2f}  Cash: ${account_before['cash']:.2f}"
    )
    audit_log.log_run_start(
        mode, account_before["portfolio_value"], account_before["cash"], config.IS_PAPER
    )

    if mode == "open":
        portfolio_tracker.save_daily_baseline(account_before["portfolio_value"])

    # Set the experiment-start equity once, on the first live open run.
    # Never overwritten — used to enforce MAX_EXPERIMENT_DRAWDOWN_USD for the lifetime of the run.
    if not config.IS_PAPER and not dry_run and mode == "open":
        save_experiment_baseline(account_before["portfolio_value"])

    # ── Reconcile position metadata ───────────────────────────────────────────
    # reconcile_positions returns symbols that exist at the broker but had no
    # local record — unexpected positions detected BEFORE they are normalised.
    unexpected_positions = trader.reconcile_positions(client)
    if unexpected_positions and not config.IS_PAPER and should_run_live_gates:
        msg = (
            f"Unexpected broker position(s) with no local record: {sorted(unexpected_positions)}. "
            "These have been added as placeholders. Verify origin before continuing."
        )
        logger.critical(msg)
        audit_log.log_event("UNEXPECTED_POSITIONS", {"symbols": sorted(unexpected_positions)})
        alerts.alert_error("UNEXPECTED BROKER POSITIONS", msg)
        # Write halt — unknown positions means broker state cannot be trusted
        os.makedirs(config.LOG_DIR, exist_ok=True)
        with open(config.HALT_FILE, "w") as _hf:
            json.dump(
                {
                    "halted": True,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "reason": "unexpected_broker_positions",
                    "symbols": sorted(unexpected_positions),
                },
                _hf,
                indent=2,
            )
            _hf.write("\n")
        sys.exit(1)

    stops_ok = trader.ensure_stops_attached(client)
    if not stops_ok and not config.IS_PAPER and not dry_run:
        logger.critical(
            "ensure_stops_attached reported uncovered live positions — halting to prevent "
            "unprotected exposure. Resolve stops manually, then run --clear-halt."
        )
        os.makedirs(config.LOG_DIR, exist_ok=True)
        halt_data = {
            "halted": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": "uncovered_positions_at_startup",
        }
        with open(config.HALT_FILE, "w") as f:
            json.dump(halt_data, f, indent=2)
            f.write("\n")
        alerts.alert_error(
            "HALT — UNCOVERED POSITIONS",
            "Bot halted at startup: one or more live positions have no stop protection. "
            "Attach stops manually, then run --clear-halt.",
        )
        sys.exit(1)

    # ── Startup health check ──────────────────────────────────────────────────
    health = run_startup_health_check(client)
    health.log()
    audit_log.log_event(
        "STARTUP_HEALTH",
        {"status": health.status, "issues": health.issues, "metrics": health.metrics},
    )
    if health.status == HealthStatus.RED:
        logger.critical(
            f"Startup health check RED ({len(health.issues)} issue(s)) — "
            "new buys suspended this run. Resolve issues and restart."
        )
        alerts.alert_error(
            "STARTUP HEALTH RED",
            f"{len(health.issues)} safety issue(s) at startup: " + "; ".join(health.issues[:3]),
        )
        # In live non-dry-run mode a RED health is fatal for buys; we continue
        # to allow sells/exits so existing positions can be managed.
    elif health.status == HealthStatus.YELLOW:
        logger.warning(
            f"Startup health check YELLOW ({len(health.issues)} issue(s)) — "
            "new buys suspended this run."
        )

    # ── Circuit breaker ───────────────────────────────────────────────────────
    history = portfolio_tracker.load_history()
    cb_triggered, cb_drawdown = risk_manager.check_circuit_breaker(history)
    if cb_triggered:
        audit_log.log_circuit_breaker(cb_drawdown)
        alerts.alert_circuit_breaker(cb_drawdown)
        logger.warning("Circuit breaker active — no new buys today.")

    # ── Daily loss check ──────────────────────────────────────────────────────
    _baseline = portfolio_tracker.load_daily_baseline() or account_before["portfolio_value"]
    dl_triggered, dl_pct = risk_manager.check_daily_loss(
        _baseline, account_before["portfolio_value"]
    )
    # Dollar-denominated daily loss cap (active when MAX_DAILY_LOSS_USD > 0)
    if not dl_triggered and config.MAX_DAILY_LOSS_USD > 0:
        daily_loss_usd = _baseline - account_before["portfolio_value"]
        if daily_loss_usd >= config.MAX_DAILY_LOSS_USD:
            dl_triggered = True
            dl_pct = (account_before["portfolio_value"] / _baseline - 1) * 100
            logger.warning(
                f"Dollar daily loss limit hit: ${daily_loss_usd:.2f} >= "
                f"${config.MAX_DAILY_LOSS_USD:.2f}"
            )
    if dl_triggered:
        audit_log.log_daily_loss_limit(dl_pct)
        alerts.alert_daily_loss(dl_pct)
        logger.warning("Daily loss limit hit — closing all positions.")
        if not dry_run:
            for pos in trader.get_open_positions(client):
                trader.close_position(client, pos["symbol"])
                audit_log.log_position_closed(
                    pos["symbol"], "daily_loss_limit", pos["unrealized_plpc"]
                )
                trader.record_sell(pos["symbol"])
        return

    # ── Experiment drawdown cap ───────────────────────────────────────────────
    # Active when MAX_EXPERIMENT_DRAWDOWN_USD > 0 and a baseline has been set.
    # Blocks new buys (but allows sells/exits) once total experiment loss is reached.
    _exp_drawdown_triggered = False
    if config.MAX_EXPERIMENT_DRAWDOWN_USD > 0:
        _exp_baseline = load_experiment_baseline()
        if _exp_baseline is not None:
            _exp_loss = _exp_baseline - account_before["portfolio_value"]
            if _exp_loss >= config.MAX_EXPERIMENT_DRAWDOWN_USD:
                _exp_drawdown_triggered = True
                logger.critical(
                    f"Experiment drawdown cap reached: lost ${_exp_loss:.2f} of "
                    f"${_exp_baseline:.2f} start equity "
                    f"(limit ${config.MAX_EXPERIMENT_DRAWDOWN_USD:.2f}) — "
                    "new buys blocked for remainder of experiment."
                )
                audit_log.log_event(
                    "EXPERIMENT_DRAWDOWN_CAP",
                    {
                        "start_equity": _exp_baseline,
                        "current_equity": account_before["portfolio_value"],
                        "loss_usd": round(_exp_loss, 2),
                        "limit_usd": config.MAX_EXPERIMENT_DRAWDOWN_USD,
                    },
                )
                alerts.alert_error(
                    "EXPERIMENT DRAWDOWN CAP",
                    f"Total experiment loss ${_exp_loss:.2f} >= "
                    f"${config.MAX_EXPERIMENT_DRAWDOWN_USD:.2f} limit — no new buys.",
                )

    all_trades: list = []
    daily_notional_spent = trader.get_daily_notional(today)  # persisted across runs on same date
    executed_symbols: set[str] = set()

    # ── Market context ────────────────────────────────────────────────────────
    logger.info("Fetching market context...")
    vix = market_data.get_vix()
    regime = stock_scanner.get_market_regime(config.BEAR_MARKET_SPY_THRESHOLD, vix=vix)
    macro = macro_calendar.get_macro_risk()
    sector_perf = sector_data.get_sector_performance()
    leading_sectors = sector_data.get_leading_sectors(top_n=3)
    lessons = get_latest_review()

    if vix:
        logger.info(f"VIX: {vix}  Regime: {regime.get('regime', 'UNKNOWN')}")
    if macro["is_high_risk"]:
        logger.warning(f"Macro risk: {macro['event']}")
        audit_log.log_macro_skip(macro["event"])

    # ── Open positions & earnings guard ──────────────────────────────────────
    open_positions = trader.get_open_positions(client)
    held_symbols = {p["symbol"] for p in open_positions}
    earnings_risk = earnings_calendar.get_earnings_risk_positions(
        list(held_symbols), config.EARNINGS_WARNING_DAYS
    )

    # ── Partial exits (all modes) ─────────────────────────────────────────────
    partials = _handle_partial_exits(client, open_positions, dry_run)
    all_trades.extend(partials)

    open_positions = trader.get_open_positions(client)
    held_symbols = {p["symbol"] for p in open_positions}

    # ── Full cycle (all modes) ────────────────────────────────────────────────

    # Exit earnings-risk positions
    for symbol, ed in earnings_risk.items():
        if symbol in held_symbols:
            logger.warning(f"Exiting {symbol} — earnings on {ed}")
            audit_log.log_earnings_exit(symbol, str(ed))
            if not dry_run:
                result = trader.close_position(client, symbol)
                if result.is_success:
                    meta = trader.get_position_meta(symbol)
                    pos = next((p for p in open_positions if p["symbol"] == symbol), None)
                    if pos:
                        performance.record_trade_outcome(
                            meta["signal"],
                            pos["unrealized_plpc"],
                            regime=meta["regime"],
                            confidence=meta["confidence"],
                        )
                        audit_log.log_position_closed(
                            symbol, "earnings_exit", pos["unrealized_plpc"]
                        )
                    trader.record_sell(symbol)
                    all_trades.append(
                        {"symbol": symbol, "action": "SELL", "detail": "earnings exit"}
                    )
                else:
                    logger.error(
                        f"  Earnings exit FAILED {symbol} — {result.rejection_reason or 'close failed after retries'}"
                    )

    open_positions = trader.get_open_positions(client)
    held_symbols = {p["symbol"] for p in open_positions}
    position_ages = trader.get_position_ages()
    stale = [
        sym
        for sym, age in position_ages.items()
        if age
        >= config.SIGNAL_MAX_HOLD_DAYS.get(
            trader.get_position_meta(sym).get("signal", "unknown"),
            config.MAX_HOLD_DAYS,
        )
    ]

    # ── Scan universe ─────────────────────────────────────────────────────────
    logger.info("Scanning for top movers...")
    top_movers = stock_scanner.get_top_movers(config.TOP_MOVERS_COUNT)
    scan_symbols = list(set(build_scan_universe(client)) | held_symbols | set(top_movers))
    logger.info(f"Scanning {len(scan_symbols)} symbols")

    # ── Market data (parallel) ────────────────────────────────────────────────
    logger.info("Fetching market data...")
    snapshots = market_data.get_market_snapshots(scan_symbols, config.LOOKBACK_DAYS)
    if not snapshots:
        logger.error("No market data. Aborting.")
        return

    # ── Pre-filter buy candidates ─────────────────────────────────────────────
    held_snaps = [s for s in snapshots if s["symbol"] in held_symbols]
    candidate_snaps = [s for s in snapshots if s["symbol"] not in held_symbols]

    # Small-account universe price filter — restricts to names where one whole share
    # can be stop-protected within the per-order cap.
    if config.MIN_PRICE_USD > 0 or config.MAX_PRICE_USD > 0:
        before_price_filter = len(candidate_snaps)
        candidate_snaps = [
            s
            for s in candidate_snaps
            if (config.MIN_PRICE_USD == 0 or s.get("current_price", 0) >= config.MIN_PRICE_USD)
            and (config.MAX_PRICE_USD == 0 or s.get("current_price", 0) <= config.MAX_PRICE_USD)
        ]
        logger.info(
            f"Price filter (${config.MIN_PRICE_USD:.0f}–${config.MAX_PRICE_USD:.0f}): "
            f"{before_price_filter} → {len(candidate_snaps)} candidates"
        )

    filtered_candidates = stock_scanner.prefilter_candidates(candidate_snaps)
    ai_snapshots = held_snaps + filtered_candidates
    logger.info(
        f"Pre-filter: {len(candidate_snaps)} candidates → {len(filtered_candidates)} passed"
    )

    # Symbols the AI actually received — used as the validation universe.
    # Broader known_symbols (all fetched) would allow hallucinated tickers from
    # top_movers that never made it through the pre-filter.
    ai_known_symbols = {s["symbol"] for s in ai_snapshots}

    # ── Options flow ──────────────────────────────────────────────────────────
    options_syms = [s["symbol"] for s in filtered_candidates]
    options_sigs = options_scanner.get_options_signals(options_syms) if options_syms else {}
    if options_sigs:
        logger.info(f"Options signals fetched for: {list(options_sigs.keys())}")

    # ── News (sanitized against prompt injection) ─────────────────────────────
    logger.info("Fetching news and sentiment...")
    raw_news = news_fetcher.fetch_news(scan_symbols)
    news = sanitize_headlines(raw_news)

    sent = sentiment_module.get_sentiment(list(held_symbols) + top_movers[:10])

    # ── AI analysis ──────────────────────────────────────────────────────────
    track_record = portfolio_tracker.get_track_record(10)
    account_now = trader.get_account_info(client)
    logger.info("Running AI analysis...")
    decisions = ai_analyst.get_trading_decisions(
        snapshots=ai_snapshots,
        current_positions=open_positions,
        available_cash=account_now["cash"],
        portfolio_value=account_now["portfolio_value"],
        news_by_symbol=news,
        track_record=track_record,
        market_regime=regime,
        position_ages=position_ages,
        stale_positions=stale,
        vix=vix,
        sector_performance=sector_perf,
        sentiment=sent,
        earnings_risk={sym: str(ed) for sym, ed in earnings_risk.items()},
        macro_risk=macro,
        leading_sectors=leading_sectors,
        options_signals=options_sigs,
        lessons=lessons,
        run_id=run_id,
    )

    if not decisions:
        logger.error("AI analysis failed. Aborting.")
        return

    # ── Validate AI response before acting ───────────────────────────────────
    is_valid, validation_errors = validate_ai_response(
        decisions, ai_known_symbols, held_symbols=held_symbols
    )
    if not is_valid:
        audit_log.log_validation_failure(validation_errors)
        buy_domain_only = validation_errors and all(
            e.startswith("BUY candidate '") for e in validation_errors
        )
        if buy_domain_only:
            # BUY domain errors (out-of-universe, already-held) only taint buy decisions.
            # Sell decisions are independent and must still execute.
            logger.warning(
                f"AI response has {len(validation_errors)} BUY domain error(s) — "
                f"blocking buys only, preserving sell decisions: {validation_errors}"
            )
            decisions["buy_candidates"] = []
        else:
            # Structural/schema errors: the whole response is untrustworthy.
            logger.error(
                f"AI response validation failed ({len(validation_errors)} structural error(s)) — "
                f"blocking all Claude-driven decisions: {validation_errors}"
            )
            alerts.alert_error(
                "VALIDATION FAILURE",
                f"AI response invalid ({len(validation_errors)} errors) — no Claude orders this run",
            )
            decisions["buy_candidates"] = []
            decisions["position_decisions"] = []

    logger.info(f"Market: {decisions.get('market_summary', '')}")
    decision_log.log_decisions(decisions, mode, executed_symbols)
    audit_log.log_ai_decision(
        decisions.get("market_summary", ""),
        len(decisions.get("buy_candidates", [])),
        sum(1 for d in decisions.get("position_decisions", []) if d.get("action") == "SELL"),
    )

    # ── Execute sells ─────────────────────────────────────────────────────────
    symbols_to_sell = {
        d["symbol"] for d in decisions.get("position_decisions", []) if d["action"] == "SELL"
    }
    symbols_to_sell |= {sym for sym in stale if sym in held_symbols}

    for symbol in symbols_to_sell:
        decision = next(
            (d for d in decisions.get("position_decisions", []) if d["symbol"] == symbol), None
        )
        if decision:
            reason = decision["reasoning"]
        else:
            signal = trader.get_position_meta(symbol).get("signal", "unknown")
            limit = config.SIGNAL_MAX_HOLD_DAYS.get(signal, config.MAX_HOLD_DAYS)
            reason = f"Time-based exit (≥{limit} days for {signal} signal)"
        logger.info(f"  SELL {symbol} — {reason}")
        if not dry_run:
            pos = next((p for p in open_positions if p["symbol"] == symbol), None)
            result = trader.close_position(client, symbol)
            if result.is_success:
                if pos:
                    meta = trader.get_position_meta(symbol)
                    performance.record_trade_outcome(
                        meta["signal"],
                        pos["unrealized_plpc"],
                        regime=meta["regime"],
                        confidence=meta["confidence"],
                    )
                    audit_log.log_position_closed(symbol, reason[:50], pos["unrealized_plpc"])
                trader.record_sell(symbol)
                executed_symbols.add(symbol)
                all_trades.append({"symbol": symbol, "action": "SELL", "detail": reason})
            else:
                logger.error(
                    f"  SELL FAILED {symbol} — {result.rejection_reason or 'close failed after retries'}. Manual review required."
                )
                alerts.alert_error(
                    "SELL FAILED",
                    f"{symbol}: failed to close position — {result.rejection_reason or 'unknown error'}",
                )
        else:
            executed_symbols.add(symbol)
            all_trades.append({"symbol": symbol, "action": "SELL", "detail": "dry run"})

    # ── Execute buys (open mode only; midday/close are position-management runs) ──
    skip_buys = (
        mode in ("midday", "close", "open_sells")
        or cb_triggered
        or regime.get("is_bearish")
        or macro.get("is_high_risk")
        or health.status in (HealthStatus.RED, HealthStatus.YELLOW)
        or _exp_drawdown_triggered
    )
    if skip_buys:
        reasons = []
        if mode in ("midday", "close"):
            reasons.append(f"{mode} mode")
        if cb_triggered:
            reasons.append("circuit breaker")
        if regime.get("is_bearish"):
            reasons.append("bear market filter")
        if macro.get("is_high_risk"):
            reasons.append(f"macro event: {macro.get('event')}")
        if health.status in (HealthStatus.RED, HealthStatus.YELLOW):
            reasons.append(f"startup health {health.status}")
        if _exp_drawdown_triggered:
            reasons.append("experiment drawdown cap reached")
        logger.warning(f"Skipping new buys: {', '.join(reasons)}")
    else:
        account_now = trader.get_account_info(client)
        open_positions = trader.get_open_positions(client)
        available_cash = account_now["cash"] * (1 - config.CASH_RESERVE_PCT)
        max_positions = min(
            position_sizer.get_max_positions(account_now["portfolio_value"]),
            config.MAX_POSITIONS,
        )
        slots = max_positions - len(open_positions)
        logger.info(f"Position slots: {len(open_positions)}/{max_positions}")

        if slots > 0:
            regime_name = regime.get("regime", "UNKNOWN")
            # Mechanical regime gates — tighter than verbal prompt advice alone
            if regime_name == "CHOPPY":
                regime_max_orders = 1
                regime_conf_bump = 1
            elif regime_name == "HIGH_VOL":
                regime_max_orders = 2
                regime_conf_bump = 1
            else:
                regime_max_orders = config.MAX_ORDERS_PER_RUN
                regime_conf_bump = 0
            effective_max_orders = min(config.MAX_ORDERS_PER_RUN, regime_max_orders)

            min_confidence = (
                config.MIN_CONFIDENCE + regime_conf_bump + (1 if vix and vix > 25 else 0)
            )
            if regime_conf_bump:
                logger.info(
                    f"Regime {regime_name}: min_confidence raised to {min_confidence}, max_orders capped at {effective_max_orders}"
                )

            raw_candidates = [
                c for c in decisions.get("buy_candidates", []) if c["confidence"] >= min_confidence
            ]
            valid_candidates = risk_manager.validate_buy_candidates(
                raw_candidates,
                held_symbols={p["symbol"] for p in open_positions},
                sector_map_fn=sector_data.get_sector,
                max_per_sector=config.MAX_SECTOR_POSITIONS,
            )
            valid_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            valid_candidates = valid_candidates[:slots]

            # VIX-adjusted trail percent — wider stops in high-vol regimes
            vix_trail_pct = risk_manager.check_vix_stop_adjustment(vix)
            if abs(vix_trail_pct - config.TRAILING_STOP_PCT) > 0.01:
                logger.info(
                    f"VIX-adjusted trail: {config.TRAILING_STOP_PCT}% → {vix_trail_pct}% "
                    f"(VIX={vix})"
                )

            orders_placed = 0
            for candidate in valid_candidates:
                if orders_placed >= effective_max_orders:
                    logger.warning(
                        f"MAX_ORDERS_PER_RUN ({config.MAX_ORDERS_PER_RUN}) reached — no more buys this run"
                    )
                    break
                symbol = candidate["symbol"]
                confidence = candidate["confidence"]

                # Order-intent ledger guard — survives process restarts unlike broker queries.
                # Blocks re-submission when a prior same-day intent is still active.
                # OrderLedgerUnavailable means the guard is inoperative → suspend all buys.
                try:
                    from utils.order_ledger import has_active_intent

                    if has_active_intent(symbol, "BUY", today):
                        logger.warning(
                            f"  Skipping {symbol}: active order intent in ledger for today"
                        )
                        continue
                except ImportError:
                    pass
                except OrderLedgerUnavailable as e:
                    logger.error(f"Order ledger unavailable — suspending all buys this run: {e}")
                    alerts.alert_error("ORDER LEDGER UNAVAILABLE", str(e))
                    break

                # Pending-order guard — prevents duplicate buys after timeout/restart.
                # BrokerStateUnavailable means we cannot verify safety → suspend all buys.
                if should_run_live_gates:
                    try:
                        if trader.has_pending_buy(client, symbol):
                            logger.warning(
                                f"  Skipping {symbol}: broker already has a pending buy order for this symbol"
                            )
                            continue
                    except BrokerStateUnavailable as e:
                        logger.error(
                            f"Broker state unavailable — suspending all buys this run: {e}"
                        )
                        alerts.alert_error("BROKER STATE UNAVAILABLE", str(e))
                        break

                # Size order
                if config.SMALL_ACCOUNT_MODE:
                    notional = min(
                        position_sizer.small_account_size(
                            account_now["portfolio_value"],
                            max_single_order=config.MAX_SINGLE_ORDER_USD,
                        ),
                        available_cash,
                    )
                else:
                    notional = min(
                        position_sizer.risk_budget_size(
                            account_now["portfolio_value"],
                            confidence,
                            signal=candidate.get("key_signal", "unknown"),
                            regime=regime.get("regime", "UNKNOWN"),
                        ),
                        available_cash,
                    )

                # Pre-trade controls (MiFID II) — includes open-exposure cap when configured.
                # BrokerStateUnavailable means we cannot verify exposure → suspend all buys.
                if should_run_live_gates:
                    try:
                        open_exposure = trader.get_total_open_exposure(client)
                    except BrokerStateUnavailable as e:
                        logger.error(
                            f"Cannot query open exposure — suspending all buys this run: {e}"
                        )
                        alerts.alert_error("BROKER STATE UNAVAILABLE", str(e))
                        break
                else:
                    open_exposure = 0.0
                approved, rejection_reason = check_pre_trade(
                    symbol,
                    notional,
                    daily_notional_spent,
                    config.MAX_SINGLE_ORDER_USD,
                    config.MAX_DAILY_NOTIONAL_USD,
                    open_exposure_usd=open_exposure,
                    max_deployed_usd=config.MAX_DEPLOYED_USD,
                )
                if not approved:
                    logger.warning(f"  Pre-trade check failed: {rejection_reason}")
                    continue

                logger.info(f"  BUY {symbol}: ${notional:.2f} | conf={confidence}")

                qg = None  # populated by quote gate in live mode; used for execution quality log
                # Live quote gate — validates real-time conditions before order submission.
                # Runs in live mode and live-shadow mode; skipped in dry-run and paper mode.
                if should_run_live_gates and not config.IS_PAPER:
                    try:
                        qg = check_quote_gate(symbol, notional)
                        if not qg.approved:
                            logger.warning(f"  Quote gate rejected {symbol}: {qg.reject_reason}")
                            audit_log.log_event(
                                "QUOTE_GATE_REJECTED",
                                {
                                    "symbol": symbol,
                                    "reason": qg.reject_reason,
                                    "bid": qg.bid,
                                    "ask": qg.ask,
                                    "spread_bps": qg.spread_bps,
                                },
                            )
                            continue
                        logger.info(
                            f"  Quote gate OK {symbol}: bid={qg.bid:.2f} ask={qg.ask:.2f} "
                            f"spread={qg.spread_bps:.1f}bps age={qg.quote_age_seconds:.1f}s"
                        )
                    except BrokerStateUnavailable as e:
                        logger.error(
                            f"Quote gate data API unavailable — suspending all buys this run: {e}"
                        )
                        alerts.alert_error("BROKER STATE UNAVAILABLE", str(e))
                        break

                snap = next((s for s in snapshots if s["symbol"] == symbol), None)
                if notional >= 1.0:
                    # Guard: skip if notional buys < 1 whole share — Alpaca cannot stop-protect sub-share positions
                    if snap and notional / snap["current_price"] < 1.0:
                        logger.warning(
                            f"  Skipping {symbol}: ${notional:.2f} at ${snap['current_price']:.2f}"
                            f" = {notional / snap['current_price']:.3f} shares — sub-share position cannot be stop-protected"
                        )
                        continue
                    if not dry_run:
                        t_buy_submit = time.monotonic()
                        result = trader.place_buy_order(client, symbol, notional)
                        t_fill = time.monotonic()
                        if result and result.is_success:
                            fill_latency_ms = round((t_fill - t_buy_submit) * 1000)
                            orders_placed += 1
                            daily_notional_spent += notional
                            trader.add_daily_notional(today, notional)
                            entry_price = snap["current_price"] if snap else 0.0
                            trader.record_buy(
                                symbol,
                                entry_price,
                                signal=candidate.get("key_signal", "unknown"),
                                regime=regime.get("regime", "UNKNOWN"),
                                confidence=confidence,
                            )
                            audit_log.log_order_placed(
                                symbol, "BUY", notional, result.broker_order_id
                            )
                            if result.filled_qty:
                                audit_log.log_order_filled(
                                    symbol, result.broker_order_id, result.filled_qty
                                )
                                current_price = snap["current_price"] if snap else None
                                # Floor to whole shares — Alpaca rejects fractional stop orders
                                stop_qty = int(math.floor(result.filled_qty))
                                t_stop_submit = time.monotonic()
                                stop_result = (
                                    trader.place_trailing_stop(
                                        client,
                                        symbol,
                                        stop_qty,
                                        current_price=current_price,
                                        trail_percent=vix_trail_pct,
                                    )
                                    if stop_qty >= 1
                                    else None
                                )
                                t_stop_accept = time.monotonic()
                                unprotected_ms = round((t_stop_submit - t_fill) * 1000)
                                stop_latency_ms = round((t_stop_accept - t_stop_submit) * 1000)
                                audit_log.log_event(
                                    "ORDER_TIMING",
                                    {
                                        "symbol": symbol,
                                        "fill_latency_ms": fill_latency_ms,
                                        "unprotected_window_ms": unprotected_ms,
                                        "stop_latency_ms": stop_latency_ms,
                                        "stop_ok": stop_result is not None
                                        and stop_result.is_success,
                                    },
                                )
                                fill_avg = result.filled_avg_price
                                if qg and qg.bid and qg.ask and fill_avg:
                                    mid = (qg.bid + qg.ask) / 2
                                    slippage_bps = (
                                        round((fill_avg - mid) / mid * 10_000, 1) if mid else None
                                    )
                                    audit_log.log_event(
                                        "ORDER_EXEC_QUALITY",
                                        {
                                            "symbol": symbol,
                                            "bid": qg.bid,
                                            "ask": qg.ask,
                                            "mid": round(mid, 4),
                                            "spread_bps": qg.spread_bps,
                                            "fill_avg_price": fill_avg,
                                            "slippage_vs_mid_bps": slippage_bps,
                                            "fill_latency_ms": fill_latency_ms,
                                        },
                                    )
                                if stop_result is None or not stop_result.is_success:
                                    _handle_stop_failure(client, symbol, dry_run)
                                    # Position may now be closed or halted — don't count as successful trade
                                    continue
                            executed_symbols.add(symbol)
                            detail = f"${notional:.2f} | {candidate.get('key_signal')} | confidence={confidence}"
                            all_trades.append({"symbol": symbol, "action": "BUY", "detail": detail})
                        elif result and result.status in (
                            OrderStatus.PARTIAL,
                            OrderStatus.TIMEOUT,
                        ):
                            # Ambiguous fill — broker may hold exposure we didn't fully record.
                            # Re-query immediately and attach stops; don't wait until next startup.
                            logger.warning(
                                f"  BUY {symbol}: ambiguous result "
                                f"status={result.status.name} "
                                f"filled_qty={result.filled_qty} — "
                                "running immediate stop coverage check"
                            )
                            audit_log.log_event(
                                "ORDER_AMBIGUOUS",
                                {
                                    "symbol": symbol,
                                    "status": result.status.name,
                                    "filled_qty": result.filled_qty,
                                    "broker_order_id": result.broker_order_id,
                                },
                            )
                            alerts.alert_error(
                                "ORDER AMBIGUOUS",
                                f"{symbol} buy returned {result.status.name} — "
                                "checking stop coverage now.",
                            )
                            immediate_stops_ok = trader.ensure_stops_attached(client)
                            if not immediate_stops_ok:
                                _handle_stop_failure(client, symbol, dry_run)
                    else:
                        orders_placed += 1
                        daily_notional_spent += notional
                        executed_symbols.add(symbol)
                        all_trades.append(
                            {
                                "symbol": symbol,
                                "action": "BUY",
                                "detail": f"dry run ${notional:.2f}",
                            }
                        )
                else:
                    logger.warning(f"  Skipping {symbol}: ${notional:.2f} too small")

    # ── Attach any missing stops (catches fills that arrived after wait_for_fill timed out) ──
    if not dry_run and all_trades:
        trader.ensure_stops_attached(client)

    # ── Finalise ──────────────────────────────────────────────────────────────
    account_after = trader.get_account_info(client)
    save_date = today if mode == "open" else f"{today}-{mode}"
    record = portfolio_tracker.save_daily_run(
        date=save_date,
        account_before=account_before,
        account_after=account_after,
        ai_decisions=decisions,
        trades_executed=all_trades,
        stop_losses_triggered=[],
        run_id=run_id,
    )
    portfolio_tracker.print_summary(record)
    performance.generate_dashboard(portfolio_tracker.load_history())
    audit_log.log_run_end(
        mode, record["daily_pnl"], len(all_trades), account_after["portfolio_value"]
    )
    if mode == "close" and not dry_run:
        day_summary = get_day_summary(today)
        if day_summary:
            emailer.send_summary(day_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=["open", "open_sells", "midday", "close"], default="open")
    parser.add_argument(
        "--kill-switch", action="store_true", help="Emergency: liquidate all and halt"
    )
    parser.add_argument(
        "--clear-halt", action="store_true", help="Remove halt file and resume trading"
    )
    parser.add_argument(
        "--safety-check",
        action="store_true",
        help="Verify broker state, stops, and caps without trading. Exit 0=GREEN 1=YELLOW 2=RED",
    )
    parser.add_argument(
        "--live-shadow",
        action="store_true",
        help="Full pipeline run against real broker state; logs WOULD_BUY decisions without placing orders",
    )
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Starting capital for backtest. Defaults to current Alpaca portfolio value.",
    )
    args = parser.parse_args()

    if args.kill_switch:
        _run_kill_switch()
    elif args.clear_halt:
        _run_clear_halt()
    elif args.safety_check:
        init_db()
        _run_safety_check()
    elif args.live_shadow:
        _run_live_shadow(args.mode)
    elif args.backtest:
        capital = args.capital
        if capital is None:
            try:
                client = trader.get_client()
                capital = trader.get_account_info(client)["portfolio_value"]
                logger.info(f"Seeding backtest with current portfolio value: ${capital:,.2f}")
            except Exception as e:
                logger.warning(f"Could not fetch account value ({e}) — defaulting to $100,000")
                capital = 100000.0
        from backtest import run_backtest

        run_backtest(
            config.STOCK_UNIVERSE,
            args.start,
            args.end,
            capital,
            max_positions=config.MAX_POSITIONS,
        )
    else:
        run(dry_run=args.dry_run, mode=args.mode)
