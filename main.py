"""
Daily trading run.

Usage:
    python main.py                     # Full cycle at market open
    python main.py --mode midday       # Manage positions, partial exits, buys on intraday signals
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
import signal
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import config
from analysis import ai_analyst, performance
from analysis.weekly_review import get_latest_review
from data import (
    av_sentiment,
    earnings_surprise,
    edgar_client,
    insider_feed,
    market_data,
    news_fetcher,
    options_scanner,
    sector_data,
    short_interest,
)
from data import (
    options_data as options_data_module,
)
from data import (
    sentiment as sentiment_module,
)
from data.macro_data import get_macro_snapshot
from data.sentiment_client import get_sentiment_snapshot
from execution import short_risk, stock_scanner, trader
from execution.quote_gate import check_quote_gate
from execution.short_universe import get_short_universe, scan_short_universe
from execution.universe import build_scan_universe
from models import (
    BrokerStateUnavailable,
    DataBundle,
    MarketContext,
    OrderLedgerUnavailable,
    OrderResult,
    OrderStatus,
    PositionSnapshot,
    RiskFlags,
)
from notifications import alerts, emailer
from risk import (
    correlation,
    earnings_calendar,
    exit_optimiser,
    macro_calendar,
    position_sizer,
    risk_manager,
)
from risk.regime_policy import get_regime_policy
from risk.risk_config import RiskConfig
from signals.evaluator import INTRADAY_SHORT_SIGNALS, INTRADAY_SIGNALS
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
_current_lock_fd: int | None = None  # module-level so the SIGTERM handler can reach it


def _sigterm_handler(signum, frame):
    """Release the lock file on SIGTERM so the next scheduled run isn't blocked.

    Python's finally blocks don't execute when a process receives SIGTERM (the
    default signal sent by tmux kill-session, systemctl stop, and most process
    managers).  Without this handler a killed run leaves a stale .lock file that
    blocks every subsequent run for up to _LOCK_MAX_AGE_SECONDS (30 min).
    """
    global _current_lock_fd
    if _current_lock_fd is not None:
        _release_lock(_current_lock_fd)
        _current_lock_fd = None
    sys.exit(0)


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


def _fetch_atr_for_held(held_symbols: set) -> dict:
    """Return {symbol: atr_pct_or_None} for every currently held symbol."""
    return {sym: exit_optimiser.compute_atr_pct(sym) for sym in held_symbols}


_RS_DECAY_SIGNALS = frozenset({"rs_leader", "momentum", "momentum_12_1"})
_DEFENSIVE_REGIMES = frozenset({"DEFENSIVE_DOWNTREND", "BEAR_MARKET"})


def _fetch_adverse_vol_for_held(held_symbols: set) -> dict:
    """Return per-symbol vol_ratio and daily return for today and yesterday.

    Used by adverse_volume_triggered to detect two consecutive institutional-selling days.
    Requires 30d of daily bars to compute a proper 20-day volume average.
    Returns {} for any symbol where data is unavailable.
    """
    result = {}
    for sym in held_symbols:
        try:
            import pandas as pd
            import yfinance as yf

            df = yf.download(sym, period="30d", interval="1d", progress=False, auto_adjust=True)
            if df is None or len(df) < 3:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"]
            volume = df["Volume"]
            avg_vol = volume.rolling(20).mean()
            today_avg = float(avg_vol.iloc[-1])
            yday_avg = float(avg_vol.iloc[-2])
            if today_avg <= 0 or yday_avg <= 0 or today_avg != today_avg or yday_avg != yday_avg:
                continue
            result[sym] = {
                "vol_ratio_today": round(float(volume.iloc[-1]) / today_avg, 2),
                "ret_today": round(float((close.iloc[-1] / close.iloc[-2] - 1) * 100), 2),
                "vol_ratio_yday": round(float(volume.iloc[-2]) / yday_avg, 2),
                "ret_yday": round(float((close.iloc[-2] / close.iloc[-3] - 1) * 100), 2),
            }
        except Exception as e:
            logger.debug(f"_fetch_adverse_vol_for_held({sym}): {e}")
    return result


def _check_rule_based_stops(
    positions: list,
    position_ages: dict,
    atr_by_symbol: dict,
    snapshots_by_symbol: dict | None = None,
) -> set:
    """Return symbols that breach the hard stop, time-decay stop, or RS decay threshold."""
    rc = RiskConfig.from_config()
    symbols_to_exit: set = set()
    for pos in positions:
        symbol = pos["symbol"]
        meta = trader.get_position_meta(symbol)
        signal = meta.get("signal", "unknown")
        max_hold = config.SIGNAL_MAX_HOLD_DAYS.get(signal, config.MAX_HOLD_DAYS)
        days_held = position_ages.get(symbol, 1)
        levels = exit_optimiser.compute_exit_levels(
            rc.stop_loss_pct,
            rc.take_profit_pct,
            atr_by_symbol.get(symbol),
            days_held,
            max_hold,
        )
        plpc = pos["unrealized_plpc"]
        if plpc <= levels["stop_pct"]:
            logger.info(
                f"Rule-based hard stop hit: {symbol} {plpc:.2f}% <= {levels['stop_pct']:.2f}%"
            )
            symbols_to_exit.add(symbol)
        elif levels["apply_timedecay"] and plpc <= levels["timedecay_stop_pct"]:
            logger.info(
                f"Time-decay stop hit: {symbol} {plpc:.2f}% <= {levels['timedecay_stop_pct']:.2f}% "
                f"(day {days_held}/{max_hold})"
            )
            symbols_to_exit.add(symbol)
        elif signal in _RS_DECAY_SIGNALS and snapshots_by_symbol and symbol not in symbols_to_exit:
            entry_rs = meta.get("rs_rank_pct")
            current_snap = snapshots_by_symbol.get(symbol)
            current_rs = current_snap.get("rs_rank_pct") if current_snap else None
            if (
                entry_rs is not None
                and current_rs is not None
                and exit_optimiser.rs_decay_triggered(current_rs, entry_rs)
            ):
                logger.info(
                    f"RS decay exit: {symbol} rs_rank {entry_rs:.1f}% → {current_rs:.1f}% "
                    f"(>25pt drop) [{signal}]"
                )
                symbols_to_exit.add(symbol)
    return symbols_to_exit


def _handle_partial_exits(client, positions: list, atr_by_symbol: dict, dry_run: bool) -> list:
    """Sell half of any position that has reached the dynamic partial-exit threshold."""
    rc = RiskConfig.from_config()
    executed = []
    for pos in positions:
        symbol = pos["symbol"]
        meta = trader.get_position_meta(symbol)
        signal = meta.get("signal", "unknown")
        max_hold = config.SIGNAL_MAX_HOLD_DAYS.get(signal, config.MAX_HOLD_DAYS)
        days_held = 0  # partial exits evaluated before position_ages computed
        levels = exit_optimiser.compute_exit_levels(
            rc.stop_loss_pct,
            rc.take_profit_pct,
            atr_by_symbol.get(symbol),
            days_held,
            max_hold,
        )
        if pos["unrealized_plpc"] >= levels["partial_pct"]:
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
                        symbol,
                        "SELL_PARTIAL",
                        pos["market_value"] / 2,
                        result.broker_order_id or "",
                    )
                    audit_log.log_order_filled(
                        symbol, result.broker_order_id or "", result.filled_qty
                    )
                    trader.record_partial_exit(symbol)
                    trader.place_trailing_stop(
                        client, symbol, pos["qty"] - half_qty, current_price=pos["current_price"]
                    )
                    executed.append(
                        {
                            "symbol": symbol,
                            "action": "PARTIAL SELL",
                            "detail": f"50% at +{pos['unrealized_plpc']:.1f}%",
                            "decision_type": "rule_based",
                        }
                    )
            else:
                logger.info(f"  [DRY RUN] Would partial-sell {symbol}")
                executed.append(
                    {
                        "symbol": symbol,
                        "action": "PARTIAL SELL",
                        "detail": "dry run",
                        "decision_type": "rule_based",
                    }
                )
    return executed


def _execute_shorts(
    client,
    snapshots: list[dict],
    regime: dict,
    open_positions: list[dict],
    account_now: dict,
    all_trades: list,
    executed_symbols: set,
    dry_run: bool,
    _live_shadow: bool,
) -> None:
    """Scan for and execute short positions (bottom-quartile RS, rule-gated).

    - Max MAX_SHORT_POSITIONS concurrent shorts.
    - Total short notional capped at MAX_SHORT_HEDGE_RATIO × long notional.
    - Each position sized at SHORT_SIZE_SCALE × standard long size (whole shares only).
    - Regime gate: BULL_TREND or NEUTRAL_CHOP only.
    """
    regime_name = regime.get("regime", "UNKNOWN")

    # VIX term structure gate: only enter shorts when near-term fear (VIX9D) exceeds
    # medium-term fear (VIX) — inverted term structure signals higher short edge.
    # Default True (allow) when data unavailable to avoid accidentally blocking all shorts.
    vix_term_inverted: bool = regime.get("vix_term_inverted", True)
    if not vix_term_inverted:
        logger.info("VIX term structure not inverted (VIX9D/VIX ≤ 1.05) — skipping short entries")
        return

    held_symbols = {p["symbol"] for p in open_positions}
    short_candidates = stock_scanner.scan_short_candidates(snapshots, regime_name, held_symbols)
    if not short_candidates:
        return

    open_shorts = trader.get_open_shorts()
    short_slots = config.MAX_SHORT_POSITIONS - len(open_shorts)
    if short_slots <= 0:
        logger.info(f"Short slots full: {len(open_shorts)}/{config.MAX_SHORT_POSITIONS}")
        return

    long_notional = trader.get_long_notional(client)
    if long_notional == 0:
        logger.info("No long positions — skipping short scan (shorts are hedges only)")
        return
    short_notional = trader.get_short_notional(client)
    hedge_cap = long_notional * config.MAX_SHORT_HEDGE_RATIO
    if short_notional >= hedge_cap:
        logger.info(
            f"Short hedge cap reached: ${short_notional:.0f} >= ${hedge_cap:.0f} "
            f"({config.MAX_SHORT_HEDGE_RATIO:.0%} of long book)"
        )
        return

    logger.info(
        f"Short scan: {len(short_candidates)} candidates | "
        f"slots={short_slots} | hedge {short_notional:.0f}/{hedge_cap:.0f}"
    )

    shorts_placed = 0
    portfolio_value = account_now["portfolio_value"]
    for candidate in short_candidates:
        if shorts_placed >= short_slots:
            break
        symbol = candidate["symbol"]
        if symbol in executed_symbols:
            continue

        # Correlation gate — skip if correlated with any held position
        if correlation.correlated_with_held(symbol, held_symbols):
            logger.info(f"  Short skip {symbol}: correlated with existing position")
            continue

        # Squeeze risk gate — blocks crowded shorts and stocks in active squeezes
        _squeeze_info = short_risk.fetch_squeeze_info(symbol)
        _squeeze_blocked, _squeeze_reason = short_risk.is_squeeze_risk(
            symbol, candidate, **_squeeze_info
        )
        if _squeeze_blocked:
            logger.info(f"  Short skip {symbol}: squeeze risk — {_squeeze_reason}")
            continue

        current_price = candidate.get("current_price", 0.0)
        if not current_price:
            continue

        key_signal = candidate.get("key_signal", "rs_short")
        confidence = candidate.get("confidence", 0)
        matched_signals = candidate.get("matched_signals", [])

        # Size: SHORT_SIZE_SCALE × standard long notional → whole shares
        base_notional = position_sizer.risk_budget_size(
            portfolio_value,
            confidence=confidence,
            signal=key_signal,
            regime=regime_name,
        )
        target_notional = base_notional * config.SHORT_SIZE_SCALE
        qty_shares = int(math.floor(target_notional / current_price))
        if qty_shares < 1:
            logger.info(
                f"  Short skip {symbol}: ${target_notional:.0f} / ${current_price:.2f} = 0 shares"
            )
            continue

        # Hedge cap check per-order
        order_notional = qty_shares * current_price
        if short_notional + order_notional > hedge_cap:
            logger.info(f"  Short skip {symbol}: would breach hedge cap")
            continue

        signals_str = ", ".join(matched_signals) if matched_signals else key_signal
        logger.info(
            f"  SHORT {symbol}: {qty_shares} shares @ ~${current_price:.2f} "
            f"(${order_notional:.0f}) [{signals_str}] conf={confidence} "
            f"rs_rank={candidate.get('rs_rank_pct'):.1f}%"
        )

        reasoning = f"{signals_str} (RS rank {candidate.get('rs_rank_pct'):.1f}%) in {regime_name}"

        if not dry_run and not _live_shadow:
            short_result = trader.place_short_order(client, symbol, qty_shares)
            if short_result and short_result.broker_order_id:
                # Order reached the broker — consume the slot regardless of fill confirmation.
                # Timeout orders will be reconciled by _reconcile_late_fills on the next run.
                shorts_placed += 1
                short_notional += order_notional
            if short_result and short_result.is_success:
                entry_price = short_result.filled_avg_price or current_price
                trader.record_short(
                    symbol,
                    entry_price,
                    signal=key_signal,
                    regime=regime_name,
                    confidence=confidence,
                )
                audit_log.log_order_placed(
                    symbol, "SHORT", order_notional, short_result.broker_order_id or ""
                )
                if short_result.filled_qty:
                    cover_result = trader.place_short_cover_stop(
                        client, symbol, short_result.filled_qty
                    )
                    if cover_result is None or not cover_result.is_success:
                        logger.error(
                            f"  SHORT {symbol}: cover stop FAILED — closing position immediately"
                        )
                        trader.close_position(client, symbol)
                        trader.record_cover(symbol)
                        short_notional -= order_notional
                        continue
                executed_symbols.add(symbol)
                all_trades.append(
                    {
                        "symbol": symbol,
                        "action": "SHORT",
                        "detail": (
                            f"{qty_shares} shares @ ~${current_price:.2f} | "
                            f"{key_signal} | conf={confidence} | "
                            f"rs_rank={candidate.get('rs_rank_pct'):.1f}%"
                        ),
                        "decision_type": "short",
                        "confidence": confidence,
                        "key_signal": key_signal,
                        "reasoning": reasoning,
                    }
                )
        elif _live_shadow:
            audit_log.log_event(
                "WOULD_SHORT",
                {
                    "symbol": symbol,
                    "qty_shares": qty_shares,
                    "notional": round(order_notional, 2),
                    "rs_rank_pct": candidate.get("rs_rank_pct"),
                    "key_signal": key_signal,
                    "confidence": confidence,
                    "regime": regime_name,
                },
            )
            all_trades.append(
                {
                    "symbol": symbol,
                    "action": "WOULD_SHORT",
                    "detail": f"shadow {qty_shares} shares @ ~${current_price:.2f}",
                    "decision_type": "short",
                    "confidence": confidence,
                    "key_signal": key_signal,
                    "reasoning": reasoning,
                }
            )
            shorts_placed += 1
            executed_symbols.add(symbol)
        else:
            all_trades.append(
                {
                    "symbol": symbol,
                    "action": "SHORT",
                    "detail": f"dry run {qty_shares} shares",
                    "decision_type": "short",
                    "confidence": confidence,
                    "key_signal": key_signal,
                    "reasoning": reasoning,
                }
            )
            shorts_placed += 1
            executed_symbols.add(symbol)


# ── Pipeline phase helpers ────────────────────────────────────────────────────


def _get_position_snapshot(client) -> PositionSnapshot:
    """Fetch a point-in-time view of all broker positions and derived state."""
    open_positions = trader.get_open_positions(client)
    held_symbols = {p["symbol"] for p in open_positions}
    position_ages = trader.get_position_ages()
    open_shorts_db = trader.get_open_shorts()
    stale = [
        sym
        for sym, age in position_ages.items()
        if age
        >= (
            config.MAX_SHORT_HOLD_DAYS
            if sym in open_shorts_db
            else config.SIGNAL_MAX_HOLD_DAYS.get(
                trader.get_position_meta(sym).get("signal", "unknown"),
                config.MAX_HOLD_DAYS,
            )
        )
    ]
    earnings_risk = earnings_calendar.get_earnings_risk_positions(
        list(held_symbols), config.EARNINGS_WARNING_DAYS
    )
    atr_by_symbol = _fetch_atr_for_held(held_symbols)
    return PositionSnapshot(
        open_positions=open_positions,
        held_symbols=held_symbols,
        position_ages=position_ages,
        stale=stale,
        open_shorts_db=open_shorts_db,
        earnings_risk=earnings_risk,
        atr_by_symbol=atr_by_symbol,
    )


def _manage_existing_positions(
    client, dry_run: bool, all_trades: list, snap: PositionSnapshot, mode: str = "open"
) -> None:
    """Execute partial exits, earnings-risk exits, and profit-acceleration exits."""
    partials = _handle_partial_exits(client, snap.open_positions, snap.atr_by_symbol, dry_run)
    all_trades.extend(partials)

    # Refresh after partial exits before earnings checks
    post_partial_positions = trader.get_open_positions(client)
    post_partial_held = {p["symbol"] for p in post_partial_positions}
    position_ages = trader.get_position_ages()

    for symbol, ed in snap.earnings_risk.items():
        if symbol not in post_partial_held:
            continue
        logger.warning(f"Exiting {symbol} — earnings on {ed}")
        audit_log.log_earnings_exit(symbol, str(ed))
        if not dry_run:
            result = trader.close_position(client, symbol)
            if result.is_success:
                meta = trader.get_position_meta(symbol)
                pos = next((p for p in post_partial_positions if p["symbol"] == symbol), None)
                if pos:  # pragma: no branch
                    performance.record_trade_outcome(
                        meta["signal"],
                        pos["unrealized_plpc"],
                        regime=meta["regime"],
                        confidence=meta["confidence"],
                        sector=sector_data.get_sector(symbol),
                        hold_days=position_ages.get(symbol, 1),
                        symbol=symbol,
                        entry_date=meta.get("entry_date"),
                        entry_price=meta.get("entry_price"),
                        exit_reason="earnings_exit",
                    )
                    audit_log.log_position_closed(symbol, "earnings_exit", pos["unrealized_plpc"])
                trader.record_sell(symbol)
                all_trades.append(
                    {
                        "symbol": symbol,
                        "action": "SELL",
                        "detail": "earnings exit",
                        "decision_type": "rule_based",
                    }
                )
            else:
                logger.error(
                    f"  Earnings exit FAILED {symbol} — "
                    f"{result.rejection_reason or 'close failed after retries'}"
                )

    # Profit acceleration — fast exit for mean-reversion signals with rapid early gains.
    # Only evaluated in open and midday modes; close mode handles EOD exits separately.
    if mode in ("open", "midday"):
        for pos in post_partial_positions:
            symbol = pos["symbol"]
            meta = trader.get_position_meta(symbol)
            signal = meta.get("signal", "unknown")
            days_held = position_ages.get(symbol, 1)
            unrealised_pct = pos.get("unrealized_plpc", 0.0)
            action = exit_optimiser.profit_acceleration_triggered(unrealised_pct, days_held, signal)
            if action == "full_exit":
                logger.info(
                    f"Profit acceleration full exit: {symbol} +{unrealised_pct:.1f}% "
                    f"in {days_held}d [{signal}]"
                )
                if not dry_run:
                    result = trader.close_position(client, symbol)
                    if result.is_success:
                        performance.record_trade_outcome(
                            signal,
                            unrealised_pct,
                            regime=meta.get("regime", "UNKNOWN"),
                            confidence=meta.get("confidence", 0),
                            sector=sector_data.get_sector(symbol),
                            hold_days=days_held,
                            symbol=symbol,
                            entry_date=meta.get("entry_date"),
                            entry_price=meta.get("entry_price"),
                            exit_reason="profit_acceleration",
                        )
                        audit_log.log_position_closed(symbol, "profit_acceleration", unrealised_pct)
                        trader.record_sell(symbol)
                        all_trades.append(
                            {
                                "symbol": symbol,
                                "action": "SELL",
                                "detail": f"profit acceleration +{unrealised_pct:.1f}% in {days_held}d",
                                "decision_type": "rule_based",
                            }
                        )
                    else:
                        logger.error(
                            f"  Profit acceleration SELL FAILED {symbol} — "
                            f"{result.rejection_reason or 'close failed'}"
                        )
                else:
                    all_trades.append(
                        {
                            "symbol": symbol,
                            "action": "SELL",
                            "detail": f"profit acceleration +{unrealised_pct:.1f}% in {days_held}d [dry run]",
                            "decision_type": "rule_based",
                        }
                    )
            elif action == "partial_exit":
                if meta.get("partial_exit_taken_at"):
                    logger.info(f"  Partial already taken for {symbol} — skipping profit accel")
                    continue
                logger.info(
                    f"Profit acceleration partial exit: {symbol} +{unrealised_pct:.1f}% "
                    f"in {days_held}d [{signal}]"
                )
                if not dry_run:
                    half_qty = pos["qty"] / 2
                    result = trader.place_partial_sell(client, symbol, half_qty)
                    if result and result.is_success:
                        audit_log.log_order_placed(
                            symbol,
                            "SELL_PARTIAL",
                            pos["market_value"] / 2,
                            result.broker_order_id or "",
                        )
                        trader.record_partial_exit(symbol)
                        all_trades.append(
                            {
                                "symbol": symbol,
                                "action": "PARTIAL SELL",
                                "detail": f"profit acceleration 50% at +{unrealised_pct:.1f}%",
                                "decision_type": "rule_based",
                            }
                        )
                else:
                    all_trades.append(
                        {
                            "symbol": symbol,
                            "action": "PARTIAL SELL",
                            "detail": f"profit acceleration 50% at +{unrealised_pct:.1f}% [dry run]",
                            "decision_type": "rule_based",
                        }
                    )


def _fetch_market_context() -> MarketContext:
    """Fetch market-wide context: VIX, regime, macro, sector, lessons, cross-asset, sentiment."""
    logger.info("Fetching market context...")

    def _vix_and_regime() -> tuple:
        v = market_data.get_vix()
        r = stock_scanner.get_market_regime(config.BEAR_MARKET_SPY_THRESHOLD, vix=v)
        return v, r

    with ThreadPoolExecutor(max_workers=7) as ex:
        fut_vr = ex.submit(_vix_and_regime)
        fut_macro = ex.submit(macro_calendar.get_macro_risk)
        fut_cross_asset = ex.submit(get_macro_snapshot)
        fut_sentiment = ex.submit(get_sentiment_snapshot)
        fut_sector = ex.submit(sector_data.get_sector_performance)
        fut_leading = ex.submit(sector_data.get_leading_sectors, top_n=3)
        fut_lessons = ex.submit(get_latest_review)
        vix, regime = fut_vr.result()
        macro = fut_macro.result()
        cross_asset = fut_cross_asset.result()
        sentiment_snap = fut_sentiment.result()
        sector_perf = fut_sector.result()
        leading_sectors = fut_leading.result()
        lessons = fut_lessons.result()

    if vix:
        logger.info(f"VIX: {vix}  Regime: {regime.get('regime', 'UNKNOWN')}")
    if macro["is_high_risk"]:
        logger.warning(f"Macro risk: {macro['event']}")
        audit_log.log_macro_skip(macro["event"])
    if cross_asset.data_available:
        logger.info(
            "Cross-asset macro: credit_stress=%s duration_flight=%s copper_gold=%s usd_strong=%s",
            cross_asset.credit_stress,
            cross_asset.duration_flight,
            cross_asset.copper_gold_positive,
            cross_asset.usd_strong,
        )
    if sentiment_snap.fear_greed is not None:
        logger.info(
            "Fear & Greed: %.0f (%s) — contrarian_long=%s contrarian_short=%s",
            sentiment_snap.fear_greed.score,
            sentiment_snap.fear_greed.label,
            sentiment_snap.contrarian_long_signal,
            sentiment_snap.contrarian_short_signal,
        )
    from dataclasses import asdict

    return MarketContext(
        vix=vix,
        regime=regime,
        macro=macro,
        sector_perf=sector_perf,
        leading_sectors=leading_sectors,
        lessons=lessons,
        cross_asset_macro=asdict(cross_asset),
        sentiment_snapshot=asdict(sentiment_snap),
    )


def _build_data_bundle(
    client, snap: PositionSnapshot, mc: MarketContext, mode: str = "open"
) -> DataBundle | None:
    """Fetch, enrich, and pre-filter market data. Returns None if no snapshots available."""
    logger.info("Scanning for top movers...")
    top_movers = stock_scanner.get_top_movers(config.TOP_MOVERS_COUNT)
    scan_symbols = list(set(build_scan_universe(client)) | snap.held_symbols | set(top_movers))
    logger.info(f"Scanning {len(scan_symbols)} symbols")

    logger.info("Fetching market data...")
    snapshots = market_data.get_market_snapshots(scan_symbols, config.LOOKBACK_DAYS)
    if not snapshots:
        logger.error("No market data. Aborting.")
        return None

    # ── Intraday enrichment (VWAP, ORB, gap, intraday momentum) ─────────────
    intraday = market_data.get_intraday_data([s["symbol"] for s in snapshots])
    if intraday:  # pragma: no branch
        for s in snapshots:  # pragma: no cover
            if s["symbol"] in intraday:
                s.update(intraday[s["symbol"]])

    # ── Insider activity (SEC EDGAR Form 4 — open-market cluster purchases) ──
    logger.info("Fetching insider activity...")
    insider_data = insider_feed.get_insider_activity([s["symbol"] for s in snapshots])
    for s in snapshots:
        if s["symbol"] in insider_data:
            s.update(insider_data[s["symbol"]])

    # ── News sentiment (Alpha Vantage structured scores) ─────────────────────
    logger.info("Fetching AV news sentiment...")
    av_data = av_sentiment.get_av_sentiment([s["symbol"] for s in snapshots])
    for s in snapshots:
        if s["symbol"] in av_data:
            s.update(av_data[s["symbol"]])

    # ── PEAD (Post-Earnings Announcement Drift) candidates ───────────────────
    logger.info("Fetching earnings surprise data...")
    pead_data = earnings_surprise.get_earnings_surprise([s["symbol"] for s in snapshots])
    for s in snapshots:
        if s["symbol"] in pead_data:
            s.update(pead_data[s["symbol"]])

    # ── Negative PEAD (earnings miss) — bearish short signal ─────────────────
    logger.info("Fetching earnings miss data...")
    miss_data = earnings_surprise.get_earnings_miss([s["symbol"] for s in snapshots])
    for s in snapshots:
        if s["symbol"] in miss_data:
            s.update(miss_data[s["symbol"]])

    # ── Short interest (live-only — not backtestable) ─────────────────────────
    logger.info("Fetching short interest data...")
    si_data = short_interest.get_short_interest([s["symbol"] for s in snapshots])
    for s in snapshots:
        if s["symbol"] in si_data:
            s.update(si_data[s["symbol"]])

    # ── EDGAR corporate events (guidance, activist, secondary offering) ────────
    # Must run before prefilter so guidance_positive and activist_filing appear
    # in matched_signals. Cache warmed at 07:00 ET; this is a single JSON read.
    logger.info("Fetching EDGAR corporate event signals...")
    edgar_batch = edgar_client.get_edgar_signals_batch([s["symbol"] for s in snapshots])
    for s in snapshots:
        entry = edgar_batch.get(s["symbol"], {})
        guidance = entry.get("guidance") or {}
        activist = entry.get("activist") or {}
        offering = entry.get("secondary_offering") or {}
        s["guidance_positive"] = bool(guidance.get("guidance_positive", False))
        s["guidance_negative"] = bool(guidance.get("guidance_negative", False))
        s["activist_filing"] = bool((activist or {}).get("known_activist", False))
        s["secondary_offering"] = bool((offering or {}).get("offering_detected", False))

    # ── Pre-filter buy candidates ─────────────────────────────────────────────
    held_snaps = [s for s in snapshots if s["symbol"] in snap.held_symbols]
    candidate_snaps = [s for s in snapshots if s["symbol"] not in snap.held_symbols]

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

    filtered_candidates = stock_scanner.prefilter_candidates(
        candidate_snaps, regime=mc.regime.get("regime")
    )
    # In exit-only modes (close, open_sells) buys are never executed — skip sending
    # the full candidate list to Claude; only held positions need AI exit evaluation.
    if mode in ("close", "open_sells"):
        ai_snapshots = held_snaps
    else:
        ai_snapshots = held_snaps + filtered_candidates
    logger.info(
        f"Pre-filter: {len(candidate_snaps)} candidates → {len(filtered_candidates)} passed"
    )
    _filtered_syms = {c["symbol"] for c in filtered_candidates}
    audit_log.log_event(
        "PREFILTER_CANDIDATES",
        {
            "total_candidates": len(candidate_snaps),
            "passed": len(filtered_candidates),
            "candidates": [
                {
                    "symbol": c["symbol"],
                    "matched_signals": c.get("matched_signals", []),
                    "rsi_14": c.get("rsi_14"),
                    "vol_ratio": c.get("vol_ratio"),
                    "ret_5d_pct": c.get("ret_5d_pct"),
                }
                for c in filtered_candidates
            ],
            "rejected_symbols": [
                s["symbol"] for s in candidate_snaps if s["symbol"] not in _filtered_syms
            ],
        },
    )

    # Symbols the AI actually received — used as the validation universe.
    # Broader known_symbols (all fetched) would allow hallucinated tickers from
    # top_movers that never made it through the pre-filter.
    ai_known_symbols = {s["symbol"] for s in ai_snapshots}

    # ── Options flow (basic put/call from options_scanner) ───────────────────
    options_syms = [s["symbol"] for s in filtered_candidates]
    options_sigs = options_scanner.get_options_signals(options_syms) if options_syms else {}
    if options_sigs:
        logger.info(f"Options signals fetched for: {list(options_sigs.keys())}")

    # ── Options IV surface (richer signals from options_data with daily cache) ─
    # Run after prefilter — only enriches filtered candidates to keep latency low.
    # iv_cheap, iv_expensive, unusual_call_oi, panic_put_skew, call_skew_spike.
    if options_syms:
        logger.info("Fetching options IV surface for %d candidates...", len(options_syms))
        iv_batch = options_data_module.get_options_batch(options_syms)
        for s in filtered_candidates:
            iv_snap = iv_batch.get(s["symbol"])
            if iv_snap is not None:
                s["iv_cheap"] = iv_snap.iv_cheap
                s["iv_expensive"] = iv_snap.iv_expensive
                s["unusual_call_oi"] = iv_snap.unusual_call_oi
                s["panic_put_skew"] = iv_snap.panic_put_skew
                s["call_skew_spike"] = iv_snap.call_skew_spike

    # ── News (sanitized against prompt injection) ─────────────────────────────
    # Restrict to symbols the AI actually received snapshots for — sending news
    # for prefilter-rejected top movers causes the AI to recommend them despite
    # having no snapshot data, producing validator rejections on every run.
    logger.info("Fetching news and sentiment...")
    news_symbols = list(ai_known_symbols)
    raw_news = news_fetcher.fetch_news(news_symbols)
    news = sanitize_headlines(raw_news)

    sent = sentiment_module.get_sentiment(list(snap.held_symbols) + top_movers[:10])

    # ── Short universe (expanded beyond the long scan) ───────────────────────
    # Alpaca easy-to-borrow list or static fallback; RS ranks computed cross-sectionally.
    logger.info("Building short universe...")
    short_syms = get_short_universe(client)
    short_snapshots = scan_short_universe(short_syms)
    si_short = short_interest.get_short_interest([s["symbol"] for s in short_snapshots])
    for s in short_snapshots:
        if s["symbol"] in si_short:
            s.update(si_short[s["symbol"]])
    logger.info(f"Short universe: {len(short_snapshots)} snapshots enriched")

    return DataBundle(
        snapshots=snapshots,
        ai_snapshots=ai_snapshots,
        filtered_candidates=filtered_candidates,
        options_sigs=options_sigs,
        news=news,
        sentiment=sent,
        short_snapshots=short_snapshots,
    )


def _run_ai_phase(
    db: DataBundle,
    snap: PositionSnapshot,
    mc: MarketContext,
    account_now: dict,
    run_id: str,
    mode: str,
    executed_symbols: set,
) -> dict | None:
    """Run AI analysis, validate, log decisions. Returns the decisions dict or None on failure."""
    if not db.ai_snapshots:
        logger.info("AI analysis skipped — no positions to evaluate.")
        return {"buy_candidates": [], "position_decisions": [], "market_summary": "", "date": ""}

    track_record = portfolio_tracker.get_track_record(10)
    logger.info("Running AI analysis...")
    decisions = ai_analyst.get_trading_decisions(
        snapshots=db.ai_snapshots,
        current_positions=snap.open_positions,
        available_cash=account_now["cash"],
        portfolio_value=account_now["portfolio_value"],
        news_by_symbol=db.news,
        track_record=track_record,
        market_regime=mc.regime,
        position_ages=snap.position_ages,
        stale_positions=snap.stale,
        vix=mc.vix,
        sector_performance=mc.sector_perf,
        sentiment=db.sentiment,
        earnings_risk={sym: str(ed) for sym, ed in snap.earnings_risk.items()},
        macro_risk=mc.macro,
        leading_sectors=mc.leading_sectors,
        options_signals=db.options_sigs,
        lessons=mc.lessons,
        run_id=run_id,
    )

    if not decisions:
        logger.error("AI analysis failed. Aborting.")
        return None

    # ── Validate AI response before acting ───────────────────────────────────
    ai_known_symbols = {s["symbol"] for s in db.ai_snapshots}
    is_valid, validation_errors = validate_ai_response(
        decisions, ai_known_symbols, held_symbols=snap.held_symbols
    )
    if not is_valid:
        audit_log.log_validation_failure(validation_errors)
        buy_domain_only = validation_errors and all(
            e.startswith("BUY candidate '") or e.startswith("buy_candidates")
            for e in validation_errors
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
    _selected_syms = {b["symbol"] for b in decisions.get("buy_candidates", [])}
    _ranked = sorted(db.filtered_candidates, key=stock_scanner.score_candidate, reverse=True)
    audit_log.log_event(
        "CANDIDATE_SELECTION",
        {
            "prefiltered_count": len(db.filtered_candidates),
            "claude_buy_count": len(decisions.get("buy_candidates", [])),
            "selected": [
                {
                    "symbol": b["symbol"],
                    "confidence": b.get("confidence"),
                    "deterministic_rank": next(
                        (i + 1 for i, c in enumerate(_ranked) if c["symbol"] == b["symbol"]),
                        None,
                    ),
                }
                for b in decisions.get("buy_candidates", [])
            ],
            "not_selected": [
                {
                    "symbol": c["symbol"],
                    "deterministic_score": stock_scanner.score_candidate(c),
                    "matched_signals": c.get("matched_signals", []),
                }
                for c in db.filtered_candidates
                if c["symbol"] not in _selected_syms
            ],
        },
    )
    return decisions


def _execute_sell_phase(
    client,
    snap: PositionSnapshot,
    decisions: dict,
    all_trades: list,
    executed_symbols: set,
    dry_run: bool,
    _live_shadow: bool,
    snapshots: list | None = None,
    regime_name: str = "UNKNOWN",
) -> None:
    """Execute AI sell decisions, stale-long exits, rule-based stops, and stale short covers."""
    snap_by_symbol = {s["symbol"]: s for s in (snapshots or [])}

    # Adverse volume — two consecutive institutional-selling days trigger exit
    long_held = snap.held_symbols - snap.open_shorts_db
    vol_hist = _fetch_adverse_vol_for_held(long_held)

    symbols_to_sell = {
        d["symbol"] for d in decisions.get("position_decisions", []) if d["action"] == "SELL"
    }
    symbols_to_sell |= {
        sym for sym in snap.stale if sym in snap.held_symbols and sym not in snap.open_shorts_db
    }
    symbols_to_sell |= _check_rule_based_stops(
        snap.open_positions,
        snap.position_ages,
        snap.atr_by_symbol,
        snapshots_by_symbol=snap_by_symbol,
    )

    for pos in snap.open_positions:
        symbol = pos["symbol"]
        if symbol in snap.open_shorts_db or symbol in symbols_to_sell:
            continue
        vh = vol_hist.get(symbol, {})
        if exit_optimiser.adverse_volume_triggered(
            vh.get("vol_ratio_today", 0.0),
            vh.get("ret_today", 0.0),
            vh.get("vol_ratio_yday", 0.0),
            vh.get("ret_yday", 0.0),
        ):
            logger.info(
                f"Adverse volume exit: {symbol} — "
                f"vol×{vh['vol_ratio_today']:.1f}/{vh['vol_ratio_yday']:.1f} "
                f"ret {vh['ret_today']:.1f}%/{vh['ret_yday']:.1f}%"
            )
            symbols_to_sell.add(symbol)

    # Regime-change exit — close new longs immediately on DEFENSIVE/BEAR downgrade
    if regime_name in _DEFENSIVE_REGIMES:
        for pos in snap.open_positions:
            symbol = pos["symbol"]
            if symbol in snap.open_shorts_db or symbol in symbols_to_sell:
                continue
            days_held = snap.position_ages.get(symbol, 1)
            if days_held < 2:
                logger.info(
                    f"Regime-change exit: {symbol} — {regime_name} with {days_held} day(s) held"
                )
                symbols_to_sell.add(symbol)
            elif days_held >= 3:
                logger.info(
                    f"Regime-change stop advisory: {symbol} — {regime_name}, "
                    f"{days_held} days held (manual stop tighten to 2% recommended)"
                )

    # Exclude any short positions from the sell set (handled separately below)
    symbols_to_sell -= snap.open_shorts_db

    symbols_to_cover = {sym for sym in snap.stale if sym in snap.open_shorts_db}

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
            pos = next((p for p in snap.open_positions if p["symbol"] == symbol), None)
            result = trader.close_position(client, symbol)
            if result.is_success:
                if pos:
                    meta = trader.get_position_meta(symbol)
                    performance.record_trade_outcome(
                        meta["signal"],
                        pos["unrealized_plpc"],
                        regime=meta["regime"],
                        confidence=meta["confidence"],
                        sector=sector_data.get_sector(symbol),
                        hold_days=snap.position_ages.get(symbol, 1),
                        symbol=symbol,
                        entry_date=meta.get("entry_date"),
                        entry_price=meta.get("entry_price"),
                        exit_reason="ai_sell" if decision else "time_exit",
                    )
                    audit_log.log_position_closed(symbol, reason[:50], pos["unrealized_plpc"])
                trader.record_sell(symbol)
                executed_symbols.add(symbol)
                all_trades.append(
                    {
                        "symbol": symbol,
                        "action": "SELL",
                        "detail": reason,
                        "decision_type": "sell" if decision else "rule_based",
                        "confidence": decision.get("confidence") if decision else None,
                        "reasoning": decision.get("reasoning", "") if decision else reason,
                    }
                )
            else:
                fail_detail = result.rejection_reason or "close failed after retries"
                logger.error(f"  SELL FAILED {symbol} — {fail_detail}. Manual review required.")
                alerts.alert_error("SELL FAILED", f"{symbol}: {fail_detail}")
        else:
            if _live_shadow:
                audit_log.log_event("WOULD_SELL", {"symbol": symbol, "reason": reason[:80]})
                all_trades.append(
                    {
                        "symbol": symbol,
                        "action": "WOULD_SELL",
                        "detail": reason,
                        "decision_type": "sell" if decision else "rule_based",
                        "confidence": decision.get("confidence") if decision else None,
                        "reasoning": decision.get("reasoning", "") if decision else reason,
                    }
                )
            else:
                all_trades.append(
                    {
                        "symbol": symbol,
                        "action": "SELL",
                        "detail": "dry run",
                        "decision_type": "sell" if decision else "rule_based",
                        "confidence": decision.get("confidence") if decision else None,
                        "reasoning": decision.get("reasoning", "") if decision else reason,
                    }
                )
            executed_symbols.add(symbol)

    # ── Cover stale short positions ───────────────────────────────────────────
    for symbol in symbols_to_cover:
        reason = f"Time-based cover (≥{config.MAX_SHORT_HOLD_DAYS} days short)"
        logger.info(f"  COVER {symbol} — {reason}")
        if not dry_run:
            pos = next((p for p in snap.open_positions if p["symbol"] == symbol), None)
            result = trader.close_position(client, symbol)
            if result.is_success:
                audit_log.log_position_closed(
                    symbol, reason[:50], -(pos["unrealized_plpc"]) if pos else 0.0
                )
                trader.record_cover(symbol)
                executed_symbols.add(symbol)
                all_trades.append(
                    {
                        "symbol": symbol,
                        "action": "COVER",
                        "detail": reason,
                        "decision_type": "rule_based",
                        "confidence": None,
                        "reasoning": reason,
                    }
                )
            else:
                fail_detail = result.rejection_reason or "cover failed"
                logger.error(f"  COVER FAILED {symbol} — {fail_detail}. Manual review required.")
                alerts.alert_error("COVER FAILED", f"{symbol}: {fail_detail}")
        else:
            all_trades.append(
                {
                    "symbol": symbol,
                    "action": "COVER",
                    "detail": "dry run",
                    "decision_type": "rule_based",
                    "confidence": None,
                    "reasoning": reason,
                }
            )
            executed_symbols.add(symbol)


def _execute_buy_phase(
    client,
    snap: PositionSnapshot,
    db: DataBundle,
    mc: MarketContext,
    decisions: dict,
    health_status: HealthStatus,
    cb_triggered: bool,
    _exp_drawdown_triggered: bool,
    _dd_scalar: float,
    account_now: dict,
    should_run_live_gates: bool,
    today: str,
    mode: str,
    dry_run: bool,
    _live_shadow: bool,
    all_trades: list,
    executed_symbols: set,
) -> tuple[list, dict]:
    """Execute the buy loop. Returns (open_positions, account_now) for use by execute_shorts."""
    # Initialised from the post-exit snapshot; overwritten with a fresher fetch
    # inside the else branch when buys are not skipped.
    open_positions = snap.open_positions

    # Same-day open guard: only one buy phase per calendar day in open mode.
    # Prevents duplicate buys when the bot is restarted mid-session or triggered
    # more than once by launchd/manual invocation.
    if mode == "open" and not dry_run and audit_log.has_open_buys_run_today(today):
        logger.warning(
            "Same-day open guard: open buys already executed today — skipping buy phase."
        )
        return open_positions, account_now

    daily_notional_spent = trader.get_daily_notional(today)  # persisted across runs on same date
    skip_buys = (
        mode in ("close", "open_sells")
        or cb_triggered
        or mc.regime.get("is_bearish")
        or mc.macro.get("is_high_risk")
        or health_status in (HealthStatus.RED, HealthStatus.YELLOW)
        or _exp_drawdown_triggered
    )
    if skip_buys:
        reasons = []
        if mode == "close":
            reasons.append("close mode")
        if cb_triggered:
            reasons.append("circuit breaker")
        if mc.regime.get("is_bearish"):
            reasons.append("bear market filter")
        if mc.macro.get("is_high_risk"):
            reasons.append(f"macro event: {mc.macro.get('event')}")
        if health_status in (HealthStatus.RED, HealthStatus.YELLOW):
            reasons.append(f"startup health {health_status}")
        if _exp_drawdown_triggered:
            reasons.append("experiment drawdown cap reached")
        logger.warning(f"Skipping new buys: {', '.join(reasons)}")
    else:
        if mode == "open" and not dry_run:
            audit_log.log_open_buys_locked(today)
        account_now = trader.get_account_info(client)
        open_positions = trader.get_open_positions(client)
        long_positions = [p for p in open_positions if p.get("qty", 0) > 0]
        available_cash = account_now["cash"] * (1 - config.CASH_RESERVE_PCT)
        max_positions = min(
            position_sizer.get_max_positions(account_now["portfolio_value"]),
            config.MAX_POSITIONS,
        )
        slots = max_positions - len(long_positions)
        logger.info(f"Position slots: {len(long_positions)}/{max_positions}")

        if slots > 0:
            regime_name = mc.regime.get("regime", "UNKNOWN")
            regime_policy = get_regime_policy(regime_name)
            effective_max_orders = min(config.MAX_ORDERS_PER_RUN, regime_policy.max_orders_per_run)
            regime_conf_bump = regime_policy.min_confidence_bump

            min_confidence = (
                config.MIN_CONFIDENCE + regime_conf_bump + (1 if mc.vix and mc.vix > 25 else 0)
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
            vix_trail_pct = risk_manager.check_vix_stop_adjustment(mc.vix)
            if abs(vix_trail_pct - config.TRAILING_STOP_PCT) > 0.01:
                logger.info(
                    f"VIX-adjusted trail: {config.TRAILING_STOP_PCT}% → {vix_trail_pct}% "
                    f"(VIX={mc.vix})"
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
                key_signal = candidate.get("key_signal", "unknown")
                n_signals = len(candidate.get("matched_signals") or [key_signal])
                sig_multiplier = position_sizer.get_signal_size_multiplier(key_signal)
                cofire_multiplier = position_sizer.cofiring_boost(n_signals)

                # Correlation filter — skip if returns too closely track an existing position
                if correlation.correlated_with_held(symbol, {p["symbol"] for p in open_positions}):
                    logger.info(f"Skipping {symbol}: correlated with an existing position")
                    continue

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
                    # ATR-based equal-risk sizing when available; falls back to risk_budget_size
                    _cand_atr = exit_optimiser.compute_atr_pct(symbol)
                    if _cand_atr is not None:
                        _base = position_sizer.atr_position_size(
                            account_now["portfolio_value"], _cand_atr
                        )
                        logger.debug(f"  ATR sizing {symbol}: atr={_cand_atr:.2f}% → ${_base:.2f}")
                    else:
                        _base = position_sizer.risk_budget_size(
                            account_now["portfolio_value"],
                            confidence,
                            signal=key_signal,
                            regime=mc.regime.get("regime", "UNKNOWN"),
                        )
                    notional = min(
                        _base * _dd_scalar * sig_multiplier * cofire_multiplier,
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

                logger.info(
                    f"  BUY {symbol}: ${notional:.2f} | conf={confidence} | "
                    f"sig×{sig_multiplier:.1f} cofire×{cofire_multiplier:.1f}"
                )

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

                sym_snap = next(
                    (s for s in db.snapshots if s["symbol"] == symbol),
                    None,  # type: ignore[arg-type]
                )
                if notional >= 1.0:
                    # Guard: skip if notional buys < 1 whole share — Alpaca cannot stop-protect sub-share positions
                    if sym_snap and notional / sym_snap["current_price"] < 1.0:
                        logger.warning(
                            f"  Skipping {symbol}: ${notional:.2f} at ${sym_snap['current_price']:.2f}"
                            f" = {notional / sym_snap['current_price']:.3f} shares — sub-share position cannot be stop-protected"
                        )
                        continue
                    if not dry_run:
                        t_buy_submit = time.monotonic()
                        buy_result = trader.place_buy_order(client, symbol, notional)
                        t_fill = time.monotonic()
                        if buy_result and buy_result.is_success:
                            fill_latency_ms = round((t_fill - t_buy_submit) * 1000)
                            orders_placed += 1
                            daily_notional_spent += notional
                            trader.add_daily_notional(today, notional)
                            entry_price = sym_snap["current_price"] if sym_snap else 0.0
                            _signal = candidate.get("key_signal", "unknown")
                            trader.record_buy(
                                symbol,
                                entry_price,
                                signal=_signal,
                                regime=mc.regime.get("regime", "UNKNOWN"),
                                confidence=confidence,
                                track="intraday" if _signal in INTRADAY_SIGNALS else "multiday",
                                rs_rank_pct=candidate.get("rs_rank_pct"),
                            )
                            audit_log.log_order_placed(
                                symbol, "BUY", notional, buy_result.broker_order_id or ""
                            )
                            if buy_result.filled_qty:
                                audit_log.log_order_filled(
                                    symbol, buy_result.broker_order_id or "", buy_result.filled_qty
                                )
                                current_price = sym_snap["current_price"] if sym_snap else None
                                # Floor to whole shares — Alpaca rejects fractional stop orders
                                stop_qty = int(math.floor(buy_result.filled_qty))
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
                                fill_avg = buy_result.filled_avg_price
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
                            all_trades.append(
                                {
                                    "symbol": symbol,
                                    "action": "BUY",
                                    "detail": detail,
                                    "decision_type": "buy",
                                    "confidence": confidence,
                                    "key_signal": candidate.get("key_signal"),
                                    "reasoning": candidate.get("reasoning", ""),
                                }
                            )
                        elif buy_result and buy_result.status in (
                            OrderStatus.PARTIAL,
                            OrderStatus.TIMEOUT,
                        ):
                            # Ambiguous fill — broker may hold exposure we didn't fully record.
                            # Re-query immediately and attach stops; don't wait until next startup.
                            logger.warning(
                                f"  BUY {symbol}: ambiguous result "
                                f"status={buy_result.status.name} "
                                f"filled_qty={buy_result.filled_qty} — "
                                "running immediate stop coverage check"
                            )
                            audit_log.log_event(
                                "ORDER_AMBIGUOUS",
                                {
                                    "symbol": symbol,
                                    "status": buy_result.status.name,
                                    "filled_qty": buy_result.filled_qty,
                                    "broker_order_id": buy_result.broker_order_id,
                                },
                            )
                            alerts.alert_error(
                                "ORDER AMBIGUOUS",
                                f"{symbol} buy returned {buy_result.status.name} — "
                                "checking stop coverage now.",
                            )
                            immediate_stops_ok = trader.ensure_stops_attached(client)
                            if not immediate_stops_ok:
                                _handle_stop_failure(client, symbol, dry_run)
                    else:
                        if _live_shadow:
                            audit_log.log_event(
                                "WOULD_BUY",
                                {
                                    "symbol": symbol,
                                    "notional": round(notional, 2),
                                    "confidence": confidence,
                                    "signal": candidate.get("key_signal"),
                                    "regime": mc.regime.get("regime"),
                                    "sizing": "small_account"
                                    if config.SMALL_ACCOUNT_MODE
                                    else "risk_budget",
                                },
                            )
                            all_trades.append(
                                {
                                    "symbol": symbol,
                                    "action": "WOULD_BUY",
                                    "detail": f"shadow ${notional:.2f} | {candidate.get('key_signal')} | conf={confidence}",
                                    "decision_type": "buy",
                                    "confidence": confidence,
                                    "key_signal": candidate.get("key_signal"),
                                    "reasoning": candidate.get("reasoning", ""),
                                }
                            )
                        else:
                            all_trades.append(
                                {
                                    "symbol": symbol,
                                    "action": "BUY",
                                    "detail": f"dry run ${notional:.2f}",
                                    "decision_type": "buy",
                                    "confidence": confidence,
                                    "key_signal": candidate.get("key_signal"),
                                    "reasoning": candidate.get("reasoning", ""),
                                }
                            )
                        orders_placed += 1
                        daily_notional_spent += notional
                        executed_symbols.add(symbol)
                else:
                    logger.warning(f"  Skipping {symbol}: ${notional:.2f} too small")

    return open_positions, account_now


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

    global _current_lock_fd
    _current_lock_fd = lock_fd
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGHUP, _sigterm_handler)

    try:
        _run_inner(dry_run=dry_run, mode=mode, today=today, _live_shadow=_live_shadow)
    except Exception as e:
        logger.error(f"Unhandled error in trading run: {e}", exc_info=True)
        alerts.alert_error("main.run", str(e))
    finally:
        _release_lock(lock_fd)
        _current_lock_fd = None


def _force_cover_intraday_positions(client, dry_run: bool, all_trades: list) -> None:
    """Market-sell / cover all positions tagged track='intraday' before close."""
    symbols = trader.get_intraday_positions()
    if not symbols:
        return
    logger.info(f"Force-covering {len(symbols)} intraday position(s): {symbols}")
    for symbol in symbols:
        try:
            meta = trader.get_position_meta(symbol)
            is_short = meta.get("side", "long") == "short"
            action_label = "COVER" if is_short else "SELL"
            if not dry_run:
                result = trader.close_position(client, symbol)
                if result and result.is_success:
                    if is_short:
                        trader.record_cover(symbol)
                    else:
                        trader.record_sell(symbol)
                    all_trades.append(
                        {
                            "symbol": symbol,
                            "action": action_label,
                            "detail": "intraday force-cover at close",
                        }
                    )
                    audit_log.log_event(
                        "INTRADAY_FORCE_COVER",
                        {"symbol": symbol, "side": "short" if is_short else "long"},
                    )
                else:
                    logger.error(f"Force-cover failed for {symbol}: {result}")
            else:
                logger.info(f"[dry-run] Would force-cover intraday position: {symbol}")
                all_trades.append(
                    {
                        "symbol": symbol,
                        "action": action_label,
                        "detail": "intraday force-cover at close [dry-run]",
                    }
                )
        except Exception as e:
            logger.error(f"Force-cover error for {symbol}: {e}")


def _reconcile_late_fills(
    client,
    today: str,
    mode: str,
    mc: MarketContext,
    decisions: dict,
    executed_symbols: set,
    all_trades: list,
    dry_run: bool,
) -> None:
    """Attach missing stops and reconcile buy orders that filled after wait_for_fill timed out."""
    if not dry_run:
        trader.ensure_stops_attached(client)

    if not dry_run:
        try:
            from utils.order_ledger import (
                get_unresolved_intents,
            )
            from utils.order_ledger import (
                log_order_event as _log_oe,
            )
            from utils.order_ledger import (
                update_intent as _update_intent,
            )

            timeout_intents = [
                i for i in get_unresolved_intents(trade_date=today) if i["status"] == "timeout"
            ]
            if timeout_intents:
                current_live = trader.get_open_positions(client)
                live_pos_map = {p["symbol"]: p for p in current_live}
                for intent in timeout_intents:
                    sym = intent["symbol"]
                    if sym not in live_pos_map:
                        continue
                    if sym in executed_symbols:
                        continue
                    pos = live_pos_map[sym]
                    entry_price = pos["avg_entry_price"]
                    is_short = intent.get("side", "BUY") == "SHORT"
                    if is_short:
                        stored = trader.get_position_meta(sym)
                        signal = stored.get("signal") or "rs_short"
                        confidence = stored.get("confidence", 0)
                    else:
                        candidate = next(
                            (c for c in decisions.get("buy_candidates", []) if c["symbol"] == sym),
                            {},
                        )
                        signal = candidate.get("key_signal") or trader.get_position_meta(sym).get(
                            "signal", "unknown"
                        )
                        confidence = candidate.get("confidence") or trader.get_position_meta(
                            sym
                        ).get("confidence", 0)
                    action_label = "SHORT" if is_short else "BUY"
                    logger.info(
                        f"Late-fill reconciliation: {sym} found in broker positions "
                        f"@ ${entry_price:.4f} — recording {action_label} (mode={mode})"
                    )
                    _update_intent(intent["client_order_id"], "filled")
                    _log_oe(
                        intent["client_order_id"],
                        "ORDER_LATE_FILL_RECONCILED",
                        {"entry_price": round(entry_price, 4), "run_mode": mode},
                        broker_order_id=intent.get("broker_order_id"),
                    )
                    if is_short:
                        trader.record_short(
                            sym,
                            entry_price,
                            signal=signal,
                            regime=mc.regime.get("regime", "UNKNOWN"),
                            confidence=confidence,
                            track="intraday" if signal in INTRADAY_SHORT_SIGNALS else "multiday",
                        )
                    else:
                        trader.record_buy(
                            sym,
                            entry_price,
                            signal=signal,
                            regime=mc.regime.get("regime", "UNKNOWN"),
                            confidence=confidence,
                            track="intraday" if signal in INTRADAY_SIGNALS else "multiday",
                        )
                    executed_symbols.add(sym)
                    all_trades.append(
                        {
                            "symbol": sym,
                            "action": action_label,
                            "detail": (
                                f"late-fill @ ${entry_price:.2f} | "
                                f"{signal} | conf={confidence} | mode={mode}"
                            ),
                        }
                    )
                    audit_log.log_event(
                        "ORDER_LATE_FILL_RECONCILED",
                        {
                            "symbol": sym,
                            "entry_price": round(entry_price, 4),
                            "signal": signal,
                            "confidence": confidence,
                            "run_mode": mode,
                        },
                    )
        except Exception as e:
            logger.warning(f"Late-fill reconciliation failed: {e}")


def _finalise(
    client,
    today: str,
    mode: str,
    account_before: dict,
    decisions: dict,
    all_trades: list,
    run_id: str,
    dry_run: bool,
    _live_shadow: bool,
) -> None:
    """Save run record, print summary, generate dashboard, send close-mode email."""
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
    if _live_shadow:
        would_buys = [t for t in all_trades if t["action"] == "WOULD_BUY"]
        would_sells = [t for t in all_trades if t["action"] == "WOULD_SELL"]
        audit_log.log_event(
            "LIVE_SHADOW_COMPLETE",
            {
                "mode": mode,
                "would_buy_count": len(would_buys),
                "would_sell_count": len(would_sells),
                "would_buys": [t["detail"] for t in would_buys],
                "would_sells": [t["symbol"] for t in would_sells],
            },
        )
        logger.info(
            f"[LIVE SHADOW] Complete — {len(would_buys)} WOULD_BUY, "
            f"{len(would_sells)} WOULD_SELL (no orders placed)"
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


def _evaluate_risk_limits(client, account_before: dict, dry_run: bool) -> RiskFlags | None:
    """Run startup health, circuit breaker, daily loss, and experiment drawdown checks.

    Returns None when the daily loss limit is hit (positions are closed inside this function;
    the caller must immediately return so no further trading occurs).
    """
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
    _dd_scalar = position_sizer.drawdown_scalar(history)
    cb_triggered, cb_drawdown = risk_manager.check_circuit_breaker(history)
    if cb_triggered:
        audit_log.log_circuit_breaker(cb_drawdown)
        alerts.alert_circuit_breaker(cb_drawdown)
        logger.warning("Circuit breaker active — no new buys today.")

    # ── Daily loss check ───────────────────────────────────────────────────────
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
        return None

    # ── Experiment drawdown cap ───────────────────────────────────────────────
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

    return RiskFlags(
        health_status=health.status,
        dd_scalar=_dd_scalar,
        cb_triggered=cb_triggered,
        daily_loss_triggered=False,
        daily_loss_pct=dl_pct,
        exp_drawdown_triggered=_exp_drawdown_triggered,
    )


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

    # ── Reconcile position metadata ───────────────────────────────────────────
    # Runs before the market-closed check so the DB always reflects broker state
    # regardless of market hours — stale entries from JSON migration are pruned here.
    # reconcile_positions returns symbols that exist at the broker but had no
    # local record — unexpected positions detected BEFORE they are normalised.
    unexpected_positions = trader.reconcile_positions(client)

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

    # ── Risk limits (health, circuit breaker, daily loss, experiment drawdown) ─
    flags = _evaluate_risk_limits(client, account_before, dry_run)
    if flags is None:
        return

    all_trades: list = []
    executed_symbols: set[str] = set()

    # ── Market context ────────────────────────────────────────────────────────
    mc = _fetch_market_context()

    # ── Position state + managed exits ───────────────────────────────────────
    snap = _get_position_snapshot(client)
    _manage_existing_positions(client, dry_run, all_trades, snap, mode=mode)
    snap = _get_position_snapshot(client)  # refresh after exits

    # ── Fetch + enrich market data, pre-filter candidates ────────────────────
    db = _build_data_bundle(client, snap, mc, mode)
    if db is None:
        return

    # ── AI analysis + validation + decision logging ───────────────────────────
    account_now = trader.get_account_info(client)
    decisions = _run_ai_phase(db, snap, mc, account_now, run_id, mode, executed_symbols)
    if decisions is None:
        return

    # ── Force-cover intraday positions (close pass only) ─────────────────────
    if mode == "close":
        _force_cover_intraday_positions(client, dry_run, all_trades)

    # ── Execute sells / covers ────────────────────────────────────────────────
    _execute_sell_phase(
        client,
        snap,
        decisions,
        all_trades,
        executed_symbols,
        dry_run,
        _live_shadow,
        snapshots=db.snapshots,
        regime_name=mc.regime.get("regime", "UNKNOWN"),
    )

    # ── Execute buys ────────────────────────────────────────────────────────────
    open_positions, account_now = _execute_buy_phase(
        client=client,
        snap=snap,
        db=db,
        mc=mc,
        decisions=decisions,
        health_status=flags.health_status,
        cb_triggered=flags.cb_triggered,
        _exp_drawdown_triggered=flags.exp_drawdown_triggered,
        _dd_scalar=flags.dd_scalar,
        account_now=account_now,
        should_run_live_gates=should_run_live_gates,
        today=today,
        mode=mode,
        dry_run=dry_run,
        _live_shadow=_live_shadow,
        all_trades=all_trades,
        executed_symbols=executed_symbols,
    )

    # ── Execute shorts ────────────────────────────────────────────────────────
    _execute_shorts(
        client=client,
        snapshots=db.short_snapshots,
        regime=mc.regime,
        open_positions=open_positions,
        account_now=account_now,
        all_trades=all_trades,
        executed_symbols=executed_symbols,
        dry_run=dry_run,
        _live_shadow=_live_shadow,
    )

    # ── Attach stops + late-fill reconciliation ──────────────────────────────
    _reconcile_late_fills(client, today, mode, mc, decisions, executed_symbols, all_trades, dry_run)

    # ── Finalise ──────────────────────────────────────────────────────────────
    _finalise(
        client, today, mode, account_before, decisions, all_trades, run_id, dry_run, _live_shadow
    )


if __name__ == "__main__":  # pragma: no cover
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
