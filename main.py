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

import sys
import os
import time
import logging
import argparse
from datetime import date, datetime, timezone

import config
from data import market_data, news_fetcher, options_scanner, sector_data
from data import sentiment as sentiment_module
from analysis import ai_analyst, performance
from analysis.weekly_review import get_latest_review
from risk import risk_manager, earnings_calendar, macro_calendar, position_sizer
from execution import trader, stock_scanner
from notifications import emailer, alerts
from utils import portfolio_tracker
from utils.portfolio_tracker import get_day_summary
from utils import audit_log
from utils import decision_log
from utils.validators import validate_ai_response, sanitize_headlines, check_pre_trade

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

def _acquire_lock() -> bool:
    lock_file = _lock_file()
    os.makedirs(config.LOG_DIR, exist_ok=True)
    if os.path.exists(lock_file):
        age = time.time() - os.path.getmtime(lock_file)
        if age > _LOCK_MAX_AGE_SECONDS:
            logger.warning(f"Stale lock file found ({age / 3600:.1f}h old) — auto-clearing")
            os.remove(lock_file)
        else:
            logger.warning("Lock file exists — another run may be in progress. Remove .lock file to override.")
            return False
    with open(lock_file, "w"):
        pass
    return True


def _release_lock():
    try:
        os.remove(_lock_file())
    except FileNotFoundError:
        pass


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

    # 2. Close all open positions
    positions = trader.get_open_positions(client)
    closed = 0
    for pos in positions:
        result = trader.close_position(client, pos["symbol"])
        if result:
            closed += 1
            pl = pos["unrealized_pl"]
            logger.critical(f"  Closed {pos['symbol']}  P&L: {'+' if pl >= 0 else ''}{pl:.2f}")
            audit_log.log_position_closed(pos["symbol"], "kill_switch", pos["unrealized_plpc"])

    # 3. Write HALT file
    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(config.HALT_FILE, "w") as f:
        f.write(f"HALTED: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Positions closed: {closed}\n")
        f.write(f"To resume trading: python main.py --clear-halt\n")

    audit_log.log_kill_switch(closed)
    alerts.alert_error("KILL SWITCH", f"Emergency halt activated. {closed} positions liquidated. Delete HALT file to resume.")
    logger.critical(f"Kill switch complete. {closed} positions closed.")
    logger.critical(f"To resume: python main.py --clear-halt")


def _run_clear_halt():
    if os.path.exists(config.HALT_FILE):
        os.remove(config.HALT_FILE)
        audit_log.log_halt_cleared()
        logger.info("Halt cleared. Trading will resume on next scheduled run.")
    else:
        logger.info("No halt file found — trading is already active.")


# ── Partial exits ─────────────────────────────────────────────────────────────

def _handle_partial_exits(client, positions: list, dry_run: bool) -> list:
    """Sell half of any position up more than PARTIAL_PROFIT_PCT."""
    executed = []
    for pos in positions:
        if pos["unrealized_plpc"] >= config.PARTIAL_PROFIT_PCT:
            symbol = pos["symbol"]
            half_qty = pos["qty"] / 2
            logger.info(f"Partial exit: {symbol} +{pos['unrealized_plpc']:.1f}% — selling {half_qty:.6f} shares")
            if not dry_run:
                trader.cancel_open_orders(client, symbol)
                result = trader.place_partial_sell(client, symbol, half_qty)
                if result:
                    audit_log.log_order_placed(symbol, "SELL_PARTIAL", pos["market_value"] / 2, result["order_id"])
                    remaining_qty = trader.wait_for_fill(client, result["order_id"])
                    if remaining_qty is not None:
                        audit_log.log_order_filled(symbol, result["order_id"], remaining_qty)
                        trader.place_trailing_stop(client, symbol, pos["qty"] - half_qty, current_price=pos["current_price"])
                    executed.append({
                        "symbol": symbol,
                        "action": "PARTIAL SELL",
                        "detail": f"50% at +{pos['unrealized_plpc']:.1f}%",
                    })
            else:
                logger.info(f"  [DRY RUN] Would partial-sell {symbol}")
                executed.append({"symbol": symbol, "action": "PARTIAL SELL", "detail": "dry run"})
    return executed


# ── Main run ──────────────────────────────────────────────────────────────────

def run(dry_run: bool = False, mode: str = "open"):
    today = config.today_et().isoformat()
    mode_label = "PAPER" if config.IS_PAPER else "*** LIVE ***"
    logger.info(f"=== Trading bot | {today} | mode={mode} | {mode_label} {'[DRY RUN]' if dry_run else ''} ===")
    if not config.IS_PAPER and not dry_run:
        if config.LIVE_CONFIRM != "I-ACCEPT-REAL-MONEY-RISK":
            logger.error(
                "Live trading requires LIVE_CONFIRM=I-ACCEPT-REAL-MONEY-RISK in the environment. "
                "Set it in .env or export it before running."
            )
            sys.exit(1)

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
        logger.critical(f"Trading is HALTED. Delete halt file or run: python main.py --clear-halt")
        sys.exit(1)

    if not _acquire_lock():
        return

    try:
        _run_inner(dry_run=dry_run, mode=mode, today=today)
    except Exception as e:
        logger.error(f"Unhandled error in trading run: {e}", exc_info=True)
        alerts.alert_error("main.run", str(e))
    finally:
        _release_lock()


def _run_inner(dry_run: bool, mode: str, today: str):
    client = trader.get_client()

    if not trader.is_market_open(client):
        logger.info("Market is closed. Nothing to do.")
        return

    # ── Account snapshot ──────────────────────────────────────────────────────
    account_before = trader.get_account_info(client)
    logger.info(f"Portfolio: ${account_before['portfolio_value']:.2f}  Cash: ${account_before['cash']:.2f}")
    audit_log.log_run_start(mode, account_before["portfolio_value"], account_before["cash"], config.IS_PAPER)

    if mode == "open":
        portfolio_tracker.save_daily_baseline(account_before["portfolio_value"])

    # ── Reconcile position metadata ───────────────────────────────────────────
    trader.reconcile_positions(client)
    trader.ensure_stops_attached(client)

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
    if dl_triggered:
        audit_log.log_daily_loss_limit(dl_pct)
        alerts.alert_daily_loss(dl_pct)
        logger.warning("Daily loss limit hit — closing all positions.")
        if not dry_run:
            for pos in trader.get_open_positions(client):
                trader.close_position(client, pos["symbol"])
                audit_log.log_position_closed(pos["symbol"], "daily_loss_limit", pos["unrealized_plpc"])
                trader.record_sell(pos["symbol"])
        return

    all_trades: list = []
    daily_notional_spent = 0.0  # tracks total notional for daily cap check
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

    # ── Midday/close: position management only, no new buys ──────────────────
    if mode in ("midday", "close"):
        account_after = trader.get_account_info(client)
        record = portfolio_tracker.save_daily_run(
            date=f"{today}-{mode}",
            account_before=account_before,
            account_after=account_after,
            ai_decisions={"market_summary": f"{mode} check", "position_decisions": [], "buy_candidates": []},
            trades_executed=all_trades,
            stop_losses_triggered=[],
        )
        portfolio_tracker.print_summary(record)
        audit_log.log_run_end(mode, record["daily_pnl"], len(all_trades), account_after["portfolio_value"])
        if mode == "close" and not dry_run:
            day_summary = get_day_summary(today)
            if day_summary:
                emailer.send_summary(day_summary)
        return

    # ── Full open-mode cycle ──────────────────────────────────────────────────

    # Exit earnings-risk positions
    for symbol, ed in earnings_risk.items():
        if symbol in held_symbols:
            logger.warning(f"Exiting {symbol} — earnings on {ed}")
            audit_log.log_earnings_exit(symbol, str(ed))
            if not dry_run:
                result = trader.close_position(client, symbol)
                if result:
                    meta = trader.get_position_meta(symbol)
                    pos = next((p for p in open_positions if p["symbol"] == symbol), None)
                    if pos:
                        performance.record_trade_outcome(
                            meta["signal"], pos["unrealized_plpc"],
                            regime=meta["regime"], confidence=meta["confidence"],
                        )
                        audit_log.log_position_closed(symbol, "earnings_exit", pos["unrealized_plpc"])
                    trader.record_sell(symbol)
                    all_trades.append({**result, "action": "SELL", "detail": "earnings exit"})

    open_positions = trader.get_open_positions(client)
    held_symbols = {p["symbol"] for p in open_positions}
    position_ages = trader.get_position_ages()
    stale = trader.get_stale_positions(config.MAX_HOLD_DAYS)

    # ── Scan universe ─────────────────────────────────────────────────────────
    logger.info("Scanning for top movers...")
    top_movers = stock_scanner.get_top_movers(config.TOP_MOVERS_COUNT)
    scan_symbols = list(set(config.STOCK_UNIVERSE) | held_symbols | set(top_movers))
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
    filtered_candidates = stock_scanner.prefilter_candidates(candidate_snaps)
    ai_snapshots = held_snaps + filtered_candidates
    logger.info(f"Pre-filter: {len(candidate_snaps)} candidates → {len(filtered_candidates)} passed")

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
    )

    if not decisions:
        logger.error("AI analysis failed. Aborting.")
        return

    # ── Validate AI response before acting ───────────────────────────────────
    is_valid, validation_errors = validate_ai_response(decisions, ai_known_symbols, held_symbols=held_symbols)
    if not is_valid:
        audit_log.log_validation_failure(validation_errors)
        logger.error(f"AI response failed validation — aborting buys. Errors: {validation_errors}")
        # Allow sells to proceed for safety, but block all buys
        decisions["buy_candidates"] = []

    logger.info(f"Market: {decisions.get('market_summary', '')}")
    decision_log.log_decisions(decisions, mode, executed_symbols)
    audit_log.log_ai_decision(
        decisions.get("market_summary", ""),
        len(decisions.get("buy_candidates", [])),
        sum(1 for d in decisions.get("position_decisions", []) if d.get("action") == "SELL"),
    )

    # ── Execute sells ─────────────────────────────────────────────────────────
    symbols_to_sell = {d["symbol"] for d in decisions.get("position_decisions", []) if d["action"] == "SELL"}
    symbols_to_sell |= {sym for sym in stale if sym in held_symbols}

    for symbol in symbols_to_sell:
        decision = next((d for d in decisions.get("position_decisions", []) if d["symbol"] == symbol), None)
        reason = decision["reasoning"] if decision else f"Time-based exit (≥{config.MAX_HOLD_DAYS} days)"
        logger.info(f"  SELL {symbol} — {reason}")
        if not dry_run:
            pos = next((p for p in open_positions if p["symbol"] == symbol), None)
            result = trader.close_position(client, symbol)
            if result:
                if pos:
                    meta = trader.get_position_meta(symbol)
                    performance.record_trade_outcome(
                        meta["signal"], pos["unrealized_plpc"],
                        regime=meta["regime"], confidence=meta["confidence"],
                    )
                    audit_log.log_position_closed(symbol, reason[:50], pos["unrealized_plpc"])
                trader.record_sell(symbol)
                executed_symbols.add(symbol)
                all_trades.append({**result, "action": "SELL", "detail": reason})
            else:
                logger.error(f"  SELL FAILED {symbol} — close_position returned None after retries. Manual review required.")
                alerts.alert_error("SELL FAILED", f"{symbol}: failed to close position after 3 attempts — manual review required.")
        else:
            executed_symbols.add(symbol)
            all_trades.append({"symbol": symbol, "action": "SELL", "detail": "dry run"})

    # ── Execute buys (with pre-trade controls) ────────────────────────────────
    skip_buys = cb_triggered or regime.get("is_bearish") or macro.get("is_high_risk")
    if skip_buys:
        reasons = []
        if cb_triggered: reasons.append("circuit breaker")
        if regime.get("is_bearish"): reasons.append("bear market filter")
        if macro.get("is_high_risk"): reasons.append(f"macro event: {macro.get('event')}")
        logger.warning(f"Skipping new buys: {', '.join(reasons)}")
    else:
        account_now = trader.get_account_info(client)
        open_positions = trader.get_open_positions(client)
        available_cash = account_now["cash"] * (1 - config.CASH_RESERVE_PCT)
        max_positions = position_sizer.get_max_positions(account_now["portfolio_value"])
        slots = max_positions - len(open_positions)
        logger.info(f"Position slots: {len(open_positions)}/{max_positions}")

        if slots > 0:
            min_confidence = config.MIN_CONFIDENCE + (1 if vix and vix > 25 else 0)
            raw_candidates = [
                c for c in decisions.get("buy_candidates", [])
                if c["confidence"] >= min_confidence
            ]
            valid_candidates = risk_manager.validate_buy_candidates(
                raw_candidates,
                held_symbols={p["symbol"] for p in open_positions},
                sector_map_fn=sector_data.get_sector,
                max_per_sector=config.MAX_SECTOR_POSITIONS,
            )
            valid_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            valid_candidates = valid_candidates[:slots]

            orders_placed = 0
            for candidate in valid_candidates:
                if orders_placed >= config.MAX_ORDERS_PER_RUN:
                    logger.warning(f"MAX_ORDERS_PER_RUN ({config.MAX_ORDERS_PER_RUN}) reached — no more buys this run")
                    break
                symbol = candidate["symbol"]
                confidence = candidate["confidence"]
                kelly = position_sizer.kelly_fraction(
                    confidence,
                    signal=candidate.get("key_signal", "unknown"),
                    regime=regime.get("regime", "UNKNOWN"),
                )
                notional = min(
                    available_cash * kelly,
                    account_now["portfolio_value"] * config.MAX_POSITION_PCT,
                )

                # Pre-trade controls (MiFID II)
                approved, rejection_reason = check_pre_trade(
                    symbol, notional, daily_notional_spent,
                    config.MAX_SINGLE_ORDER_USD, config.MAX_DAILY_NOTIONAL_USD,
                )
                if not approved:
                    logger.warning(f"  Pre-trade check failed: {rejection_reason}")
                    continue

                logger.info(f"  BUY {symbol}: ${notional:.2f} Kelly {kelly:.0%} | conf={confidence}")

                if notional >= 1.0:
                    if not dry_run:
                        result = trader.place_buy_order(client, symbol, notional)
                        if result:
                            orders_placed += 1
                            daily_notional_spent += notional
                            snap = next((s for s in snapshots if s["symbol"] == symbol), None)
                            entry_price = snap["current_price"] if snap else 0.0
                            trader.record_buy(
                                symbol, entry_price,
                                signal=candidate.get("key_signal", "unknown"),
                                regime=regime.get("regime", "UNKNOWN"),
                                confidence=confidence,
                            )
                            audit_log.log_order_placed(symbol, "BUY", notional, result["order_id"])
                            if result.get("filled_qty"):
                                audit_log.log_order_filled(symbol, result["order_id"], result["filled_qty"])
                                current_price = snap["current_price"] if snap else None
                                stop_result = trader.place_trailing_stop(client, symbol, result["filled_qty"], current_price=current_price)
                                if stop_result is None:
                                    logger.error(f"  Stop placement FAILED for {symbol} — position unprotected!")
                                    alerts.alert_error("STOP FAILED", f"{symbol}: trailing stop placement failed after buy — position has no downside protection.")
                            executed_symbols.add(symbol)
                            detail = f"${notional:.2f} | Kelly {kelly:.0%} | {candidate.get('key_signal')} | confidence={confidence}"
                            all_trades.append({**result, "action": "BUY", "detail": detail})
                    else:
                        orders_placed += 1
                        daily_notional_spent += notional
                        executed_symbols.add(symbol)
                        all_trades.append({"symbol": symbol, "action": "BUY", "detail": f"dry run ${notional:.2f}"})
                else:
                    logger.warning(f"  Skipping {symbol}: ${notional:.2f} too small")

    # ── Attach any missing stops (catches fills that arrived after wait_for_fill timed out) ──
    if not dry_run and all_trades:
        trader.ensure_stops_attached(client)

    # ── Finalise ──────────────────────────────────────────────────────────────
    account_after = trader.get_account_info(client)
    record = portfolio_tracker.save_daily_run(
        date=today,
        account_before=account_before,
        account_after=account_after,
        ai_decisions=decisions,
        trades_executed=all_trades,
        stop_losses_triggered=[],
    )
    portfolio_tracker.print_summary(record)
    performance.generate_dashboard(portfolio_tracker.load_history())
    audit_log.log_run_end(mode, record["daily_pnl"], len(all_trades), account_after["portfolio_value"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=["open", "midday", "close"], default="open")
    parser.add_argument("--kill-switch", action="store_true", help="Emergency: liquidate all and halt")
    parser.add_argument("--clear-halt", action="store_true", help="Remove halt file and resume trading")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--capital", type=float, default=None,
                        help="Starting capital for backtest. Defaults to current Alpaca portfolio value.")
    args = parser.parse_args()

    if args.kill_switch:
        _run_kill_switch()
    elif args.clear_halt:
        _run_clear_halt()
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
            config.STOCK_UNIVERSE, args.start, args.end, capital,
            max_positions=config.MAX_POSITIONS,
        )
    else:
        run(dry_run=args.dry_run, mode=args.mode)
