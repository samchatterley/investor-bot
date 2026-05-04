import logging
import math
import time
from datetime import date

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, TrailingStopOrderRequest

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    IS_PAPER,
    TRAILING_STOP_PCT,
    today_et,
)
from models import BrokerStateUnavailable, OrderResult, OrderStatus
from utils.retry import with_retry

logger = logging.getLogger(__name__)


def get_client() -> TradingClient:
    return TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=IS_PAPER,
    )


@with_retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
def get_account_info(client: TradingClient) -> dict:
    account = client.get_account()
    return {
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
    }


@with_retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
def get_open_positions(client: TradingClient) -> list[dict]:
    positions = client.get_all_positions()
    result = []
    for p in positions:
        result.append(
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,  # convert to %
                "market_value": float(p.market_value),
            }
        )
    return result


def has_pending_buy(client: TradingClient, symbol: str) -> bool:
    """Return True if broker has a pending/active buy order for this symbol.

    Raises BrokerStateUnavailable if the broker cannot be queried — callers must
    treat this as trade-blocking, not permissive. Unknown broker state is more
    dangerous than a missed buy opportunity.
    """
    try:
        orders = client.get_orders()
        for o in orders:
            if (
                o.symbol == symbol
                and o.side == OrderSide.BUY
                and str(o.status)
                in (
                    "new",
                    "accepted",
                    "pending_new",
                    "partially_filled",
                )
            ):
                return True
        return False
    except Exception as e:
        raise BrokerStateUnavailable(f"has_pending_buy({symbol}): {e}") from e


def place_buy_order(
    client: TradingClient,
    symbol: str,
    notional_usd: float,
    run_id: str | None = None,
) -> OrderResult | None:
    """Place a fractional market buy, wait for fill, return OrderResult.

    client_order_id is symbol+date-stable so reruns on the same day reuse the
    same ID. Alpaca will 409 a duplicate submission, which surfaces as a
    REJECTED result that the caller treats as a no-op (no new exposure created).

    Every attempt is recorded in the order_intents ledger before broker submission
    and updated on every state transition so process restarts are safe.
    """
    if notional_usd < 1.0:
        logger.warning(f"Order too small for {symbol}: ${notional_usd:.2f}")
        return None

    client_order_id = f"ib-{symbol}-BUY-{today_et().isoformat()}"
    trade_date = today_et().isoformat()

    # Record intent before broker submission — if the process crashes after
    # submission the ledger will show 'submitted' and flag for reconciliation.
    try:
        from utils.order_ledger import create_intent, log_order_event, update_intent

        _ledger = True
    except Exception:
        _ledger = False

    if _ledger:
        create_intent(symbol, "BUY", trade_date, notional_usd, client_order_id)
        log_order_event(client_order_id, "INTENT_CREATED", {"notional": notional_usd})

    try:
        req_kwargs: dict = {
            "symbol": symbol,
            "notional": round(notional_usd, 2),
            "side": OrderSide.BUY,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": client_order_id,
        }
        order = client.submit_order(MarketOrderRequest(**req_kwargs))
        order_id = str(order.id)
        logger.info(f"BUY order placed: {symbol} ${notional_usd:.2f} | order_id={order_id}")

        if _ledger:
            update_intent(client_order_id, "submitted", broker_order_id=order_id)
            log_order_event(
                client_order_id,
                "ORDER_SUBMITTED",
                {"broker_order_id": order_id},
                broker_order_id=order_id,
            )

        filled_qty = wait_for_fill(client, order_id)
        if filled_qty is not None:
            if _ledger:
                update_intent(client_order_id, "filled")
                log_order_event(
                    client_order_id,
                    "ORDER_FILLED",
                    {"filled_qty": filled_qty},
                    broker_order_id=order_id,
                )
            return OrderResult(
                status=OrderStatus.FILLED,
                symbol=symbol,
                broker_order_id=order_id,
                filled_qty=filled_qty,
            )
        try:
            final = client.get_order_by_id(order_id)
            if str(final.status) == "partially_filled" and final.filled_qty:
                partial_qty = float(final.filled_qty)
                logger.warning(
                    f"BUY for {symbol} partially filled {partial_qty} of ${notional_usd:.2f}"
                )
                if _ledger:
                    update_intent(client_order_id, "partial")
                    log_order_event(
                        client_order_id,
                        "ORDER_PARTIAL",
                        {"filled_qty": partial_qty},
                        broker_order_id=order_id,
                    )
                return OrderResult(
                    status=OrderStatus.PARTIAL,
                    symbol=symbol,
                    broker_order_id=order_id,
                    filled_qty=partial_qty,
                )
        except Exception:
            pass
        if _ledger:
            update_intent(client_order_id, "timeout")
            log_order_event(client_order_id, "ORDER_TIMEOUT", {}, broker_order_id=order_id)
        return OrderResult(
            status=OrderStatus.TIMEOUT, symbol=symbol, broker_order_id=order_id, filled_qty=0.0
        )
    except Exception as e:
        logger.error(f"Failed to place BUY for {symbol}: {e}")
        if _ledger:
            update_intent(client_order_id, "rejected")
            log_order_event(client_order_id, "ORDER_REJECTED", {"reason": str(e)})
        return OrderResult(status=OrderStatus.REJECTED, symbol=symbol, rejection_reason=str(e))


_TERMINAL_FAIL_STATUSES = frozenset({"rejected", "cancelled", "expired", "done_for_day", "stopped"})


def wait_for_fill(client: TradingClient, order_id: str, max_wait: int = 30) -> float | None:
    """Poll until a market order reaches 'filled'. Returns filled qty on success, None otherwise.

    Exits early on terminal failure statuses (rejected/cancelled/expired) rather than
    burning the full poll window. Does NOT return early on 'partially_filled' — callers
    must check for partial fill qty themselves after this returns None.
    """
    for _ in range(max_wait):
        try:
            o = client.get_order_by_id(order_id)
            status_str = str(o.status)
            if status_str == "filled" and o.filled_qty:
                return float(o.filled_qty)
            if status_str in _TERMINAL_FAIL_STATUSES:
                logger.info(f"Order {order_id} reached terminal state: {status_str}")
                return None
        except Exception:
            pass
        time.sleep(1)
    logger.warning(f"Order {order_id} did not fill within {max_wait}s")
    return None


def place_trailing_stop(
    client: TradingClient,
    symbol: str,
    qty: float,
    current_price: float | None = None,
    trail_percent: float | None = None,
) -> OrderResult | None:
    """Attach a stop to an open position.

    Alpaca rejects stop orders for fractional share quantities (error 42210000).
    When qty is fractional, a fixed stop is placed for the whole-share portion and
    the fractional remainder is immediately liquidated via a market sell so no shares
    run unprotected. If floor(qty) < 1 (entire position is sub-share), no stop can
    be placed and UNPROTECTED is returned.

    trail_percent overrides config.TRAILING_STOP_PCT when provided (used for VIX adjustment).

    Returns None only for the qty <= 0 pre-condition violation. All real attempts
    return an OrderResult (FILLED on success, STOP_FAILED/UNPROTECTED on failure).
    """
    if not qty or qty <= 0:
        return None
    effective_trail = trail_percent if trail_percent is not None else TRAILING_STOP_PCT
    # Truncate (not round) to avoid requesting more qty than Alpaca reports available
    safe_qty = math.floor(qty * 1_000_000) / 1_000_000
    is_fractional = abs(safe_qty - round(safe_qty)) > 0.000001

    if is_fractional:
        whole_qty = int(math.floor(safe_qty))
        remainder = round(safe_qty - whole_qty, 6)
        if whole_qty < 1:
            logger.warning(
                f"Cannot stop-protect {symbol}: position is entirely sub-share "
                f"({safe_qty:.6f}) — Alpaca does not support fractional stop orders"
            )
            return OrderResult(status=OrderStatus.UNPROTECTED, symbol=symbol)
        if not current_price:
            logger.warning(f"Cannot place stop for {symbol}: no current price")
            return OrderResult(
                status=OrderStatus.STOP_FAILED, symbol=symbol, rejection_reason="no current price"
            )
        stop_price = round(current_price * (1 - effective_trail / 100), 2)
        try:
            order = client.submit_order(
                StopOrderRequest(
                    symbol=symbol,
                    qty=whole_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_price,
                )
            )
            order_id = str(order.id)
            logger.info(
                f"Stop order: {symbol} {whole_qty} shares (of {safe_qty:.6f}) "
                f"stop=${stop_price} | order_id={order_id}"
            )
            # Liquidate fractional remainder — Alpaca cannot stop-protect it
            if remainder >= 0.001:
                try:
                    client.submit_order(
                        MarketOrderRequest(
                            symbol=symbol,
                            qty=remainder,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                        )
                    )
                    logger.info(f"Liquidated fractional remainder {remainder:.6f} of {symbol}")
                except Exception as e:
                    logger.warning(
                        f"Could not liquidate fractional remainder {remainder:.6f} of {symbol}: {e}"
                    )
            return OrderResult(status=OrderStatus.FILLED, symbol=symbol, stop_order_id=order_id)
        except Exception as e:
            logger.error(f"Failed to place trailing stop for {symbol}: {e}")
            return OrderResult(
                status=OrderStatus.STOP_FAILED, symbol=symbol, rejection_reason=str(e)
            )
    else:
        try:
            order = client.submit_order(
                TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=safe_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    trail_percent=effective_trail,
                )
            )
            order_id = str(order.id)
            logger.info(f"Trailing stop: {symbol} {effective_trail}% trail | order_id={order_id}")
            return OrderResult(status=OrderStatus.FILLED, symbol=symbol, stop_order_id=order_id)
        except Exception as e:
            logger.error(f"Failed to place trailing stop for {symbol}: {e}")
            return OrderResult(
                status=OrderStatus.STOP_FAILED, symbol=symbol, rejection_reason=str(e)
            )


def place_sell_order(client: TradingClient, symbol: str, qty: float) -> OrderResult:
    """Sell an entire position (by quantity). Polls for fill confirmation."""
    try:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 6),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        order_id = str(order.id)
        logger.info(f"SELL order placed: {symbol} qty={qty} | order_id={order_id}")
        filled_qty = wait_for_fill(client, order_id)
        if filled_qty is not None:
            return OrderResult(
                status=OrderStatus.FILLED,
                symbol=symbol,
                broker_order_id=order_id,
                filled_qty=filled_qty,
            )
        try:
            final = client.get_order_by_id(order_id)
            if str(final.status) == "partially_filled" and final.filled_qty:
                partial_qty = float(final.filled_qty)
                logger.warning(f"SELL for {symbol} partially filled {partial_qty} of {qty}")
                return OrderResult(
                    status=OrderStatus.PARTIAL,
                    symbol=symbol,
                    broker_order_id=order_id,
                    filled_qty=partial_qty,
                )
        except Exception:
            pass
        return OrderResult(
            status=OrderStatus.TIMEOUT, symbol=symbol, broker_order_id=order_id, filled_qty=0.0
        )
    except Exception as e:
        logger.error(f"Failed to place SELL for {symbol}: {e}")
        return OrderResult(status=OrderStatus.REJECTED, symbol=symbol, rejection_reason=str(e))


def close_position(client: TradingClient, symbol: str) -> OrderResult:
    """Close an entire position. Cancels open orders first (trailing stops hold shares),
    then polls until the market order is confirmed filled."""
    cancel_open_orders(client, symbol)
    try:
        order = client.close_position(symbol)
        order_id = str(order.id)
        logger.info(f"Close submitted: {symbol} | order_id={order_id}")
        filled_qty = wait_for_fill(client, order_id)
        if filled_qty is not None:
            return OrderResult(
                status=OrderStatus.FILLED,
                symbol=symbol,
                broker_order_id=order_id,
                filled_qty=filled_qty,
            )
        try:
            final = client.get_order_by_id(order_id)
            if str(final.status) == "partially_filled" and final.filled_qty:
                partial_qty = float(final.filled_qty)
                logger.warning(
                    f"Close for {symbol} partially filled {partial_qty} — position may still be open"
                )
                return OrderResult(
                    status=OrderStatus.PARTIAL,
                    symbol=symbol,
                    broker_order_id=order_id,
                    filled_qty=partial_qty,
                )
        except Exception:
            pass
        logger.warning(
            f"Close for {symbol} did not confirm fill within poll window — position may still be open"
        )
        return OrderResult(
            status=OrderStatus.TIMEOUT, symbol=symbol, broker_order_id=order_id, filled_qty=0.0
        )
    except Exception as e:
        logger.error(f"Failed to close position {symbol}: {e}")
        return OrderResult(status=OrderStatus.REJECTED, symbol=symbol, rejection_reason=str(e))


@with_retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
def is_market_open(client: TradingClient) -> bool:
    clock = client.get_clock()
    return clock.is_open


# ---------- Position metadata (SQLite) ----------


def _db():
    from utils.db import get_db

    return get_db()


def get_position_signal(symbol: str) -> str:
    """Return the entry signal stored for a position, or 'unknown'."""
    try:
        with _db() as conn:
            row = conn.execute("SELECT signal FROM positions WHERE symbol=?", (symbol,)).fetchone()
        return row["signal"] if row else "unknown"
    except Exception:
        return "unknown"


def record_buy(
    symbol: str,
    entry_price: float,
    signal: str = "unknown",
    regime: str = "UNKNOWN",
    confidence: int = 0,
):
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO positions (symbol, entry_date, entry_price, signal, regime, confidence) "
            "VALUES (?,?,?,?,?,?)",
            (symbol, today_et().isoformat(), round(entry_price, 4), signal, regime, confidence),
        )


def record_sell(symbol: str):
    with _db() as conn:
        conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))


def record_partial_exit(symbol: str):
    """Mark that a partial profit exit has been taken for this position."""
    from datetime import UTC, datetime

    try:
        with _db() as conn:
            conn.execute(
                "UPDATE positions SET partial_exit_taken_at=? WHERE symbol=?",
                (datetime.now(UTC).isoformat(), symbol),
            )
    except Exception as e:
        logger.warning(f"record_partial_exit failed for {symbol}: {e}")


def get_position_meta(symbol: str) -> dict:
    """Return full entry metadata for a position (signal, regime, confidence, entry_price, entry_date)."""
    defaults = {"signal": "unknown", "regime": "UNKNOWN", "confidence": 0, "entry_price": 0.0}
    try:
        with _db() as conn:
            row = conn.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        if row:
            return {**defaults, **dict(row)}
    except Exception:
        pass
    return defaults


def _load_all_positions() -> dict:
    """Return all position metadata as {symbol: dict}."""
    try:
        with _db() as conn:
            rows = conn.execute("SELECT * FROM positions").fetchall()
        return {row["symbol"]: dict(row) for row in rows}
    except Exception:
        return {}


def get_position_ages() -> dict[str, int]:
    """Return {symbol: approximate_trading_days_held}."""
    meta = _load_all_positions()
    today = today_et()
    ages = {}
    for symbol, data in meta.items():
        try:
            entry = date.fromisoformat(data["entry_date"])
            # bdate_range is inclusive of both endpoints, so subtract 1 to get
            # days elapsed since entry (same-day entry = 0 elapsed → clamped to 1).
            # Stale threshold is >= MAX_HOLD_DAYS (default 3), so same-day = 1 is safe.
            trading_days = max(1, len(pd.bdate_range(entry.isoformat(), today.isoformat())) - 1)
            ages[symbol] = trading_days
        except Exception:
            ages[symbol] = 1
    return ages


def get_stale_positions(max_days: int = 3) -> list[str]:
    """Return symbols held for more than max_days trading days."""
    return [sym for sym, age in get_position_ages().items() if age >= max_days]


def reconcile_positions(client: TradingClient):
    """
    Sync the SQLite positions table with actual Alpaca positions.
    Removes stale entries for positions that no longer exist,
    adds placeholder entries for positions with no metadata.
    """
    try:
        positions = client.get_all_positions()
    except Exception as e:
        logger.error(f"reconcile_positions: could not fetch positions — {e}")
        return
    actual = {p.symbol for p in positions}
    try:
        with _db() as conn:
            stored = {row["symbol"] for row in conn.execute("SELECT symbol FROM positions")}

            for sym in stored - actual:
                logger.info(f"Reconcile: removing stale metadata for {sym}")
                conn.execute("DELETE FROM positions WHERE symbol=?", (sym,))

            for sym in actual - stored:
                logger.info(f"Reconcile: adding missing metadata for {sym}")
                conn.execute(
                    "INSERT OR IGNORE INTO positions (symbol, entry_date, entry_price, signal, regime, confidence) "
                    "VALUES (?,?,?,?,?,?)",
                    (sym, today_et().isoformat(), 0.0, "unknown", "UNKNOWN", 0),
                )
    except Exception as e:
        logger.error(f"reconcile_positions: database error — {e}")


def ensure_stops_attached(client: TradingClient) -> bool:
    """
    Detect open positions with no active trailing stop and attach one.

    Guards against the gap where place_buy_order timed out waiting for fill
    but the order subsequently filled — leaving a live position unprotected.
    Called at the start of every run after reconcile_positions.

    Returns True when all positions are protected (or only sub-share remainder unprotectable).
    Returns False when a stop attachment attempt fails for a whole-share position — caller
    should treat this as a fatal condition in live mode and write a halt file.
    """
    try:
        positions = client.get_all_positions()
        if not positions:
            return True
        open_orders = client.get_orders()

        # Sum protected qty per symbol across all active sell stop orders
        stop_qty: dict[str, float] = {}
        for o in open_orders:
            if (
                o.order_type in {OrderType.TRAILING_STOP, OrderType.STOP, OrderType.STOP_LIMIT}
                and o.side == OrderSide.SELL
            ):
                stop_qty[o.symbol] = stop_qty.get(o.symbol, 0.0) + float(o.qty or 0)

        all_protected = True
        for pos in positions:
            pos_qty = float(pos.qty)
            covered = stop_qty.get(pos.symbol, 0.0)
            uncovered = pos_qty - covered
            if uncovered > 0.000001:  # tolerance avoids acting on float dust
                whole_uncovered = int(math.floor(uncovered))
                if whole_uncovered < 1:
                    # Sub-share remainder — Alpaca cannot stop-protect this, skip silently
                    continue
                current_price = float(pos.current_price) if pos.current_price else None
                logger.warning(
                    f"Position {pos.symbol}: {pos_qty:.6f} shares, "
                    f"{covered:.6f} covered by stops — attaching stop for {whole_uncovered}"
                )
                result = place_trailing_stop(
                    client, pos.symbol, whole_uncovered, current_price=current_price
                )
                if result is None or not result.is_success:
                    logger.error(
                        f"ensure_stops_attached: FAILED to protect {pos.symbol} "
                        f"({whole_uncovered} shares uncovered)"
                    )
                    all_protected = False
        return all_protected
    except Exception as e:
        logger.error(f"ensure_stops_attached failed: {e}")
        return False


def get_total_open_exposure(client: TradingClient) -> float:
    """Return conservative deployed-capital estimate: market value of filled positions
    plus committed notional of pending (not-yet-filled) buy orders.

    Pending orders for symbols already in positions are skipped to avoid
    double-counting partially-filled orders.

    Raises BrokerStateUnavailable if the broker cannot be queried — callers must
    treat this as trade-blocking, not permissive.
    """
    try:
        total = 0.0
        positions = client.get_all_positions()
        filled_symbols = {p.symbol for p in positions}
        total += sum(float(p.market_value) for p in positions)
        orders = client.get_orders()
        for o in orders:
            if (
                o.side == OrderSide.BUY
                and str(o.status) in ("new", "accepted", "pending_new")
                and o.symbol not in filled_symbols
                and o.notional
            ):
                total += float(o.notional)
        return total
    except BrokerStateUnavailable:
        raise
    except Exception as e:
        raise BrokerStateUnavailable(f"get_total_open_exposure: {e}") from e


def cancel_open_orders(client: TradingClient, symbol: str):
    """Cancel all open orders for a symbol (e.g. before a partial exit)."""
    try:
        orders = client.get_orders()
        for order in orders:
            if order.symbol == symbol and str(order.status) in ("new", "accepted", "pending_new"):
                client.cancel_order_by_id(str(order.id))
                logger.info(f"Cancelled order {order.id} for {symbol}")
    except Exception as e:
        logger.error(f"Failed to cancel orders for {symbol}: {e}")


def place_partial_sell(client: TradingClient, symbol: str, qty: float) -> OrderResult | None:
    """Sell a partial quantity of a position. Polls for fill confirmation."""
    if qty <= 0:
        return None
    try:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 6),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        order_id = str(order.id)
        logger.info(f"Partial SELL: {symbol} qty={qty:.6f} | order_id={order_id}")
        filled_qty = wait_for_fill(client, order_id)
        if filled_qty is not None:
            return OrderResult(
                status=OrderStatus.FILLED,
                symbol=symbol,
                broker_order_id=order_id,
                filled_qty=filled_qty,
            )
        try:
            final = client.get_order_by_id(order_id)
            if str(final.status) == "partially_filled" and final.filled_qty:
                partial_qty = float(final.filled_qty)
                logger.warning(
                    f"Partial sell for {symbol} itself partially filled: {partial_qty} of {qty}"
                )
                return OrderResult(
                    status=OrderStatus.PARTIAL,
                    symbol=symbol,
                    broker_order_id=order_id,
                    filled_qty=partial_qty,
                )
        except Exception:
            pass
        return OrderResult(
            status=OrderStatus.TIMEOUT, symbol=symbol, broker_order_id=order_id, filled_qty=0.0
        )
    except Exception as e:
        logger.error(f"Partial sell failed for {symbol}: {e}")
        return OrderResult(status=OrderStatus.REJECTED, symbol=symbol, rejection_reason=str(e))


# ---------- Daily notional tracking (SQLite) ----------


def get_daily_notional(market_date: str) -> float:
    """Return total confirmed buy notional for the given market date."""
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT buy_notional FROM daily_notional WHERE market_date=?", (market_date,)
            ).fetchone()
        return float(row["buy_notional"]) if row else 0.0
    except Exception:
        return 0.0


def add_daily_notional(market_date: str, amount: float):
    """Increment the confirmed buy notional for the given market date."""
    from datetime import UTC, datetime

    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO daily_notional (market_date, buy_notional, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(market_date) DO UPDATE SET "
                "buy_notional=buy_notional+excluded.buy_notional, updated_at=excluded.updated_at",
                (market_date, amount, datetime.now(UTC).isoformat()),
            )
    except Exception as e:
        logger.warning(f"add_daily_notional failed: {e}")
