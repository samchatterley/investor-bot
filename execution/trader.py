from __future__ import annotations
import fcntl
import json
import logging
import os
from contextlib import contextmanager
from datetime import date
from typing import Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import time
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, IS_PAPER, TRAILING_STOP_PCT, LOG_DIR
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
        result.append({
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc) * 100,  # convert to %
            "market_value": float(p.market_value),
        })
    return result


def place_buy_order(client: TradingClient, symbol: str, notional_usd: float) -> Optional[dict]:
    """Place a fractional market buy, wait for fill, return order including filled_qty."""
    if notional_usd < 1.0:
        logger.warning(f"Order too small for {symbol}: ${notional_usd:.2f}")
        return None

    try:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                notional=round(notional_usd, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        logger.info(f"BUY order placed: {symbol} ${notional_usd:.2f} | order_id={order.id}")

        # Wait up to 10s for fill so we can attach a trailing stop
        filled_qty = wait_for_fill(client, str(order.id))

        return {
            "symbol": symbol,
            "order_id": str(order.id),
            "notional": notional_usd,
            "filled_qty": filled_qty,
            "status": str(order.status),
        }
    except Exception as e:
        logger.error(f"Failed to place BUY for {symbol}: {e}")
        return None


def wait_for_fill(client: TradingClient, order_id: str, max_wait: int = 10) -> Optional[float]:
    """Poll until a market order is filled. Returns filled qty or None on timeout."""
    for _ in range(max_wait):
        try:
            o = client.get_order_by_id(order_id)
            if str(o.status) in ("filled", "partially_filled") and o.filled_qty:
                return float(o.filled_qty)
        except Exception:
            pass
        time.sleep(1)
    logger.warning(f"Order {order_id} did not fill within {max_wait}s")
    return None


def place_trailing_stop(client: TradingClient, symbol: str, qty: float) -> Optional[dict]:
    """Attach a GTC trailing stop to an open position."""
    if not qty or qty <= 0:
        return None
    try:
        order = client.submit_order(
            TrailingStopOrderRequest(
                symbol=symbol,
                qty=round(qty, 6),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=TRAILING_STOP_PCT,
            )
        )
        logger.info(f"Trailing stop: {symbol} {TRAILING_STOP_PCT}% trail | order_id={order.id}")
        return {"symbol": symbol, "trail_pct": TRAILING_STOP_PCT, "order_id": str(order.id)}
    except Exception as e:
        logger.error(f"Failed to place trailing stop for {symbol}: {e}")
        return None


def place_sell_order(client: TradingClient, symbol: str, qty: float) -> Optional[dict]:
    """Sell an entire position (by quantity)."""
    try:
        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 6),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        logger.info(f"SELL order placed: {symbol} qty={qty} | order_id={order.id}")
        return {"symbol": symbol, "order_id": str(order.id), "qty": qty, "status": str(order.status)}
    except Exception as e:
        logger.error(f"Failed to place SELL for {symbol}: {e}")
        return None


def close_position(client: TradingClient, symbol: str) -> Optional[dict]:
    """Close an entire position using Alpaca's close_position helper."""
    try:
        client.close_position(symbol)
        logger.info(f"Position closed: {symbol}")
        return {"symbol": symbol, "status": "closed"}
    except Exception as e:
        logger.error(f"Failed to close position {symbol}: {e}")
        return None


@with_retry(max_attempts=3, delay=2.0, exceptions=(Exception,))
def is_market_open(client: TradingClient) -> bool:
    clock = client.get_clock()
    return clock.is_open


# ---------- Position age tracking ----------

_META_PATH = os.path.join(LOG_DIR, "positions_meta.json")
_META_LOCK_PATH = _META_PATH + ".lock"


@contextmanager
def _meta_lock():
    """Exclusive file lock for read-modify-write operations on positions_meta.json."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(_META_LOCK_PATH, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _load_meta() -> dict:
    if os.path.exists(_META_PATH):
        with open(_META_PATH) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def _save_meta(meta: dict):
    with open(_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def get_position_signal(symbol: str) -> str:
    """Return the entry signal stored for a position, or 'unknown'."""
    with _meta_lock():
        return _load_meta().get(symbol, {}).get("signal", "unknown")


def record_buy(symbol: str, entry_price: float, signal: str = "unknown",
               regime: str = "UNKNOWN", confidence: int = 0):
    with _meta_lock():
        meta = _load_meta()
        meta[symbol] = {
            "entry_date": date.today().isoformat(),
            "entry_price": round(entry_price, 4),
            "signal": signal,
            "regime": regime,
            "confidence": confidence,
        }
        _save_meta(meta)


def record_sell(symbol: str):
    with _meta_lock():
        meta = _load_meta()
        meta.pop(symbol, None)
        _save_meta(meta)


def get_position_meta(symbol: str) -> dict:
    """Return full entry metadata for a position (signal, regime, confidence, entry_price, entry_date)."""
    defaults = {"signal": "unknown", "regime": "UNKNOWN", "confidence": 0, "entry_price": 0.0}
    with _meta_lock():
        return {**defaults, **_load_meta().get(symbol, {})}


def get_position_ages() -> dict[str, int]:
    """Return {symbol: approximate_trading_days_held}."""
    with _meta_lock():
        meta = _load_meta()
    today = date.today()
    ages = {}
    for symbol, data in meta.items():
        try:
            entry = date.fromisoformat(data["entry_date"])
            calendar_days = (today - entry).days
            # Approximate: 5 trading days per 7 calendar days
            trading_days = max(1, round(calendar_days * 5 / 7))
            ages[symbol] = trading_days
        except Exception:
            ages[symbol] = 1
    return ages


def get_stale_positions(max_days: int = 3) -> list[str]:
    """Return symbols held for more than max_days trading days."""
    return [sym for sym, age in get_position_ages().items() if age >= max_days]


def reconcile_positions(client: TradingClient):
    """
    Sync positions_meta.json with actual Alpaca positions.
    Removes stale entries for positions that no longer exist,
    adds placeholder entries for positions with no metadata.
    """
    actual = {p.symbol for p in client.get_all_positions()}
    with _meta_lock():
        meta = _load_meta()
        changed = False

        # Remove entries for positions we no longer hold
        for sym in list(meta.keys()):
            if sym not in actual:
                logger.info(f"Reconcile: removing stale metadata for {sym}")
                del meta[sym]
                changed = True

        # Add placeholder entries for positions with no metadata
        for sym in actual:
            if sym not in meta:
                logger.info(f"Reconcile: adding missing metadata for {sym}")
                meta[sym] = {"entry_date": date.today().isoformat(), "entry_price": 0.0, "signal": "unknown"}
                changed = True

        if changed:
            _save_meta(meta)


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


def place_partial_sell(client: TradingClient, symbol: str, qty: float) -> Optional[dict]:
    """Sell a partial quantity of a position."""
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
        logger.info(f"Partial SELL: {symbol} qty={qty:.6f} | order_id={order.id}")
        return {"symbol": symbol, "order_id": str(order.id), "qty": qty}
    except Exception as e:
        logger.error(f"Partial sell failed for {symbol}: {e}")
        return None
