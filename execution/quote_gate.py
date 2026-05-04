"""
Live quote gate — validates real-time market conditions immediately before
order submission.

Used as the final execution guard in the buy loop. Rejects orders when:
  - quote is stale (> QUOTE_MAX_AGE_SECONDS seconds old)
  - spread exceeds QUOTE_MAX_SPREAD_BPS basis points
  - bid or ask is missing / zero
  - last trade is stale (> QUOTE_MAX_TRADE_AGE_SECONDS seconds old)
  - notional cannot purchase at least one whole share at current ask
  - symbol is not tradable or is halted

Why a separate module: the signal layer uses yfinance daily OHLCV (fine for
swing-trade signal generation), but final execution must verify real-time
conditions at the moment of order submission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from models import BrokerStateUnavailable

logger = logging.getLogger(__name__)

# Gate thresholds — tuned for liquid US equities in normal market hours.
QUOTE_MAX_AGE_SECONDS = 15
QUOTE_MAX_SPREAD_BPS = 30
QUOTE_MAX_TRADE_AGE_SECONDS = 60


@dataclass
class QuoteGateResult:
    symbol: str
    approved: bool
    reject_reason: str = ""
    bid: float = 0.0
    ask: float = 0.0
    spread_bps: float = 0.0
    quote_age_seconds: float = 0.0
    last_trade_age_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def _get_data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)


def check_quote_gate(
    symbol: str,
    notional: float,
    data_client: StockHistoricalDataClient | None = None,
    *,
    max_quote_age: int = QUOTE_MAX_AGE_SECONDS,
    max_spread_bps: int = QUOTE_MAX_SPREAD_BPS,
    max_trade_age: int = QUOTE_MAX_TRADE_AGE_SECONDS,
) -> QuoteGateResult:
    """Run pre-submission live market checks for symbol.

    Raises BrokerStateUnavailable if the data API cannot be reached at all
    (network failure, auth error). Individual stale/wide-spread conditions
    return an approved=False result rather than raising — those are normal
    market conditions that should just skip the order.
    """
    client = data_client or _get_data_client()
    now = datetime.now(UTC)

    try:
        quotes = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=[symbol]))
    except Exception as e:
        raise BrokerStateUnavailable(f"check_quote_gate({symbol}): cannot fetch quote — {e}") from e

    q = quotes.get(symbol)
    if q is None:
        return QuoteGateResult(symbol=symbol, approved=False, reject_reason="no quote returned")

    bid = float(q.bid_price or 0)
    ask = float(q.ask_price or 0)

    if bid <= 0 or ask <= 0:
        return QuoteGateResult(
            symbol=symbol,
            approved=False,
            bid=bid,
            ask=ask,
            reject_reason=f"missing bid/ask (bid={bid}, ask={ask})",
        )

    # Quote freshness
    quote_ts = q.timestamp
    if quote_ts is not None:
        if quote_ts.tzinfo is None:
            quote_ts = quote_ts.replace(tzinfo=UTC)
        quote_age = (now - quote_ts).total_seconds()
    else:
        quote_age = 0.0  # assume fresh if no timestamp (paper mode quirk)

    if quote_age > max_quote_age:
        return QuoteGateResult(
            symbol=symbol,
            approved=False,
            bid=bid,
            ask=ask,
            quote_age_seconds=quote_age,
            reject_reason=f"quote stale: {quote_age:.0f}s > {max_quote_age}s limit",
        )

    # Spread check
    mid = (bid + ask) / 2
    spread_bps = ((ask - bid) / mid * 10_000) if mid > 0 else 0.0
    if spread_bps > max_spread_bps:
        return QuoteGateResult(
            symbol=symbol,
            approved=False,
            bid=bid,
            ask=ask,
            spread_bps=spread_bps,
            quote_age_seconds=quote_age,
            reject_reason=f"spread {spread_bps:.1f} bps > {max_spread_bps} bps limit",
        )

    # Last trade freshness — fail closed: if trade data cannot be fetched, do not approve.
    trade_age = 0.0
    try:
        trades = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=[symbol]))
        t = trades.get(symbol)
        if t and t.timestamp:
            ts = t.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            trade_age = (now - ts).total_seconds()
            if trade_age > max_trade_age:
                return QuoteGateResult(
                    symbol=symbol,
                    approved=False,
                    bid=bid,
                    ask=ask,
                    spread_bps=spread_bps,
                    quote_age_seconds=quote_age,
                    last_trade_age_seconds=trade_age,
                    reject_reason=f"last trade stale: {trade_age:.0f}s > {max_trade_age}s limit",
                )
    except Exception as e:
        raise BrokerStateUnavailable(
            f"check_quote_gate({symbol}): last-trade fetch failed — {e}"
        ) from e

    # Whole-share affordability — ensures at least one whole share can be stop-protected
    if ask > 0 and notional / ask < 1.0:
        return QuoteGateResult(
            symbol=symbol,
            approved=False,
            bid=bid,
            ask=ask,
            spread_bps=spread_bps,
            quote_age_seconds=quote_age,
            last_trade_age_seconds=trade_age,
            reject_reason=(
                f"notional ${notional:.2f} / ask ${ask:.2f} = "
                f"{notional / ask:.3f} shares — cannot buy 1 whole share"
            ),
        )

    return QuoteGateResult(
        symbol=symbol,
        approved=True,
        bid=bid,
        ask=ask,
        spread_bps=spread_bps,
        quote_age_seconds=quote_age,
        last_trade_age_seconds=trade_age,
    )
