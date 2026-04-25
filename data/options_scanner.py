from __future__ import annotations
import yfinance as yf
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_MIN_VOLUME = 50  # ignore options with less total volume (too illiquid to signal)


def _get_signal(symbol: str) -> dict | None:
    """
    Fetch the nearest-expiry options chain and extract two signals:
      put_call_ratio  — puts traded / calls traded today.
                        < 0.7 = more calls than puts = bullish lean.
                        > 1.3 = more puts than calls = bearish lean.
      unusual_calls   — True when today's call volume is 1.5× the open interest,
                        suggesting fresh, large-scale buying (informed money).
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return None

        chain = ticker.option_chain(expirations[0])
        calls = chain.calls
        puts = chain.puts
        if calls.empty or puts.empty:
            return None

        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol = float(puts["volume"].fillna(0).sum())
        call_oi = float(calls["openInterest"].fillna(0).sum())

        if (call_vol + put_vol) < _MIN_VOLUME:
            return None

        pc_ratio = put_vol / (call_vol + 1)
        unusual_calls = bool(call_oi > 0 and call_vol > call_oi * 1.5)

        return {
            "put_call_ratio": round(pc_ratio, 2),
            "unusual_calls": unusual_calls,
        }
    except Exception as e:
        logger.debug(f"Options unavailable for {symbol}: {e}")
        return None


def get_options_signals(symbols: list[str]) -> dict:
    """Parallel fetch for a list of symbols. Returns {symbol: signal_dict}."""
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_get_signal, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                if result:
                    results[sym] = result
            except Exception as e:
                logger.debug(f"Options future failed for {sym}: {e}")
    return results
