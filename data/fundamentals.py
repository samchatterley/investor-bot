from __future__ import annotations

import contextlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import yfinance as yf

from config import LOG_DIR

logger = logging.getLogger(__name__)

_FUND_CACHE = os.path.join(LOG_DIR, "fmp_fundamentals_cache.json")
_ANALYST_CACHE = os.path.join(LOG_DIR, "fmp_analyst_cache.json")
_CACHE_TTL_HOURS = 24
_MAX_WORKERS = 5


def _load_cache(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(path: str, cache: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _is_stale(entry: dict) -> bool:
    try:
        age_s = (datetime.now(UTC) - datetime.fromisoformat(entry["fetched_at"])).total_seconds()
        return age_s > _CACHE_TTL_HOURS * 3600
    except (KeyError, ValueError):
        return True


def _fetch_ratios(sym: str) -> tuple[str, dict]:
    try:
        info = yf.Ticker(sym).info
        if not info:
            return sym, {}
        data = {
            "roe": info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
        }
        if all(v is None for v in data.values()):
            return sym, {}
        return sym, data
    except Exception as e:
        logger.debug(f"yfinance ratios {sym}: {e}")
        return sym, {}


# yfinance recommendationKey → (bullish_pct, bearish_pct) approximations.
# yfinance doesn't expose per-category analyst counts, so these are directionally
# correct estimates derived from the consensus label.
_REC_SENTIMENT: dict[str, tuple[int, int]] = {
    "strong_buy": (90, 2),
    "buy": (72, 8),
    "hold": (38, 28),
    "underperform": (18, 52),
    "sell": (8, 72),
    "strong_sell": (4, 88),
}


def _fetch_analyst(sym: str) -> tuple[str, dict]:
    try:
        info = yf.Ticker(sym).info
        if not info:
            return sym, {}

        data: dict = {}

        rec_key = (info.get("recommendationKey") or "").lower()
        n_analysts = info.get("numberOfAnalystOpinions")

        if rec_key in _REC_SENTIMENT:
            bullish, bearish = _REC_SENTIMENT[rec_key]
            data["bullish_pct"] = bullish
            data["bearish_pct"] = bearish

        if n_analysts:
            data["analyst_count"] = int(n_analysts)

        target = info.get("targetMeanPrice")
        if target:
            with contextlib.suppress(ValueError, TypeError):
                data["target_price"] = round(float(target), 2)

        return sym, data
    except Exception as e:
        logger.debug(f"yfinance analyst {sym}: {e}")
        return sym, {}


def get_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Fetch key financial ratios from yfinance with 24-hour cache.

    Returns {symbol: {roe, profit_margin, debt_to_equity, current_ratio}}.
    Individual fields are None when yfinance doesn't report them. Symbols
    with no data are omitted.
    """
    if not symbols:
        return {}

    cache = _load_cache(_FUND_CACHE)
    now_iso = datetime.now(UTC).isoformat()
    result: dict[str, dict] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        entry = cache.get(sym, {})
        if entry and not _is_stale(entry) and entry.get("data"):
            result[sym] = entry["data"]
        else:
            to_fetch.append(sym)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_ratios, sym): sym for sym in to_fetch}
            for fut in as_completed(futures):
                try:
                    sym, data = fut.result()
                    cache[sym] = {"fetched_at": now_iso, "data": data}
                    if data:
                        result[sym] = data
                except Exception as e:
                    logger.debug(f"Fundamentals fetch error: {e}")
        _save_cache(_FUND_CACHE, cache)
        logger.info(f"Fundamentals fetched for {len(to_fetch)} symbol(s)")

    logger.info(f"Fundamentals: {len(result)}/{len(symbols)} symbols")
    return result


def get_analyst_consensus(symbols: list[str]) -> dict[str, dict]:
    """Fetch analyst ratings and price targets from yfinance with 24-hour cache.

    Returns {symbol: {bullish_pct, bearish_pct, analyst_count, target_price}}.
    bullish_pct/bearish_pct are derived from yfinance's recommendationKey label.
    target_price is optional and may be absent.
    """
    if not symbols:
        return {}

    cache = _load_cache(_ANALYST_CACHE)
    now_iso = datetime.now(UTC).isoformat()
    result: dict[str, dict] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        entry = cache.get(sym, {})
        if entry and not _is_stale(entry) and entry.get("data"):
            result[sym] = entry["data"]
        else:
            to_fetch.append(sym)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_analyst, sym): sym for sym in to_fetch}
            for fut in as_completed(futures):
                try:
                    sym, data = fut.result()
                    cache[sym] = {"fetched_at": now_iso, "data": data}
                    if data:
                        result[sym] = data
                except Exception as e:
                    logger.debug(f"Analyst fetch error: {e}")
        _save_cache(_ANALYST_CACHE, cache)
        logger.info(f"Analyst consensus fetched for {len(to_fetch)} symbol(s)")

    logger.info(f"Analyst consensus: {len(result)}/{len(symbols)} symbols")
    return result
